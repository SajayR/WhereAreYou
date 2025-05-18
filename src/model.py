# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import (
    HubertModel, 
    AutoProcessor, 
    AutoTokenizer, 
    AutoModel
)
import math
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
from PIL import Image
from torch.cuda.amp import autocast

from peft import (
    LoraConfig, 
    get_peft_model,
    TaskType,
)


#################################################################
#                   Text Embedder
#################################################################
class TextEmbedder(nn.Module):
    """
    pre-trained BERT-like model to extract text features.
    Projects them down to a desired embedding dimension.
    """
    def __init__(self, embedding_dim=512, model_name="answerdotai/ModernBERT-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection1 = nn.Linear(self.encoder.config.hidden_size, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.projection2 = nn.Linear(512, embedding_dim)
        print("Using text model: ", model_name)
        
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True
        
    def forward(self, text_list):
        """
        Args:
            text_list: List[str], batch of text inputs
            
        Returns:
            text_feats: (B, Nt, D)
            attention_mask: (B, Nt)
        """
        inputs = self.tokenizer(
            text_list, 
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=128,
            return_tensors="pt"
        )
        device = next(self.parameters()).device
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        outputs = self.encoder(**inputs)  # (B, Nt, hidden_size)
        hidden_states = outputs.last_hidden_state
        text_feats = self.projection2(self.layer_norm(self.projection1(hidden_states)))  # (B, Nt, D)
        
        return text_feats, inputs["attention_mask"]


#################################################################
#                   Visual Embedder
#################################################################
class ViTEmbedder(nn.Module):
    """
    DINOv2to extract patch embeddings from an image.
    Then projects to a common dimension with a linear layer.
    """
    def __init__(self, model_name='facebookresearch/dinov2', arch='dinov2_vitb14',
                 embedding_dim=512, dropout_prob=0.1):
        super().__init__()
        self.model = torch.hub.load(model_name, arch)
        print("Using DINOv2 model: ", arch)
        self.projection1 = nn.Linear(self.model.embed_dim, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.projection2 = nn.Linear(512, embedding_dim)
        #self.dropout = nn.Dropout(dropout_prob)
        self.patch_dropout_rate = dropout_prob
        self.patch_dropout = self.patch_dropout
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True

    def patch_dropout(self, x, drop_rate):
        """
        Actually removes patch embeddings during training
        Args:
            x: patch embeddings of shape (B, N, D) where N is number of patches
            drop_rate: probability of dropping a patch
        """
        if not self.training or drop_rate == 0:
            return x
            
        B, N, D = x.shape
        dtype = x.dtype
            
        # Create keep mask
        keep_mask = torch.bernoulli(
            torch.ones(B, N, device=x.device, dtype=dtype) * (1 - drop_rate)
        ).bool()
        
        # List to store processed batch items
        output_tensors = []
        
        # Process each item in batch
        for i in range(B):
            # Select tokens to keep for this batch item
            kept_tokens = x[i][keep_mask[i]]
            output_tensors.append(kept_tokens)
        
        # Pad sequences to longest in batch
        max_len = max(tensor.size(0) for tensor in output_tensors)
        padded_outputs = []
        
        for tensor in output_tensors:
            if tensor.size(0) < max_len:
                padding = torch.zeros(max_len - tensor.size(0), D, dtype=dtype, device=x.device)
                padded_outputs.append(torch.cat([tensor, padding], dim=0))
            else:
                padded_outputs.append(tensor)
        
        # Stack back into batch
        x = torch.stack(padded_outputs, dim=0)
        return x

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W), e.g. (B,3,224,224) image batch
        Returns:
            visual_feats: (B, Nv, D)
                Nv = number of visual tokens
                D  = embedding_dim
        """
        #print(f"x shape: {x.shape}")
        if len(x.shape) == 5:  # shape: [1, 1, 3, 224, 224]
            x = x.squeeze(0)  # get [1, 3, 224, 224]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        patches = self.model.get_intermediate_layers(x, n=1)[0]  
        
        feats = self.projection2(self.layer_norm(self.projection1(patches)))
        #feats = self.dropout(feats)
        feats = self.patch_dropout(feats, self.patch_dropout_rate)
        #print(f"feats shape: {feats.shape}")
        return feats

class ViTLoRAEmbedder(nn.Module):
    """
    DINOv2 with LoRA adapters for parameter-efficient fine-tuning.
    Applies LoRA to both attention and MLP layers of the transformer.
    Projects output embeddings to a common dimension with a linear layer.
    """
    def __init__(self, model_name='facebookresearch/dinov2', arch='dinov2_vitb14',
                 embedding_dim=512, dropout_prob=0.1, lora_rank=8, lora_alpha=16):
        super().__init__()
        
        # Load the base model
        self.model = torch.hub.load(model_name, arch)
        print(f"Using DINOv2 model with LoRA adapters: {arch}")
    
        
        # Freeze the base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Define which layers to apply LoRA to - now including both attention and MLP layers
        lora_target_modules = [
            # Attention layers
            "attn.qkv",
            "attn.proj",
            # MLP layers
            #"mlp.fc1",
            #"mlp.fc2",
        ]
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.0,
            fan_in_fan_out=True,  # Add this
            bias="none",          # Add this
            modules_to_save=None  # Add this to prevent parameter duplication
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"ViTLoRAEmbedder - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")

        self.projection1 = nn.Linear(self.model.embed_dim, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.projection2 = nn.Linear(512, embedding_dim)
        self.patch_dropout_rate = dropout_prob
        self.patch_dropout = self.patch_dropout
        for param in self.model.parameters():
            param.requires_grad = True
        #set base model to false
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True

    def patch_dropout(self, x, drop_rate):
        """
        Actually removes patch embeddings during training
        Args:
            x: patch embeddings of shape (B, N, D) where N is number of patches
            drop_rate: probability of dropping a patch
        """
        if not self.training or drop_rate == 0:
            return x
            
        B, N, D = x.shape
        dtype = x.dtype
            
        # Create keep mask
        keep_mask = torch.bernoulli(
            torch.ones(B, N, device=x.device, dtype=dtype) * (1 - drop_rate)
        ).bool()
        
        # List to store processed batch items
        output_tensors = []
        
        # Process each item in batch
        for i in range(B):
            # Select tokens to keep for this batch item
            kept_tokens = x[i][keep_mask[i]]
            output_tensors.append(kept_tokens)
        
        # Pad sequences to longest in batch
        max_len = max(tensor.size(0) for tensor in output_tensors)
        padded_outputs = []
        
        for tensor in output_tensors:
            if tensor.size(0) < max_len:
                padding = torch.zeros(max_len - tensor.size(0), D, dtype=dtype, device=x.device)
                padded_outputs.append(torch.cat([tensor, padding], dim=0))
            else:
                padded_outputs.append(tensor)
        
        # Stack back into batch
        x = torch.stack(padded_outputs, dim=0)
        return x

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W), e.g. (B,3,224,224) image batch
        Returns:
            visual_feats: (B, Nv, D)
                Nv = number of visual tokens
                D  = embedding_dim
        """
        if len(x.shape) == 5:  # shape: [1, 1, 3, 224, 224]
            x = x.squeeze(0)  # get [1, 3, 224, 224]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Use intermediate layers for feature extraction - same as original
        patches = self.model.get_intermediate_layers(x, n=1)[0]
        feats = self.projection2(self.layer_norm(self.projection1(patches)))
        feats = self.patch_dropout(feats, self.patch_dropout_rate)
        
        return feats

#################################################################
#                   Unified MultiModalModel
#################################################################
class MultiModalModel(nn.Module):
    def __init__(
        self, 
        text_model_name="distilbert/distilbert-base-uncased",
        temperature=1.2,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.1,
        use_amp=True,
        num_heads: int = 2,              # NEW
        disentangle_weight: float = 0.05 # NEW
    ):
        super().__init__()

        self.text_embedder  = TextEmbedder(embedding_dim=512, model_name=text_model_name)
        self.visual_embedder = ViTLoRAEmbedder(arch='dinov2_vitb14_reg', embedding_dim=512, dropout_prob=visual_dropout_prob)
        #self.visual_embedder = ViTEmbedder(arch='dinov2_vitb14', embedding_dim=512, dropout_prob=visual_dropout_prob)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        assert 512 % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.num_heads         = num_heads
        self.disentangle_w     = disentangle_weight

        self.patch_sparsity_threshold = patch_sparsity_threshold
        self.patch_sparsity_weight = patch_sparsity_weight
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16
    ######################################################
    #               Shared Utilities
    ######################################################
    def compute_similarity_matrix(self, feats1, feats2):
        """
        Generic token-level dot-product similarity between feats1 and feats2.
        feats1: (B, N1, D)
        feats2: (B, N2, D)
        Returns sim: (B, N1, N2)
        """ 
        # ONLY NORMALIZE DURING INFERENCE, TRAINING CAUSES MODEL COLLAPSE
        #feats1 = F.normalize(feats1, dim=-1)
        #feats2 = F.normalize(feats2, dim=-1)
        #always run in full precision
        with torch.cuda.amp.autocast(enabled=False):
            B, N1, D = feats1.shape
            N2       = feats2.shape[1]
            Dh       = D // self.num_heads

            f1 = feats1.view(B, N1, self.num_heads, Dh).permute(0, 2, 1, 3)  # B,H,N1,Dh
            f2 = feats2.view(B, N2, self.num_heads, Dh).permute(0, 2, 1, 3)  # B,H,N2,Dh
            sims_h = torch.matmul(f1, f2.transpose(-2, -1)) * self.temperature  # B,H,N1,N2
            sims   = sims_h.max(dim=1).values                                    # max over heads
            return sims


    ######################################################
    #               TEXT-VISUAL PATH
    ######################################################
    def compute_all_similarities_tv(self, text_feats, visual_feats, attention_mask):
        """
        Returns:
            clip_sims        : (B,B)    – scalar similarity per pair
            token_sims_agg   : (B,B,Nt,Nv)  – after max over heads
            token_sims_full  : (B,B,H,Nt,Nv) – before max (for regularisers)
        """
        B, Nt, D  = text_feats.shape
        Nv        = visual_feats.shape[1]
        Dh        = D // self.num_heads

        # add head dim
        t = text_feats.view(B, Nt, self.num_heads, Dh)
        v = visual_feats.view(B, Nv, self.num_heads, Dh)

        # cross-batch expand
        t = t.unsqueeze(1).expand(-1, B, -1, -1, -1)  # B,B,Nt,H,Dh
        v = v.unsqueeze(0).expand(B, -1, -1, -1, -1)  # B,B,Nv,H,Dh

        token_sims_full = torch.einsum('ijthd,ijvhd->ijhtv', t, v) * self.temperature   # B,B,H,Nt,Nv
        token_sims_agg  = token_sims_full.max(dim=2).values            # max over heads  -> B,B,Nt,Nv
        max_spat        = token_sims_agg.max(dim=3).values             # max over Nv     -> B,B,Nt

        mask = attention_mask.unsqueeze(1).float().expand(-1, B, -1)   # B,B,Nt
        masked_sum   = (max_spat * mask).sum(dim=2)                    # B,B
        valid_tokens = mask.sum(dim=2).clamp(min=1e-7)                 # B,B
        clip_sims    = masked_sum / valid_tokens                       # B,B

        return clip_sims, token_sims_agg, token_sims_full


    def compute_regularization_losses_tv(self,
                                       token_sims_agg,
                                       token_sims_full=None):
        """
        1) non-negativity on negatives
        2) patch-usage sparsity on positives
        3) (optional) head disentanglement
        """
        B = token_sims_agg.shape[0]
        neg_sims = torch.clamp(token_sims_agg, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims ** 2)

        # ---------- sparsity on positive pairs ----------
        positive_sims = [token_sims_agg[i, i] for i in range(B)]  # (Nt,Nv)
        if not positive_sims:                                     # unlikely
            return 0.15 * l_nonneg
        positive_sims = torch.stack(positive_sims, dim=0)
        # softmax over patches => (B, Nt, Nv)
        patch_probs = F.softmax(positive_sims, dim=-1)
        # fraction usage per patch => sum over Nt, then / Nt => (B, Nv)
        patch_fraction = patch_probs.sum(dim=1) / patch_probs.shape[1]
        # penalize if fraction > threshold
        excess = F.relu(patch_fraction - self.patch_sparsity_threshold)  # (B, Nv)
        loss_sparsity = (excess ** 2).mean()
        reg_loss = 0.15 * l_nonneg + self.patch_sparsity_weight * loss_sparsity

        # ---------- disentanglement (H=2 only) ----------
        if (token_sims_full is not None) and (self.num_heads >= 2):
            pos_full = torch.stack([token_sims_full[i, i] for i in range(B)], dim=0)  # B,H,Nt,Nv
            head0, head1 = pos_full[:, 0], pos_full[:, 1]                             # B,Nt,Nv each
            l_dis = torch.mean(torch.abs(head0 * head1))
            reg_loss = reg_loss + self.disentangle_w * l_dis

        # temperature floor (unchanged)
        temp_low = torch.clamp(torch.log(torch.tensor(1.0, device=token_sims_agg.device))
                               - torch.log(self.temperature), min=0) ** 2
        return reg_loss + temp_low

    def compute_contrastive_loss_tv(self, clip_sims, token_sims_agg, token_sims_full):
        """
        Standard cross-entropy for text<->visual plus the reg losses,
        now with similarity statistics tracking.
        """
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)
        
        # Extract positive and negative similarities
        pos_sims = torch.diagonal(clip_sims)  # Shape: (B,)
        
        # Create a mask for negative pairs (all except diagonal)
        mask = torch.ones_like(clip_sims, dtype=torch.bool)
        mask.fill_diagonal_(0)
        neg_sims = clip_sims[mask]  # Shape: (B*(B-1),)
        
        # Compute statistics
        pos_sim_mean = pos_sims.mean().item()
        pos_sim_std = pos_sims.std().item()
        neg_sim_mean = neg_sims.mean().item()
        neg_sim_std = neg_sims.std().item()
        hardest_negative = neg_sims.max().item()
        
        # Calculate separation (gap between positive and negative means)
        separation = pos_sim_mean - neg_sim_mean
        
        # Original contrastive loss calculation
        # text->visual
        log_prob_t2v = F.log_softmax(clip_sims, dim=1)
        losses_t2v = -log_prob_t2v[torch.arange(B), labels]

        # visual->text
        log_prob_v2t = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2t = -log_prob_v2t[torch.arange(B), labels]

        contrastive_loss = (losses_t2v + losses_v2t).mean() / 2
        reg_loss = self.compute_regularization_losses_tv(token_sims_agg, token_sims_full)

        total_loss = contrastive_loss + reg_loss
        
        # Return similarity stats along with losses
        similarity_stats = {
            "tv_pos_sim_mean": pos_sim_mean,
            "tv_pos_sim_std": pos_sim_std,
            "tv_neg_sim_mean": neg_sim_mean,
            "tv_neg_sim_std": neg_sim_std,
            "tv_separation": separation,
            "tv_hardest_negative": hardest_negative
        }
        
        return total_loss, similarity_stats

    def forward_text_visual(self, frames, text_list):
        """
        frames: (B, 3, 224, 224)
        text_list: list of strings length B

        If training: return scalar contrastive loss
        else: return (sim_matrix, attention_mask)
        """
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):   
            visual_feats = self.visual_embedder(frames)              # (B, Nv, D)
            text_feats, attention_mask = self.text_embedder(text_list) # (B, Nt, D), (B, Nt)
            #visual_feats = F.normalize(visual_feats, dim=-1)
            #text_feats = F.normalize(text_feats, dim=-1)
            clip_sims, tok_agg, tok_full = self.compute_all_similarities_tv(
                                              text_feats, visual_feats, attention_mask)
            return self.compute_contrastive_loss_tv(clip_sims, tok_agg, tok_full)


    def forward(self, frames=None, text_list=None):
        assert frames is not None or text_list is not None, "At least one modality must be provided"
        # we need to conver the image into the correct format and shit
        assert frames is not str, "Frames Cross-modal retrieval using 1000 evaluation videos from the PlacesAudio and AudioSet validation datasets. DenseAV dramatically outperforms all approaches tested in all metrics. Most notably, the state-of-the-art image retrieval foundation model, ImageBind, is incapable of recognizing speech. We note that the ImageBind authors do not publish retraining code, so we evaluate their largest pretrained model. Models with a * indicate that they have been previously reported in the literature. Other numbers are calculated by using pretrained models when available or from training with the author’s official training scripts.should be a path to an image"
        if frames is not None:
            image = Image.open(frames).convert('RGB')
            transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
            ])
            frames = transform(image)
        embeddings = {}
        if frames is not None:
            embeddings['visual_feats'] = self.visual_embedder(frames)
        if text_list is not None:
            embeddings['text_feats'], _ = self.text_embedder(text_list)

        # if two or more modalities are present, we compute the similarity matrix 
        if frames is not None and text_list is not None:
            embeddings['vis_text_sim_matrix'] = self.compute_similarity_matrix(embeddings['text_feats'], embeddings['visual_feats'])
        return embeddings

#################################################################
#                        Quick Test
#################################################################
if __name__ == "__main__":
    print("Testing MultiModalModel with random inputs...")
    model = MultiModalModel(
        audio_model_name="facebook/hubert-base-ls960",
        text_model_name="distilbert/distilbert-base-uncased",
        temperature=2.0,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.2
    )
    batch_size = 2
    dummy_frames = torch.randn(batch_size, 3, 224, 224)      # image frames
    dummy_audio  = torch.randn(batch_size, 16000)            # 1 sec of 16kHz
    dummy_texts  = ["a man riding a bicycle", "a cat on a bed"]
    model.train()
    av_loss, _, _, _, _ = model.forward_audio_visual(dummy_frames, dummy_audio)
    print(f"Audio-Visual loss: {av_loss.item():.4f}")
    tv_loss, _ = model.forward_text_visual(dummy_frames, dummy_texts)
    print(f"Text-Visual loss: {tv_loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        av_sims = model.forward_audio_visual(dummy_frames, dummy_audio)
        print(f"Audio-Visual similarities shape: {av_sims.shape}")  
        # expected => (B, Na, Nv)
        tv_sims, tv_mask = model.forward_text_visual(dummy_frames, dummy_texts)
        print(f"Text-Visual similarities shape: {tv_sims.shape}, mask: {tv_mask.shape}")
        # expected => (B, Nt, Nv), (B, Nt)
    
    print("MultiModalModel test completed.")
