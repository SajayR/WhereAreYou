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
import torch._dynamo
 


class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim=256, model_name="answerdotai/ModernBERT-base", lora_rank=16, lora_alpha=32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,  
            lora_alpha=lora_alpha,  
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin"]  
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.projection1 = nn.Linear(self.encoder.config.hidden_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.projection2 = nn.Linear(256, embedding_dim)
        print("Using text model: ", model_name)
        for name, param in self.encoder.named_parameters():
            if 'lora_' not in name:  # If not a LoRA parameter
                param.requires_grad = False
            else:
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

class ViTEmbedder(nn.Module):

    def __init__(self, model_name='facebookresearch/dinov2', arch='dinov2_vitb14',
                 embedding_dim=256, dropout_prob=0.1, lora_rank=16, lora_alpha=32):
        super().__init__()

        self.model = torch.hub.load(model_name, arch)
        print(f"Using DINOv2 model with LoRA adapters: {arch}")
        for param in self.model.parameters():
            param.requires_grad = False

        lora_target_modules = [
            "attn.qkv",
            "attn.proj",
            #"mlp.fc1",
            #"mlp.fc2",
        ]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.1,
            fan_in_fan_out=True,  
            bias="none",          
            modules_to_save=None,  
            
        )

        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"ViTLoRAEmbedder - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")

        self.projection1 = nn.Linear(self.model.embed_dim, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.projection2 = nn.Linear(256, embedding_dim)
        self.patch_dropout_rate = dropout_prob
        self.patch_dropout = self.patch_dropout_layer

        for name, param in self.model.named_parameters():
            if 'lora_' not in name:  # If not a LoRA parameter
                param.requires_grad = False
            else:
                param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True

    def patch_dropout(self, x: torch.Tensor, drop_p: float):
        """
        x : (B, N, D) 
        """
        if not self.training or drop_p == 0:
            return x

        B, N, D = x.shape
        # keep_mask shape (B, N, 1)
        keep_mask = (torch.rand(B, N, 1, device=x.device) > drop_p)
        x = x * keep_mask          # zero-out dropped patches
        # (optional) renormalise so token magnitudes stay comparable
        keep_counts = keep_mask.sum(dim=1, keepdim=True).clamp_min(1)
        x = x * N / keep_counts
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
        if len(x.shape) == 5:  # [1, 1, 3, 224, 224]
            x = x.squeeze(0)  #  [1, 3, 224, 224]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
 
        patches = self.model.get_intermediate_layers(x, n=1)[0]
        feats = self.projection2(self.layer_norm(self.projection1(patches)))
        feats = self.patch_dropout(feats, self.patch_dropout_rate)
        
        return feats

class DuoDModel(nn.Module):
    def __init__(
        self, 
        text_model_name="distilbert/distilbert-base-uncased",
        temperature=1.2,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.1,
        use_amp=True,
        vit_lora_rank=16,
        vit_lora_alpha=32,
        text_lora_rank=16,
        text_lora_alpha=32
    ):
        super().__init__()

        self.visual_embedder = ViTEmbedder(arch='dinov2_vitb14_reg', embedding_dim=256, dropout_prob=visual_dropout_prob, lora_rank=vit_lora_rank, lora_alpha=vit_lora_alpha)
        self.text_embedder = TextEmbedder(embedding_dim=256, model_name=text_model_name, lora_rank=text_lora_rank, lora_alpha=text_lora_alpha)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.patch_sparsity_threshold = patch_sparsity_threshold
        self.patch_sparsity_weight = patch_sparsity_weight
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16

    def compute_similarity_matrix(self, feats1, feats2):
        """
        token-level dot-product similarity between feats1 and feats2.
        feats1: (B, N1, D)
        feats2: (B, N2, D)
        Returns sim: (B, N1, N2)
        """ 
        # ONLY NORMALIZE DURING INFERENCE, TRAINING CAUSES MODEL COLLAPSE
        #feats1 = F.normalize(feats1, dim=-1)
        #feats2 = F.normalize(feats2, dim=-1)
        
        feats1_f32 = feats1.float()
        feats2_f32 = feats2.float()
        with torch.cuda.amp.autocast(enabled=False):
            sim = torch.bmm(feats1_f32, feats2_f32.transpose(1, 2))
            return (sim * self.temperature.float()).float()


    def compute_all_similarities_tv(self,
                                    text_feats:  torch.Tensor,
                                    visual_feats: torch.Tensor,
                                    attention_mask: torch.Tensor):
        """
        Hard max over patches, mean over tokens
        Bidirectional max-mean aggregation (t → v and v → t).

        Parameters
        ----------
        text_feats      : (B, Nt, D)   text-token embeddings
        visual_feats    : (B, Nv, D)   visual-patch embeddings
        attention_mask  : (B, Nt)      1 for real tokens, 0 for padding

        Returns
        -------
        clip_sims  : (B, B)             symmetric similarity matrix
        token_sims : (B, B, Nt, Nv)     raw token-level similarities
        """
        B = text_feats.size(0)

        # Broadcast so we can compare every text in the batch with every image

        #text_feats = F.normalize(text_feats, dim=-1)
        #visual_feats = F.normalize(visual_feats, dim=-1)
        tf = text_feats.unsqueeze(1).expand(-1, B, -1, -1)          # (B, B, Nt, D)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)        # (B, B, Nv, D)

        # Full token-level similarity tensor
        token_sims = torch.matmul(tf, vf.transpose(2, 3))           # (B, B, Nt, Nv)
        token_sims = token_sims * self.temperature                   # scale by learned temp

        # ------------------------------------------------------------
        # 1)  text → visual • max over patches, mean over tokens
        # ------------------------------------------------------------
        t2v_max  = token_sims.max(dim=3).values                      # (B, B, Nt)
        t_mask   = attention_mask.unsqueeze(1).float().expand(-1, B, -1)
        t2v_sum  = (t2v_max * t_mask).sum(dim=2)                     # (B, B)
        valid_t  = t_mask.sum(dim=2).clamp(min=1e-7)
        t2v_clip = t2v_sum / valid_t                                 # (B, B)

        # ------------------------------------------------------------
        # 2)  visual → text • max over tokens, mean over patches
        # ------------------------------------------------------------
        v2t_max  = token_sims.max(dim=2).values                      # (B, B, Nv)
        v2t_clip = v2t_max.mean(dim=2)                               # (B, B)

        clip_sims = 0.5 * (t2v_clip + v2t_clip)                      # (B, B)

        return clip_sims, token_sims

    def compute_regularization_losses_tv(self, token_sims):
 
        # (B, B, Nt, Nv)
        B = token_sims.shape[0]
        # negative clamp
        neg_sims = torch.clamp(token_sims, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims**2)
        #temp constraints
        temp = self.temperature
        temp_low = torch.clamp(torch.log(torch.tensor(1.0, device=token_sims.device)) 
                               - torch.log(temp), min=0) ** 2
        temp_high = torch.clamp(torch.log(temp) - torch.log(torch.tensor(2.0, device=token_sims.device)), min=0) ** 2
        l_cal = temp_low + temp_high
        return l_nonneg + 0.4*l_cal

        
    def compute_contrastive_loss_tv(self, clip_sims, token_sims): #infonce
        
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)
        
        # Extract positive and negative sims
        pos_sims = torch.diagonal(clip_sims)  # Shape: (B,)
        
        # Create a mask for negative pairs (all except diagonal)
        mask = torch.ones_like(clip_sims, dtype=torch.bool)
        mask.fill_diagonal_(0)
        neg_sims = clip_sims[mask]  # Shape: (B*(B-1),)
        pos_sim_mean = pos_sims.mean().item()
        pos_sim_std = pos_sims.std().item()
        neg_sim_mean = neg_sims.mean().item()
        neg_sim_std = neg_sims.std().item()
        hardest_negative = neg_sims.max().item()
        # gap between positive and negative means
        separation = pos_sim_mean - neg_sim_mean
        
        # text->visual
        log_prob_t2v = F.log_softmax(clip_sims, dim=1)
        losses_t2v = -log_prob_t2v[torch.arange(B), labels]

        # visual->text
        log_prob_v2t = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2t = -log_prob_v2t[torch.arange(B), labels]

        contrastive_loss = (losses_t2v + losses_v2t).mean() / 2
        reg_loss = self.compute_regularization_losses_tv(token_sims)

        total_loss = contrastive_loss + reg_loss
        
        similarity_stats = {
            "tv_pos_sim_mean": pos_sim_mean,
            "tv_pos_sim_std": pos_sim_std,
            "tv_neg_sim_mean": neg_sim_mean,
            "tv_neg_sim_std": neg_sim_std,
            "tv_separation": separation,
            "tv_hardest_negative": hardest_negative,
            "tv_contrastive_loss": contrastive_loss,
            "tv_reg_loss": reg_loss
        }
        
        return total_loss, similarity_stats



    def forward_text_visual(self, frames, text_list):
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            visual_feats = self.visual_embedder(frames)           # (B, Nv, D)
            B = visual_feats.size(0)
                      
            text_feats, attn_mask = self.text_embedder(text_list)  # (B, Nt, D)
            clip_sims, token_sims = self.compute_all_similarities_tv(
                text_feats, visual_feats, attn_mask
            )
            loss, stats = self.compute_contrastive_loss_tv(clip_sims, token_sims)
            return loss, stats

    def forward(self, frames=None, text_list=None):
        assert frames is not None or text_list is not None, "At least one modality must be provided"
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


if __name__ == "__main__":
    print("Testing MultiModalModel with random inputs...")
    model = DuoDModel(
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
