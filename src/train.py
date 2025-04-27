import os
import gc
import math
import wandb
import torch
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
import torch.nn as nn
from dataset import LocalCaptionDataset
from model import MultiModalModel
from viz import TextVisualizer
from retrieval import compute_tv_retrieval_metrics
warnings.filterwarnings("ignore")


###########################################
#         Collate for the Text Dataset
###########################################
def collate_text_fn(batch):
    """
    Simple collate function for the LocalCaptionDataset.
    Returns:
    {
       'images': (B, 3, 224, 224),
       'captions': list[str] of length B
    }
    """
    images, captions = zip(*batch)  # from the dataset's __getitem__
    images = torch.stack(images)    # shape => (B, 3, 224, 224)
    return {
        'images': images,
        'captions': list(captions)
    }


###########################################
#         Trainer Class (Text-Visual Only)
###########################################
class MultiModalTrainer:
    """
    Trainer for text-visual training.
    Uses separate optimizers and schedulers for the non-specialized parameters,
    for the text encoder and for the vision (LoRA) parameters.
    """
    def __init__(
        self,
        text_dataset_path: str,
        output_dir: str = "./outputs_text",
        batch_size_tv: int = 32,
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        unfreeze_text_step: int = 5000,
        unfreeze_vit_step: int = 8000,
        use_wandb: bool = True,
        project_name: str = "UnifiedTextVisual",
        vis_every: int = 1000,
        num_vis_samples_tv: int = 10,
        save_every_steps: int = 10000,
        num_workers: int = 4,
        device: str = "cuda",
        force_new_training: bool = False,
        use_amp: bool = True,
        # Validation parameters
        text_dataset_val_path: str = None,
        validation_frequency: int = 10000,  # Run validation every N steps
        # New temperature scheduling parameters
        aggregator_temp_start: float = 4.0,
        aggregator_temp_end: float = 0.1,
        aggregator_temp_steps: int = 150000,
        run_name: str = "baseline",
        idf_path: str = None,
    ):
        """
        Args:
            text_dataset_path:      path to images & text files (LocalCaptionDataset)
            output_dir:             where to save logs and checkpoints
            batch_size_tv:          batch size for text-visual training
            num_epochs:             number of epochs to train
            learning_rate:          initial learning rate
            gradient_accumulation_steps: steps for gradient accumulation
            unfreeze_text_step:     step to unfreeze the text encoder
            unfreeze_vit_step:      step to unfreeze the vision (LoRA) parameters
            use_wandb:              whether to use wandb for logging
            project_name:           name for the wandb project
            vis_every:              frequency (in steps) for visualizing samples
            num_vis_samples_tv:     number of text-visual samples for visualization
            save_every_steps:       checkpoint saving frequency in steps
            num_workers:            number of workers for dataloader
            device:                 device identifier string (e.g., "cuda")
            force_new_training:     if True, starts a new training run (ignores checkpoints)
            text_dataset_val_path:  path for validation LocalCaptionDataset (optional)
            validation_frequency:   frequency (in steps) for validation
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.vis_every = vis_every
        self.save_every_steps = save_every_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validation_frequency = validation_frequency

        self.aggregator_temp_start = aggregator_temp_start
        self.aggregator_temp_end = aggregator_temp_end
        self.aggregator_temp_steps = aggregator_temp_steps

        self.config = {
            "batch_size_tv": batch_size_tv,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "unfreeze_text_step": unfreeze_text_step,
            "unfreeze_vit_step": unfreeze_vit_step,
            "vis_every": vis_every,
            "save_every_steps": save_every_steps,
            "validation_frequency": validation_frequency,
            "aggregator_temp_start": aggregator_temp_start,
            "aggregator_temp_end": aggregator_temp_end,
            "aggregator_temp_steps": aggregator_temp_steps,
            "run_name": run_name
        }
        logging.basicConfig(
            filename=str(self.output_dir / 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.use_amp = use_amp

        # -----------------------------------------------------
        #  1) Datasets / Dataloaders
        # -----------------------------------------------------
        print("Loading LocalCaptionDataset...")
        self.tv_dataset = LocalCaptionDataset(text_dataset_path)
        self.tv_dataloader = DataLoader(
            self.tv_dataset,
            batch_size=batch_size_tv,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_text_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=8,
            drop_last=True
        )
        print("LocalCaptionDataset loaded")
        self.tv_iter = None

        # -----------------------------------------------------
        #  1.1) Validation Dataset / Dataloader
        # -----------------------------------------------------
        self.val_tv_dataset = None
        self.val_tv_dataloader = None

        if text_dataset_val_path:
            print("Loading Validation LocalCaptionDataset...")
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.val_tv_dataset = LocalCaptionDataset(
                text_dataset_val_path,
                transform=val_transform
            )
            self.val_tv_dataloader = DataLoader(
                self.val_tv_dataset,
                batch_size=batch_size_tv,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_text_fn,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=8,
                drop_last=True
            )
            print("Validation LocalCaptionDataset loaded")

        # -----------------------------------------------------
        #  2) Model (Text & Vision Only)
        # -----------------------------------------------------
        self.model = MultiModalModel(
            text_model_name="distilbert/distilbert-base-uncased",
            temperature=1.5,
            patch_sparsity_threshold=0.80,
            patch_sparsity_weight=0.00,
            visual_dropout_prob=0.25,
            use_amp=use_amp,
            aggregator_temp=self.aggregator_temp_start,
            idf_path=idf_path
        ).to(self.device)
        #enabling gradient checkpointing
        #self.model.audio_embedder.hubert.gradient_checkpointing_enable()
        #self.model.text_embedder.encoder.gradient_checkpointing_enable()
        #1self.model.visual_embedder.model.gradient_checkpointing = True

        # -----------------------------------------------------
        #  3) Separate param groups => separate optimizers
        # -----------------------------------------------------
        self.text_params = []
        self.vit_lora_params = []
        self.vit_params = []
        self.others_params = []
        for name, param in self.model.named_parameters():
            if "text_embedder.encoder" in name:
                self.text_params.append(param)
            elif "visual_embedder.model" in name and "lora" in name:
                self.vit_lora_params.append(param)
            elif "visual_embedder.model" in name and "lora" not in name:
                self.vit_params.append(param)
            else:
                self.others_params.append(param)

        print(f"Number of LoRA parameters: {len(self.vit_lora_params)}")
        print(f"Number of ViT parameters: {len(self.vit_params)}")
        if len(self.vit_lora_params) == 0:
            print("WARNING: No LoRA parameters found! Check implementation.")

        # Optimizer for "others"
        self.opt_others = torch.optim.AdamW(self.others_params, lr=learning_rate)
        # Optimizer for text encoder (frozen at start)
        self.opt_text = torch.optim.AdamW(self.text_params, lr=learning_rate)
        # Optimizer for vision (LoRA) parameters
        self.opt_vit = torch.optim.AdamW(self.vit_lora_params, lr=learning_rate)

        # Freeze text parameters until reaching the unfreeze step
        for p in self.text_params:
            p.requires_grad = False
        for p in self.vit_lora_params:
            p.requires_grad = True
        for p in self.vit_params:
            p.requires_grad = False

        # -----------------------------------------------------
        #  4) Multiple OneCycle schedulers
        # -----------------------------------------------------
        total_steps_per_epoch = len(self.tv_dataloader)
        self.steps_per_epoch = total_steps_per_epoch
        self.total_updates = (total_steps_per_epoch * num_epochs) // self.gradient_accumulation_steps

        # We want "others" to run from 0 to self.total_updates
        self.sched_others = torch.optim.lr_scheduler.OneCycleLR(
            self.opt_others,
            max_lr=learning_rate,
            total_steps=self.total_updates,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        # Scheduler for text: starts after unfreeze_text_step
        text_cycle = max(1, self.total_updates - unfreeze_text_step)
        self.sched_text = torch.optim.lr_scheduler.OneCycleLR(
            self.opt_text,
            max_lr=learning_rate*0.5,
            total_steps=text_cycle,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        # Scheduler for vision (LoRA)
        vit_cycle = max(1, self.total_updates - unfreeze_vit_step)
        self.sched_vit = torch.optim.lr_scheduler.OneCycleLR(
            self.opt_vit,
            max_lr=learning_rate,
            total_steps=vit_cycle,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )

        # Local step counters for each scheduler
        self.step_others = 0
        self.step_text = 0
        self.step_vit = 0

        # -----------------------------------------------------
        #  5) State tracking & optional resume
        # -----------------------------------------------------
        self.start_epoch = 0
        self.global_step = 0
        self.current_batch_idx = 0
        self.best_loss = float('inf')

        print("Force new training: ", force_new_training)
        print("Use wandb: ", self.use_wandb)
        if not force_new_training:
            print("Loading checkpoint")
            ckpt = self.find_latest_checkpoint()
            print("Checkpoint found: ", ckpt)
            if ckpt:
                self.load_checkpoint(ckpt)
                print("Checkpoint loaded")
            elif self.use_wandb:
                print("No checkpoint found")
        
        if self.use_wandb and wandb.run is None:
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)

        # -----------------------------------------------------
        #  6) Visualization: Only text-visual
        # -----------------------------------------------------
        self.text_viz = TextVisualizer()
        self.vis_samples_tv = self._get_tv_vis_samples(num_vis_samples_tv, use_val=bool(self.val_tv_dataset))

        print("Compiling model")
        self.model = torch.compile(self.model, mode="max-autotune")

        self.logger.info("Initialized MultiModalTrainer for text-visual training.")


    def get_aggregator_temp(self, current_step):
        """
        Exponential schedule from aggregator_temp_start down to aggregator_temp_end.
        
        Args:
            current_step: Current training step
            
        Returns:
            Temperature value for the current step based on exponential decay
        """
        if current_step >= self.aggregator_temp_steps:
            return self.aggregator_temp_end
        
        alpha = current_step / self.aggregator_temp_steps  # goes 0->1
        # Exponential decay:
        # T(step) = T_start * (T_end/T_start)^alpha
        temp = self.aggregator_temp_start * (self.aggregator_temp_end / self.aggregator_temp_start) ** alpha
        
        # Log temperature changes periodically or at significant milestones
        if current_step % 500 == 0 or alpha in [0.25, 0.5, 0.75]:
            self.logger.info(f"Step {current_step}/{self.aggregator_temp_steps}: aggregator_temp = {temp:.4f}")
            
        return temp


    ###########################################
    #     Checkpointing Helpers
    ###########################################
    def find_latest_checkpoint(self):
        ckpts = list(self.output_dir.glob("checkpoint_epoch*_step*.pt"))
        if not ckpts:
            return None

        def _parse_ckpt_name(p):
            """
            e.g. 'checkpoint_epoch3_step1500.pt' => (3,1500)
            """
            name = p.stem
            ep_str = name.split('epoch')[1].split('_')[0]  # '3'
            st_str = name.split('step')[1]                 # '1500'
            return (int(ep_str), int(st_str))

        return max(ckpts, key=_parse_ckpt_name)

    def save_checkpoint(self, epoch, step, is_best=False):
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda":  torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

        ckpt = {
            "epoch": epoch,
            "step": step,
            "current_batch_idx": self.current_batch_idx,
            "rng_state": rng_state,
            "model_state_dict": self.model._orig_mod.state_dict(),
            "opt_others_state": self.opt_others.state_dict(),
            "opt_text_state": self.opt_text.state_dict(),
            "opt_vit_state": self.opt_vit.state_dict(),
            "sched_others_state": self.sched_others.state_dict(),
            "sched_text_state": self.sched_text.state_dict(),
            "sched_vit_state": self.sched_vit.state_dict(),
            "sched_step_others": self.step_others,
            "sched_step_text": self.step_text,
            "sched_step_vit": self.step_vit,
            "best_loss": self.best_loss,
            "config": self.config,
            "vis_samples_tv": self.vis_samples_tv,
            "aggregator_temp_start": self.aggregator_temp_start,
            "aggregator_temp_end": self.aggregator_temp_end,
            "current_aggregator_temp": self.model.aggregator_temp if hasattr(self.model, "aggregator_temp") else None
        }
        torch.save(ckpt, ckpt_path)
        self.logger.info(f"Saved checkpoint: {ckpt_path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            self.logger.info(f"Saved best model checkpoint: {best_path}")

    def load_checkpoint(self, ckpt_path):
        self.logger.info(f"Loading checkpoint from {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=self.device)
        if "_orig_mod." in list(ck["model_state_dict"].keys())[0]:
            print("Checkpoint with _orig_mod. found")
            state_dict = ck["model_state_dict"]
            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    clean_key = key[len('_orig_mod.'):]
                    clean_state_dict[clean_key] = value
                else:
                    clean_state_dict[key] = value
            self.model.load_state_dict(clean_state_dict)
        else:
            self.model.load_state_dict(ck["model_state_dict"])
        self.opt_others.load_state_dict(ck["opt_others_state"])
        self.opt_text.load_state_dict(ck["opt_text_state"])
        self.opt_vit.load_state_dict(ck["opt_vit_state"])
        self.sched_others.load_state_dict(ck["sched_others_state"])
        self.sched_text.load_state_dict(ck["sched_text_state"])
        self.sched_vit.load_state_dict(ck["sched_vit_state"])
        self.step_others = ck.get("sched_step_others", 0)
        self.step_text = ck.get("sched_step_text", 0)
        self.step_vit = ck.get("sched_step_vit", 0)
        self.start_epoch = ck["epoch"]
        self.global_step = ck["step"]
        self.current_batch_idx = ck.get("current_batch_idx", 0)
        self.best_loss = ck["best_loss"],
        self.aggregator_temp_start = ck.get("aggregator_temp_start", self.aggregator_temp_start)
        self.aggregator_temp_end = ck.get("aggregator_temp_end", self.aggregator_temp_end)
        
        # If the model has a temperature attribute, set it based on the checkpoint
        # or calculate it based on the current step
        if hasattr(self.model, "aggregator_temp"):
            current_temp = ck.get("current_aggregator_temp")
            if current_temp is not None:
                self.model.aggregator_temp = current_temp
            else:
                # Calculate the temperature based on the current step
                calculated_temp = self.get_aggregator_temp(self.global_step)
                self.model.aggregator_temp = calculated_temp
                self.logger.info(f"Setting aggregator_temp to {calculated_temp} based on step {self.global_step}")

        rng_state = ck.get("rng_state", None)
        if rng_state is not None:
            torch_state = rng_state["torch"]
            if not isinstance(torch_state, torch.Tensor):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            torch.set_rng_state(torch_state.cpu())

            for i, cuda_state in enumerate(rng_state["cuda"]):
                if not isinstance(cuda_state, torch.Tensor):
                    cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
                torch.cuda.set_rng_state(cuda_state.cpu(), device=i)

            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])

        self.vis_samples_tv = ck["vis_samples_tv"]
        self.best_loss = ck["best_loss"]
        self._update_frozen_params(self.global_step)
        del ck

        self.logger.info(
            f"Checkpoint loaded (epoch={self.start_epoch}, step={self.global_step}, "
            f"batch_offset={self.current_batch_idx})"
        )


    ###########################################
    #  Freeze/Unfreeze logic (Text Only)
    ###########################################
    def _update_frozen_params(self, current_step: int):
        text_module = self.model.text_embedder.encoder
        if current_step < self.config['unfreeze_text_step']:
            for p in text_module.parameters():
                p.requires_grad = False
        else:
            for p in text_module.parameters():
                p.requires_grad = True


    ###########################################
    #     Visualization logic (Text-Visual)
    ###########################################
    def _get_tv_vis_samples(self, n_samples=2, use_val=False):
        """Get clean samples for text-visualization without augmentation"""
        dataset = self.val_tv_dataset if use_val and self.val_tv_dataset else self.tv_dataset
        
        # Use dataset's clean transform for consistent visualization
        clean_transform = dataset.clean_transform
        
        batch_dataloader = DataLoader(
            dataset,
            batch_size=n_samples,
            shuffle=True,
            collate_fn=collate_text_fn
        )
        batch = next(iter(batch_dataloader))
        
        images = []
        captions = []
        for i in range(min(n_samples, len(batch['images']))):
            if hasattr(dataset, 'image_files'):
                img_path = dataset.image_files[i]
                img = Image.open(img_path).convert('RGB')
                images.append(clean_transform(img))
                txt_path = img_path.with_suffix('.txt')
                with open(txt_path, 'r') as f:
                    captions.append(f.read().strip())
            else:
                images.append(batch['images'][i])
                captions.append(batch['captions'][i])
        
        return {
            "images": torch.stack(images),
            "texts": captions
        }

    def visualize_samples(self, epoch):
        """Generate text-visualizations for the current epoch."""
        self.logger.info(f"Generating text visualization for epoch: {epoch}")
        self.model.eval()
        with torch.no_grad():
            for i in range(len(self.vis_samples_tv["images"])):
                frame = self.vis_samples_tv["images"][i].to(self.device)
                text = self.vis_samples_tv["texts"][i]
                out_file_img = self.output_dir / f"tv_viz_epoch{epoch}_sample{i}.png"
                self.text_viz.plot_token_attentions(
                    model=self.model,
                    frame=frame,
                    text=text,
                    output_path=str(out_file_img)
                )
                if self.use_wandb:
                    wandb.log({
                        f"tv_attention_sample_{i}": wandb.Image(str(out_file_img), caption=text),
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "visualization_phase": "text"
                    })
        self.model.train()
        plt.close('all')
        gc.collect()

    ###########################################
    #           Validation Method (Text-Visual)
    ###########################################
    def validate(self, phase=None):
        """
        Evaluate model on the text-visual validation dataset.
        Returns the overall text loss and total validation loss.
        """
        if phase is None:
            phase = "text"
                
        if not self.val_tv_dataloader:
            self.logger.info("No validation dataset provided. Skipping validation.")
            return None, None, None
            
        self.model.eval()
        tv_losses = []
        tv_sim_stats_list = []
        
        with torch.no_grad():
            for tv_batch in tqdm(self.val_tv_dataloader, desc="Validating Text-Visual"):
                frames_tv = tv_batch['images'].to(self.device)
                texts_tv = tv_batch['captions']
                loss_tv, tv_sim_stats = self.model.forward_text_visual(frames_tv, texts_tv)
                tv_losses.append(loss_tv.item())
                tv_sim_stats_list.append(tv_sim_stats)
        
        avg_tv_loss = np.mean(tv_losses) if tv_losses else None
        val_total_loss = avg_tv_loss
        
        avg_tv_sim_stats = {}
        if tv_sim_stats_list:
            for key in tv_sim_stats_list[0].keys():
                avg_tv_sim_stats[f"val_{key}"] = np.mean([stats[key] for stats in tv_sim_stats_list if key in stats])
        
        if self.use_wandb:
            wandb_dict = {
                "epoch": self.start_epoch,
                "global_step": self.global_step,
                "validation_phase": phase,
                "val_loss_tv": avg_tv_loss,
                "val_loss_total": val_total_loss
            }
            wandb_dict.update(avg_tv_sim_stats)
            wandb.log(wandb_dict)
        
        self.model.train()
        
        return None, avg_tv_loss, val_total_loss
    

    def eval_1000_way_retrieval(self):
        """
        1000-way retrieval for text-visual subset.
        """
        if not self.val_tv_dataset:
            self.logger.info("No 1000-way retrieval possible, no validation set loaded.")
            return

        self.logger.info("Starting 1000-way retrieval evaluation for text-visual...")

        tv_results = compute_tv_retrieval_metrics(
            model=self.model,
            dataset=self.val_tv_dataset,
            subset_file=str(self.output_dir / "subset_tv_1000.json"),
            device=self.device
        )
        self.logger.info(f"Text-Visual 1000-way retrieval:\n{tv_results}")
        if self.use_wandb:
            wandb_dict = {}
            for k, v in tv_results.items():
                wandb_dict[f"retrieval_{k}"] = v
            wandb.log(wandb_dict)

        self.logger.info("Done 1000-way retrieval evaluation.")

    ###########################################
    #           Main Training Loop
    ###########################################
    def train(self):
        accumulation_counter = 0
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            #self.eval_1000_way_retrieval()
            #self.visualize_samples(epoch)
            phase = "text"
            self.logger.info(f"Epoch {epoch} - Phase: Text-Visual Training")
            self.logger.info(f"Epoch {epoch} starting")
            self.tv_iter = iter(self.tv_dataloader)

            print(f"Resuming from batch {self.current_batch_idx}")
            for _ in tqdm(range(self.current_batch_idx), desc="Resuming from checkpoint"):
                try:
                    next(self.tv_iter)
                except StopIteration:
                    self.tv_iter = iter(self.tv_dataloader)
                    next(self.tv_iter)

            max_steps = len(self.tv_dataloader)
            pbar = tqdm(range(max_steps - self.current_batch_idx), desc=f"Epoch {epoch} ({phase})")
            epoch_losses = []

            for batch_idx in pbar:
                self._update_frozen_params(self.global_step)
                current_temp = self.get_aggregator_temp(self.global_step)
                if hasattr(self.model, "aggregator_temp"):
                    self.model.aggregator_temp = current_temp

                grad_norm_others = None
                grad_norm_text = None
                grad_norm_vit = None

                try:
                    tv_batch = next(self.tv_iter)
                except StopIteration:
                    print("StopIteration in tv_iter")
                    self.tv_iter = iter(self.tv_dataloader)
                    tv_batch = next(self.tv_iter)
                frames_tv = tv_batch['images'].to(self.device)
                texts_tv = tv_batch['captions']
                try:
                    tv_loss, tv_sim_stats = self.model.forward_text_visual(frames_tv, texts_tv)
                except torch.cuda.OutOfMemoryError:
                    print("Out of memory error in forward_text_visual")
                    self.model.zero_grad()
                    self.opt_others.zero_grad()
                    self.opt_text.zero_grad()
                    self.opt_vit.zero_grad()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                loss_total = tv_loss

                loss_scaled = loss_total / self.gradient_accumulation_steps
                loss_scaled.backward()
                accumulation_counter += 1

                if (accumulation_counter % self.gradient_accumulation_steps) == 0:
                    others_grads = [p.grad.norm() for p in self.others_params if p.grad is not None]
                    text_grads = [p.grad.norm() for p in self.text_params if p.grad is not None]
                    vit_grads = [p.grad.norm() for p in self.vit_params if p.grad is not None]

                    grad_norm_others = torch.norm(torch.stack(others_grads)) if others_grads else torch.tensor(0.0, device=self.device)
                    grad_norm_text = torch.norm(torch.stack(text_grads)) if text_grads else torch.tensor(0.0, device=self.device)
                    grad_norm_vit = torch.norm(torch.stack(vit_grads)) if vit_grads else torch.tensor(0.0, device=self.device)

                    clip_grad_norm_(self.model.text_embedder.parameters(), 10.0)

                    # Step each optimizer
                    self.opt_others.step()
                    self.opt_others.zero_grad()
                    if self.step_others < self.total_updates:
                        self.sched_others.step()
                        self.step_others += 1

                    if self.global_step >= self.config['unfreeze_text_step']:
                        self.opt_text.step()
                        self.opt_text.zero_grad()
                        if self.step_text < (self.total_updates - self.config['unfreeze_text_step']):
                            self.sched_text.step()
                            self.step_text += 1
                    else:
                        self.opt_text.zero_grad()

                    self.opt_vit.step()
                    self.opt_vit.zero_grad()
                    if self.step_vit < (self.total_updates - self.config['unfreeze_vit_step']):
                        self.sched_vit.step()
                        self.step_vit += 1

                loss_val = loss_total.item()
                epoch_losses.append(loss_val)
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "phase": phase})

                if self.use_wandb:
                    wandb_dict = {
                        "train_loss": loss_val,
                        "loss_tv": tv_loss.item() if tv_loss is not None else 0.0,
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "training_phase": phase,
                        "lr_others": self.opt_others.param_groups[0]['lr'],
                        "lr_text": self.opt_text.param_groups[0]['lr'],
                        "lr_vit": self.opt_vit.param_groups[0]['lr'],
                        #"temperature": self.model.temperature.item(),
                        "aggregator_temp": self.model.aggregator_temp
                    }
                    if grad_norm_others is not None:
                        wandb_dict["grad_norm_others"] = grad_norm_others.item()
                    if grad_norm_text is not None:
                        wandb_dict["grad_norm_text"] = grad_norm_text.item()
                    if grad_norm_vit is not None:
                        wandb_dict["grad_norm_vit"] = grad_norm_vit.item()
                    

                    wandb_dict.update(tv_sim_stats)
                    wandb.log(wandb_dict)

                del loss_total
                torch.cuda.empty_cache()
                
                if self.global_step % 500 == 0:
                    gc.collect()
                if (self.global_step % self.vis_every == 0):
                    self.visualize_samples(epoch)
                if (self.global_step > 0) and (self.global_step % self.save_every_steps == 0):
                    self.current_batch_idx = batch_idx + 1
                    self.save_checkpoint(epoch, self.global_step)
                if self.global_step > 0 and self.global_step % self.validation_frequency == 0:
                    _, val_tv_loss, val_total_loss = self.validate(phase=phase)
                    print(f"Validation_Text_Visual: {f'{val_tv_loss:.4f}' if val_tv_loss is not None else 'N/A'}, " +
                          f"Validation_Total: {f'{val_total_loss:.4f}' if val_total_loss is not None else 'N/A'}")
                    self.eval_1000_way_retrieval()
                self.global_step += 1
                
            epoch_loss = np.mean(epoch_losses)
            self.logger.info(f"Epoch {epoch} completed. Average loss={epoch_loss:.4f}")
            
            if self.val_tv_dataloader:
                self.logger.info(f"Running validation after epoch {epoch}...")
                _, val_tv_loss, val_total_loss = self.validate(phase=phase)
                self.eval_1000_way_retrieval()
                if val_total_loss is not None:
                    self.logger.info(f"Validation loss after epoch {epoch}: {val_total_loss:.4f}")
                    if val_total_loss < self.best_loss:
                        self.best_loss = val_total_loss
                        self.save_checkpoint(epoch, self.global_step, is_best=True)
                        self.logger.info(f"New best model saved with val_loss: {val_total_loss:.4f}")
            
            self.current_batch_idx = 0
            self.save_checkpoint(epoch, self.global_step)

        self.logger.info("Training complete!")


###########################################
#        Main Script
###########################################
if __name__ == "__main__":
    print("Starting text-visual training...")

    trainer = MultiModalTrainer(
        text_dataset_path="/home/cis/cc3m-ironic",
        text_dataset_val_path="/home/cis/cc3m-ironic-val",
        output_dir="./outputs-logexp-null-patch-correctedattnmask-tfidf",
        run_name="logexp-null-patch-corrected-tfidf",
        batch_size_tv=60,
        num_epochs=10,
        learning_rate=1e-4,
        use_wandb=True,
        force_new_training=False,
        vis_every=10000,
        save_every_steps=10000,
        num_workers=12,
        device="cuda",
        gradient_accumulation_steps=1,
        unfreeze_text_step=5000,
        unfreeze_vit_step=0,
        project_name="TriadText",
        num_vis_samples_tv=60,
        use_amp=True,
        validation_frequency=20000,
        aggregator_temp_start=4.0,
        aggregator_temp_end=0.001,
        aggregator_temp_steps=45000,
        idf_path="/home/cis/cc3m-ironic/idf.npy"
    )

    trainer.train()
