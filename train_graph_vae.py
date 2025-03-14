import os
import sys 
import time
import math
import json
import pickle
import logging
import argparse
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import numpy as np

import transformers
from transformers import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

from eval_graph_vae import Evaluator
from model.box_vae import SceneVAEModel
from data import build_train_dataloader
from loss import VaeGaussCriterion, BoxL1Criterion

# accelerate launch train_graph_vae.py --lr_scheduler 'linear' --checkpointing_steps=10000 --batch_size 64
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_diffusion_model_path", type=str, default='/model/anonymity/StableDiffusion/stable-diffusion-v1-5', help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument('--data_dir', type=str, default='/data/anonymity/VisualGenome', help='path to training dataset')
    parser.add_argument('--output_dir', type=str, default="./results", help='path to save checkpoint')
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--dataloader_shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument("--tracker_project_name", type=str, default="vae_box", help="The `project_name` passed to Accelerator",)
    parser.add_argument('--resolution', type=int, default=512, help='resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--checkpointing_steps", type=int, default=5000, help="Save a checkpoint of the training state every X updates.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)",)
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--vae_loss_weight", type=float, default=0.1, help="")
    parser.add_argument("--box_loss_weight", type=float, default=1, help="")
    parser.add_argument('--embedding_dim', type=int, default=64, help='embedding dim of GCN')

    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d-%Hh%Mm%Ss", time.localtime())
    args.output_dir = os.path.join(args.output_dir, 'train', f'{args.tracker_project_name}-{timestamp}') 
    return args

class Trainer:
    def __init__(self, args):
        # Init settings
        self.args = args
        self.logger = get_logger(__name__, log_level="INFO")
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Enable TF32 for faster training on Ampere GPUs,
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f'{args.output_dir}/config.json', 'wt') as f:
                    json.dump(vars(args), f, indent=4)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_diffusion_model_path, 
            subfolder="tokenizer"
        )

        # Data
        self.train_dataloader, self.val_dataloader, self.vocab = build_train_dataloader(args, self.tokenizer)

        # Model
        num_objs = len(self.vocab['object_idx_to_name'])
        num_rels = len(self.vocab['pred_idx_to_name'])
        self.vae = SceneVAEModel(self.args, num_objs, num_rels)
        self.vae.train()

        # Criterion
        self.vae_gauss_criterion = VaeGaussCriterion()
        self.box_l1_criterion = BoxL1Criterion()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=args.max_train_steps * self.accelerator.num_processes,
        )

        # Evaluate agent
        self.evaluator = Evaluator(args=self.args, logger=self.logger, eval_dataloader=self.val_dataloader, vocab=self.vocab, device=self.accelerator.device)

        # Prepare everything with our `accelerator`.
        self.vae, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.vae, 
            self.optimizer, 
            self.train_dataloader, 
            self.lr_scheduler
        )

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            args.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            args.mixed_precision = self.accelerator.mixed_precision

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        self.progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=0,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(args))
            self.accelerator.init_trackers(args.tracker_project_name, tracker_config)
        
    def start(self):
        # Print information
        self.logger.info('  Global configuration as follows:')
        for key, val in vars(self.args).items():
            self.logger.info("  {:28} {}".format(key, val))
        
        # Start to training
        total_batch_size = self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        self.logger.info("\n")
        self.logger.info(f"  Running training:")
        self.logger.info(f"  Num Iterations = {len(self.train_dataloader)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        
        self.train()

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.vae = self.accelerator.unwrap_model(self.vae)
            save_path = os.path.join(args.output_dir, f"checkpoint-final")
            os.mkdir(save_path)
            ckpt_path = os.path.join(save_path, 'ckpt.pt')
            self.accelerator.save(self.vae.state_dict(), ckpt_path)
            self.logger.info(f"Saved state to {save_path}")

            # Run a final round of validation.
            self.logger.info("Running inference for collecting generated boxes...")
            stats_path = os.path.join(save_path, 'stats.pkl')
            box_mean_est, box_cov_est = self.vae.collect_data_statistics(self.train_dataloader, self.accelerator.device)
            pickle.dump([box_mean_est, box_cov_est], open(stats_path, 'wb'))
            self.evaluator.start(save_path)

        self.accelerator.end_training()
    
    def train(self):
        self.global_step = 0

        for epoch in range(0, self.args.num_train_epochs):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        #pbar = tqdm(self.train_dataloader, file=sys.stdout)
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.vae):
                imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch
                mu, logvar, box_pred = self.vae(objs, obj_clip_embs, boxes, triples, rel_clip_embs)
                
                # Compute loss
                box_loss = self.box_l1_criterion(box_pred, boxes)
                vae_loss = self.vae_gauss_criterion(mu, logvar)
                loss = box_loss * self.args.box_loss_weight + vae_loss * self.args.vae_loss_weight

                # Gather the losses across all processes for logging (if we use distributed training).
                log_box_loss = self.gather_loss(box_loss)
                log_vae_loss = self.gather_loss(vae_loss)
                log_loss = self.gather_loss(loss)

                # Backpropagate
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.vae.parameters(), args.max_grad_norm)
                
                self.lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()
                        
            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                self.progress_bar.update(1)
                self.global_step += 1

                self.accelerator.log({"box_loss": log_box_loss}, step=self.global_step)
                self.accelerator.log({"vae_loss": log_vae_loss}, step=self.global_step)
                self.accelerator.log({"train_loss": log_loss}, step=self.global_step)
                self.accelerator.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.global_step)

                if self.global_step % args.checkpointing_steps == 0:
                    if self.accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{self.global_step}")
                        os.mkdir(save_path)
                        unwrapped_model = self.accelerator.unwrap_model(self.vae)
                        ckpt_path = os.path.join(save_path, 'ckpt.pt')
                        self.accelerator.save(unwrapped_model.state_dict(), ckpt_path)
                        self.logger.info(f"Saved state to {save_path}")
                        
                        # Validation
                        stats_path = os.path.join(save_path, 'stats.pkl')
                        box_mean_est, box_cov_est = self.vae.collect_data_statistics(self.train_dataloader, self.accelerator.device)
                        pickle.dump([box_mean_est, box_cov_est], open(stats_path, 'wb'))
                        self.evaluator.start(save_path)

            logs = {"step_loss": '%.4f' % loss.detach().item(), "lr": '%.2e' % self.lr_scheduler.get_last_lr()[0]}
            self.progress_bar.set_postfix(**logs)

            if self.global_step >= args.max_train_steps:
                break
    
    def gather_loss(self, loss):
        avg_loss = self.accelerator.gather(loss.repeat(args.batch_size)).mean()
        loss = avg_loss.item() / self.args.gradient_accumulation_steps
        return loss

if __name__ == '__main__':
    args = parse_args()

    # start training
    trainer = Trainer(args)
    trainer.start()
