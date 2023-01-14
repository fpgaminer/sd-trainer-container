import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data.distributed import DistributedSampler

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig, CLIPFeatureExtractor
from datasets import load_dataset

from pathlib import Path
import itertools

import numpy as np
import wandb
import random
import argparse
import os
import math
import gc
from typing import Optional
import sys
import json


SAMPLERS = {
	"DDIM": DDIMScheduler,
	"PNDM": PNDMScheduler,
	"EulerA": EulerAncestralDiscreteScheduler,
}
bool_arg = lambda x: (str(x).lower() in ['true', '1', 't', 'y', 'yes'])
parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
parser.add_argument('--device_batch_size', type=int, default=1, help='Device batch size')
parser.add_argument('--batch_size', type=int, default=1, help='Actual batch size; gradient accumulation is used on device_batch_size to achieve this')
parser.add_argument('--wandb_sweep', type=str, default=None, help='Wandb sweep id')
parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project')
parser.add_argument('--dataset', type=str, help='Dataset to use')
parser.add_argument('--only_subreddit', type=bool_arg, default=False, help='Only use the subreddit in the prompt, and ignore similarity score')
parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
parser.add_argument('--warmup_samples', type=int, default=10000, help='Warmup samples')
parser.add_argument('--max_samples', type=int, default=1000000, help='Max samples trained for in this session')
parser.add_argument('--use_8bit_adam', type=bool_arg, default=False, help='Use 8 bit adam')
parser.add_argument('--save_every', type=int, default=10000, help='Save a checkpoint every n samples (approx)')
parser.add_argument('--save_images_every', type=int, default=10000, help='Save images every n samples (approx)')
parser.add_argument('--save_image_prompts', type=str, default="", help='Prompts to use for image generation, separated by pipes (|)')
parser.add_argument('--save_image_sampler', type=str, default="DDIM", help=f"Sampler to use for image generation: {', '.join(SAMPLERS.keys())}")
parser.add_argument('--save_image_num_steps', type=int, default=50, help='Number of steps to run for image generation')
parser.add_argument('--test_every', type=int, default=10000, help='Test every n samples (approx)')
parser.add_argument('--fp16', type=bool_arg, default=False, help='Use fp16')
parser.add_argument('--train_text_encoder', type=bool_arg, default=True, help='Train the text encoder')
parser.add_argument('--lr_scheduler_type', type=str, default="constant_with_warmup", help='Learning rate scheduler type')
parser.add_argument('--allow_tf32', type=bool_arg, default=False, help='Allow tf32')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--xformers', type=bool_arg, default=False, help='Use xformers')


def main():
	args = parser.parse_args()

	save_image_prompts = [s.strip() for s in args.save_image_prompts.split("|")]

	def runner(args, config):
		if distributed_rank() == 0:
			with wandb.init(config=config, project=args.wandb_project):
				trainer = MainTrainer(
					config=wandb.config,
					output_dir=args.output_dir,
					resume=args.resume,
					device_batch_size=args.device_batch_size,
					dataset=args.dataset,
					save_every=args.save_every,
					save_images_every=args.save_images_every,
					save_image_prompts=save_image_prompts,
					test_every=args.test_every,
				)

				trainer.train()
		else:
			trainer = MainTrainer(
				config=None,
				output_dir=args.output_dir,
				resume=args.resume,
				device_batch_size=args.device_batch_size,
				dataset=args.dataset,
				save_every=args.save_every,
				save_images_every=args.save_images_every,
				save_image_prompts=save_image_prompts,
				test_every=args.test_every,
			)

			trainer.train()

	if args.wandb_sweep is not None:
		if distributed_rank() == 0:
			wandb.agent(args.wandb_sweep, lambda: runner(args, None), count=1)
		else:
			runner(args, None)
	else:
		config = {
			"batch_size": args.batch_size,
			"adam_beta1": 0.9,
			"adam_beta2": 0.999,
			"adam_weight_decay": 0.01,
			"adam_epsilon": 1e-8,
			"max_grad_norm": 1.0,
			"learning_rate": args.lr,
			"train_text_encoder": args.train_text_encoder,
			"clip_penultimate": False,
			"warmup_samples": args.warmup_samples,
			"max_samples": args.max_samples,
			"use_8bit_adam": args.use_8bit_adam,
			"lr_scheduler_type": args.lr_scheduler_type,
			"allow_tf32": args.allow_tf32,
			"only_subreddit": args.only_subreddit,
			"fp16": args.fp16,
			"seed": args.seed,
			"xformers": args.xformers,
			"save_image_sampler": args.save_image_sampler,
			"save_image_num_steps": args.save_image_num_steps,
		}

		runner(args, config)


class FinetuneDataset(Dataset):
	def __init__(self, tokenizer, text_encoder, device, dataset, only_subreddit, is_validation):
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.text_encoder = text_encoder.module if type(text_encoder) is torch.nn.parallel.DistributedDataParallel else text_encoder
		self.device = device
		self.only_subreddit = only_subreddit
		self.is_validation = is_validation
		self.clip_penultimate = False

		# We apply 10% dropout to the text conditioning.
		# To implement this in a reproducible way, we take a simple approach and deterministically generate a bitmask.
		# The downside is that this means the dropping is "fixed"; it will be the same for every epoch of our data.
		# A better approach that varies the dropping but is also reproducible under multiprocessing would be a bit more complicated.
		g = torch.Generator()
		g.manual_seed(42)
		self.bitmask = torch.rand(len(self.dataset), generator=g) > 0.1
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		row = self.dataset[idx % len(self.dataset)]
		title = row['title']
		subreddit = row['subreddit']
		latent = row['latent']

		# Dropping of the text conditioning
		# TODO: Should dropping be applied during validation?
		if self.bitmask[idx]:
			if not self.only_subreddit:
				title += f", subreddit:{subreddit}"
			else:
				title = f"subreddit: {subreddit}"
		else:
			title = ""

		prompt_ids = title

		return {
			'latent': latent,
			'prompt_ids': prompt_ids,
		}

	def collate_fn(self, examples):
		#latents, prompt_ids = zip(*batch)
		latents = torch.stack([example['latent'] for example in examples])
		latents.to(memory_format=torch.contiguous_format).float()

		#latents = torch.stack(latents)
		#pixel_values = torch.stack(images)
		#pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

		max_length = self.tokenizer.model_max_length - 2
		input_ids = [self.tokenizer([x['prompt_ids']], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=max_length).input_ids for x in examples]

		for i, x in enumerate(input_ids):
			for j, y in enumerate(x):
				input_ids[i][j] = [self.tokenizer.bos_token_id, *y, *np.full((self.tokenizer.model_max_length - len(y) - 1), self.tokenizer.eos_token_id)]
		
		return {
			'latents': latents,
			'input_ids': input_ids,
		}


class MainTrainer:
	def __init__(self, config, output_dir, resume, device_batch_size, dataset, save_every, save_images_every, save_image_prompts, test_every):
		self.rank = distributed_rank()
		self.world_size = distributed_world_size()
		self.output_dir = output_dir
		self.resume = resume
		self.device = torch.device('cuda')
		self.dataset = dataset
		self.save_image_prompts = save_image_prompts

		if self.rank == 0:
			self.batch_size = config['batch_size']
			self.adam_beta1 = config['adam_beta1']
			self.adam_beta2 = config['adam_beta2']
			self.adam_weight_decay = config['adam_weight_decay']
			self.adam_epsilon = config['adam_epsilon']
			self.max_grad_norm = config['max_grad_norm']
			self.learning_rate = config['learning_rate']
			self.train_text_encoder = config['train_text_encoder']
			self.clip_penultimate = config['clip_penultimate']
			self.warmup_samples = config['warmup_samples']
			self.max_samples = config['max_samples']
			self.use_8bit_adam = config['use_8bit_adam']
			self.lr_scheduler_type = config['lr_scheduler_type']
			self.allow_tf32 = config['allow_tf32']
			self.only_subreddit = config['only_subreddit']
			self.fp16 = config['fp16']
			self.seed = config['seed']
			self.xformers = config['xformers']
			self.save_image_sampler_cls = SAMPLERS[config['save_image_sampler']]
			self.save_image_num_steps = config['save_image_num_steps']

			os.makedirs(self.output_dir, exist_ok=True)

			objects = [self.batch_size, self.adam_beta1, self.adam_beta2, self.adam_weight_decay, self.adam_epsilon, self.max_grad_norm, self.learning_rate, self.train_text_encoder, self.clip_penultimate, self.warmup_samples, self.max_samples, self.use_8bit_adam, self.lr_scheduler_type, self.allow_tf32, self.only_subreddit, self.fp16, self.seed, self.xformers, self.save_image_sampler_cls, self.save_image_num_steps]
			torch.distributed.broadcast_object_list(objects)
		else:
			objects = [None] * 20
			torch.distributed.broadcast_object_list(objects)
			self.batch_size, self.adam_beta1, self.adam_beta2, self.adam_weight_decay, self.adam_epsilon, self.max_grad_norm, self.learning_rate, self.train_text_encoder, self.clip_penultimate, self.warmup_samples, self.max_samples, self.use_8bit_adam, self.lr_scheduler_type, self.allow_tf32, self.only_subreddit, self.fp16, self.seed, self.xformers, self.save_image_sampler_cls, self.save_image_num_steps = objects

		self.seed += self.rank

		if self.xformers and not is_xformers_available():
			raise ValueError("xformers is not installed`")

		if self.allow_tf32:
			torch.backends.cuda.matmul.allow_tf32 = True

		# Calculate the batch size and gradient accumulation steps to get the target batch size
		self.device_batch_size = min(self.batch_size, device_batch_size)
		self.gradient_accumulation_steps = self.batch_size // (self.device_batch_size * self.world_size)
		self.test_every = int(math.ceil(test_every / self.batch_size))
		self.save_every = int(math.ceil(save_every / self.batch_size))
		self.save_images_every = int(math.ceil(save_images_every / self.batch_size))

		assert self.batch_size == self.device_batch_size * self.gradient_accumulation_steps * self.world_size, f"batch_size {self.batch_size} must be divisible by device_batch_size {device_batch_size}"
	
	def load_models(self):
		model_path = "CompVis/stable-diffusion-v1-4"
		if self.resume is not None:
			model_path = self.resume
		
		self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=None, use_fast=False)  # TODO: use_fast?
		self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", revision=None)
		if self.save_images_every > 0:
			self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=None)
		else:
			self.vae = None
		self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=None)
		self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

		if not self.train_text_encoder:
			self.text_encoder.requires_grad_(False)
		if self.vae is not None:
			self.vae.requires_grad_(False)
		
		if self.xformers:
			self.unet.enable_xformers_memory_efficient_attention()

		# Move models to device
		self.weight_dtype = torch.float16 if self.fp16 else torch.float32
		if self.vae is not None:
			self.vae.to(self.device, dtype=self.weight_dtype)
		self.text_encoder.to(self.device, dtype=self.weight_dtype if not self.train_text_encoder else torch.float32)
		self.unet.to(self.device, dtype=torch.float32)

		self.unet = torch.nn.parallel.DistributedDataParallel(
			self.unet,
			device_ids=[self.rank],
			output_device=self.rank,
			gradient_as_bucket_view=True,
		)

		if self.train_text_encoder:
			self.text_encoder = torch.nn.parallel.DistributedDataParallel(
				self.text_encoder,
				device_ids=[self.rank],
				output_device=self.rank,
				gradient_as_bucket_view=True,
			)
	
	def build_optimizer(self):
		# Optimizer
		if self.use_8bit_adam:
			import bitsandbytes
			optimizer_class = bitsandbytes.optim.AdamW8bit
		else:
			optimizer_class = torch.optim.AdamW
		
		params_to_optimize = self.unet.parameters() if not self.train_text_encoder else itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
		self.optimizer = ZeroRedundancyOptimizer(
			params_to_optimize,
			optimizer_class=optimizer_class,
			parameters_as_bucket_view=True,
			lr=self.learning_rate,
			betas=(self.adam_beta1, self.adam_beta2),
			weight_decay=self.adam_weight_decay,
			eps=self.adam_epsilon,
		)

		if self.resume is not None:
			self.optimizer.load_state_dict(torch.load(os.path.join(self.resume, "optimizer.pt"), map_location="cpu"))
	
	def build_datasets(self):
		self.source_dataset = load_dataset(self.dataset)
		self.source_dataset.set_format("torch")
		self.dataset = FinetuneDataset(self.tokenizer, self.text_encoder, self.device, self.source_dataset["train"], self.only_subreddit, False)
		self.validation_dataset = FinetuneDataset(self.tokenizer, self.text_encoder, self.device, self.source_dataset["validation"], self.only_subreddit, True)

		self.total_steps = self.max_samples // self.batch_size
		self.total_device_batches = self.total_steps * self.gradient_accumulation_steps
	
	def build_dataloader(self):
		self.train_sampler = BetterDistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)

		if self.resume is not None:
			checkpoint = torch.load(Path(self.resume)/"info.pt", map_location="cpu")
			self.train_sampler.set_state(checkpoint["sampler_epoch"], checkpoint["sampler_index"])

		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.device_batch_size, sampler=self.train_sampler, num_workers=4, collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
		self.dataloader_iter = iter(self.dataloader)

	def train(self):
		if self.seed is not None:
			torch.manual_seed(self.seed)
			np.random.seed(self.seed)
			random.seed(self.seed)

		self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
		self.load_models()
		self.build_optimizer()
		self.build_datasets()
		self.build_dataloader()

		self.lr_scheduler = get_scheduler(
			self.lr_scheduler_type,
			optimizer=self.optimizer,
			num_warmup_steps=self.warmup_samples // self.batch_size,
			num_training_steps=None,
		)

		# Load lr scheduler, random state, scaler, etc if resuming
		if self.resume is not None:
			self.lr_scheduler.load_state_dict(torch.load(Path(self.resume)/"lr_scheduler.pt"))

			checkpoint = torch.load(Path(self.resume)/f"random_state{self.rank}.pt", map_location="cpu")
			random.setstate(checkpoint["random_state"])
			np.random.set_state(checkpoint["numpy_random_state"])
			torch.random.set_rng_state(checkpoint["torch_random_state"])
			torch.cuda.random.set_rng_state(checkpoint["torch_cuda_random_state"])

			self.scaler.load_state_dict(torch.load(Path(self.resume)/f"scaler{self.rank}.pt"))

			checkpoint = torch.load(Path(self.resume)/"info.pt", map_location="cpu")
			resume_global_step = checkpoint["global_step"]
			resume_device_step = (resume_global_step + 1) * self.gradient_accumulation_steps

			# max_samples sets the number of samples to train on in this session.
			# If we are resuming, we need to adjust the total number of steps to account for the number of steps we have already taken.
			self.total_steps = resume_global_step + 1 + (self.max_samples // self.batch_size)
			self.total_device_batches = self.total_steps * self.gradient_accumulation_steps
		else:
			resume_device_step = 0
			self.total_steps = self.max_samples // self.batch_size
			self.total_device_batches = self.total_steps * self.gradient_accumulation_steps

		loss_sum = 0

		for device_step in tqdm(range(resume_device_step, self.total_device_batches), ncols=80):
			self.global_step = device_step // self.gradient_accumulation_steps
			self.global_samples_seen = (device_step + 1) * self.device_batch_size * self.world_size

			self.unet.train()
			if self.train_text_encoder:
				self.text_encoder.train()
			
			# Reload the dataloader as needed
			try:
				batch = next(self.dataloader_iter)
			except StopIteration:
				self.train_sampler.set_epoch(self.train_sampler.epoch + 1) # This is important to ensure the data is re-shuffled after every use
				self.dataloader_iter = iter(self.dataloader)
				batch = next(self.dataloader_iter)
			
			latents = batch["latents"].to(self.device, dtype=self.weight_dtype) # Should dtype be used here? I don't see harm in having the latent stay at 32-bit precision

			# Noise
			noise = torch.randn_like(latents)
			bsz = latents.shape[0]
			timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
			timesteps = timesteps.long()

			noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

			# Conditioning
			encoder_hidden_states = [self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True).last_hidden_state[0] for input_id in batch['input_ids']]
			encoder_hidden_states = torch.stack(tuple(encoder_hidden_states))
			#encoder_hidden_states = text_encoder(batch["input_ids"].to('cuda:1'))[0]
			#encoder_hidden_states = batch["input_ids"]

			if not self.train_text_encoder:
				raise NotImplementedError("Not implemented yet")
			
			#with self.unet.join(), self.text_encoder.join():
			if True:
				# Forward pass
				with torch.autocast('cuda', enabled=self.fp16):
					model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

					# Loss
					# TODO: Not sure if this should be under autocast; PyTorch docs show an example with it under autocast, but other repos don't
					loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
					loss = loss / self.gradient_accumulation_steps

				# Backward pass
				self.scaler.scale(loss).backward()
				torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
				loss_sum += (loss.detach().item() / self.world_size)

				# Take a step if accumulation is complete
				if (device_step + 1) % self.gradient_accumulation_steps == 0:
					# Unscale gradients before clipping
					self.scaler.unscale_(self.optimizer)

					# Clip gradients
					torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
					if self.train_text_encoder:
						torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), self.max_grad_norm)
					
					# Take a step
					self.scaler.step(self.optimizer)
					self.scaler.update()
					self.lr_scheduler.step()
					self.optimizer.zero_grad()
					logs = {
						"train/loss": loss_sum,
						"train/lr": self.lr_scheduler.get_last_lr()[0],
						"train/step": self.global_step,
						"train/samples": self.global_samples_seen,
					}
					if self.rank == 0:
						wandb.log(logs, step=self.global_step)
					loss_sum = 0
			
			if (device_step + 1) % self.gradient_accumulation_steps == 0:
				# Save checkpoint
				if self.save_every > 0 and ((self.global_step + 1) % self.save_every == 0 or (self.global_step + 1) == self.total_steps):
					self.save_checkpoint()
				
				# Validation
				if self.test_every > 0 and (self.global_step + 1) % self.test_every == 0:
					self.do_validation()
				
				# Sample image generation
				if self.save_images_every > 0 and (self.global_step + 1) % self.save_images_every == 0:
					self.do_image_samples()

	def save_checkpoint(self):
		path = Path(self.output_dir) / f"checkpoint-{self.global_step}"
		self.optimizer.consolidate_state_dict(to=0)

		if self.rank == 0:
			print(f"Saving model checkpoint to {path}")
		
		path.mkdir(parents=True, exist_ok=True)

		# Only save the model, optimizer, and lr_scheduler on rank 0
		if self.rank == 0:
			text_encoder = self.text_encoder.module if type(self.text_encoder) == torch.nn.parallel.DistributedDataParallel else self.text_encoder
			
			pipeline = StableDiffusionPipeline(
				text_encoder=text_encoder,
				vae=AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", revision=None),
				unet=self.unet.module,
				tokenizer=self.tokenizer,
				scheduler=PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler", revision=None), # TODO: Why is this PNDMScheduler whereas from scratch it's DDPM?
				safety_checker=None,
				feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
			)

			pipeline.save_pretrained(path)

			sampler_epoch = self.train_sampler.epoch
			sampler_index = self.global_samples_seen // self.world_size // self.device_batch_size  # How many times has the sampler been iterated, in total
			sampler_index = sampler_index % (len(self.dataloader) * self.device_batch_size)   # How many times has the sampler been iterated, in this epoch (we use dataloader len because of drop_last)

			torch.save({
				"global_step": self.global_step,
				"global_samples_seen": self.global_samples_seen,
				"sampler_epoch": sampler_epoch,
				"sampler_index": sampler_index,
			}, path/"info.pt")
			torch.save(self.optimizer.state_dict(), path/"optimizer.pt")
			torch.save(self.lr_scheduler.state_dict(), path/"lr_scheduler.pt")
		
		# The rest is rank-dependent
		# NOTE: I don't know for sure if scaler is rank-dependent, but the losses are different on each rank so I'm assuming it is
		torch.save(self.scaler.state_dict(), path/f"scaler{self.rank}.pt")
		torch.save({
			"random_state": random.getstate(),
			"numpy_random_state": np.random.get_state(),
			"torch_random_state": torch.random.get_rng_state(),
			"torch_cuda_random_state": torch.cuda.random.get_rng_state(),
		}, path/f"random_state{self.rank}.pt")

	def do_validation(self):
		if self.rank != 0:
			return
		
		print("Doing validation")
		
		unet = self.unet.module
		text_encoder = self.text_encoder.module if type(self.text_encoder) == torch.nn.parallel.DistributedDataParallel else self.text_encoder
		
		unet.eval()
		text_encoder.eval()

		# Note: I'm using batch_size=1 here, because there was some reproducibility issue with batch_size > 1
		# TODO: Investigate this
		validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=self.validation_dataset.collate_fn)
		validation_loss = 0
		validation_steps = 0

		# Set seed for validation consistency
		old_torch_rng_state = torch.get_rng_state()
		old_torch_cuda_rng_state = torch.cuda.get_rng_state()
		torch.manual_seed(42)
		torch.cuda.manual_seed(42)

		with torch.no_grad():
			for batch in validation_dataloader:
				latents = batch["latents"].to(self.device, dtype=self.weight_dtype)  # TODO: see the todo at the other instance of this

				noise = torch.randn_like(latents)
				bsz = latents.shape[0]
				timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
				timesteps = timesteps.long()

				noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

				encoder_hidden_states = [text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True).last_hidden_state[0] for input_id in batch['input_ids']]
				encoder_hidden_states = torch.stack(tuple(encoder_hidden_states))

				with torch.autocast('cuda', enabled=self.fp16):
					model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

					loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

				validation_loss += loss.detach().item()
				validation_steps += 1
		
		# Restore PyTorch RNG state
		torch.set_rng_state(old_torch_rng_state)
		torch.cuda.set_rng_state(old_torch_cuda_rng_state)

		validation_loss /= validation_steps
		logs = {
			"validation/loss": validation_loss,
			"validation/samples": self.global_samples_seen,
		}
		wandb.log(logs, step=self.global_step)

	def do_image_samples(self):
		if self.rank != 0:
			return
		
		unet = self.unet.module
		text_encoder = self.text_encoder.module if type(self.text_encoder) == torch.nn.parallel.DistributedDataParallel else self.text_encoder
		
		unet.eval()
		text_encoder.eval()

		generator = torch.Generator(self.device).manual_seed(42)
		scheduler = self.save_image_sampler_cls.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
		pipeline = StableDiffusionPipeline(
			text_encoder=text_encoder,
			vae=self.vae,
			unet=unet,
			tokenizer=self.tokenizer,
			scheduler=scheduler,
			safety_checker=None,
			feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
		).to(self.device)

		images = []
		num_batches = int(math.ceil(len(self.save_image_prompts) / self.device_batch_size))

		with torch.no_grad():
			with torch.autocast('cuda', enabled=self.fp16):
				for i in range(num_batches):
					batch_prompts = self.save_image_prompts[i * self.device_batch_size:(i + 1) * self.device_batch_size]
					negative_prompts = ["lowres" for _ in batch_prompts]
					images += [wandb.Image(img) for img in pipeline(batch_prompts, num_inference_steps=self.save_image_num_steps, negative_prompt=negative_prompts, num_images_per_prompt=1, guidance_scale=11, generator=generator).images]
		
		wandb.log({"images/images": images, "images/samples": self.global_samples_seen}, step=self.global_step)

		del pipeline
		gc.collect()


def distributed_rank():
	if not torch.distributed.is_initialized():
		return 0
	
	return torch.distributed.get_rank()


def distributed_world_size():
	if not torch.distributed.is_initialized():
		return 1
	
	return torch.distributed.get_world_size()


def distributed_setup():
	torch.distributed.init_process_group(backend="nccl", init_method="env://")


def distributed_cleanup():
	torch.distributed.destroy_process_group()


class BetterDistributedSampler(DistributedSampler):
	def __init__(
		self, dataset: Dataset, num_replicas: Optional[int] = None,
		rank: Optional[int] = None, shuffle: bool = True,
		seed: int = 0, drop_last: bool = False
	) -> None:
		super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
		self.resume_index = None
	
	def set_state(self, epoch: int, index: int) -> None:
		"""
		Sets the epoch and fast forwards the iterator to the given index.
		Needs to be called before the dataloader is iterated over.
		"""
		self.set_epoch(epoch)
		self.resume_index = index

	def __iter__(self):
		i = super().__iter__()

		if self.resume_index is not None:
			for _ in range(self.resume_index):
				next(i)
			self.resume_index = None
		
		return i


if __name__ == "__main__":
	distributed_setup()
	torch.cuda.set_device(distributed_rank())
	main()
	distributed_cleanup()
