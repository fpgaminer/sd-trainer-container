import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig, CLIPFeatureExtractor
from datasets import load_dataset
from huggingface_hub import Repository

from pathlib import Path
import itertools

import numpy as np
import wandb
import random
import argparse
import os
import math
import gc


bool_arg = lambda x: (str(x).lower() in ['true', '1', 't', 'y', 'yes'])
parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--output_dir', type=str, default="output", help='Output directory')
parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
parser.add_argument('--device', type=str, default="cuda:1", help='Device to use')
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
parser.add_argument('--test_every', type=int, default=10000, help='Test every n samples (approx)')
parser.add_argument('--fp16', type=bool_arg, default=False, help='Use fp16')
parser.add_argument('--train_text_encoder', type=bool_arg, default=True, help='Train the text encoder')
parser.add_argument('--lr_scheduler_type', type=str, default="constant_with_warmup", help='Learning rate scheduler type')
parser.add_argument('--allow_tf32', type=bool_arg, default=False, help='Allow tf32')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--hf_repo_id', type=str, default=None, help='If set, will save the model to the huggingface hub with this id')
parser.add_argument('--xformers', type=bool_arg, default=False, help='Use xformers')


def main():
	args = parser.parse_args()

	save_image_prompts = [s.strip() for s in args.save_image_prompts.split("|")]

	def runner(args, config):
		with wandb.init(config=config, project=args.wandb_project):
			trainer = MainTrainer(
				config=wandb.config,
				output_dir=args.output_dir,
				resume=args.resume,
				device=args.device,
				device_batch_size=args.device_batch_size,
				dataset=args.dataset,
				save_every=args.save_every,
				save_images_every=args.save_images_every,
				save_image_prompts=save_image_prompts,
				test_every=args.test_every,
				hf_repo_id=args.hf_repo_id,
			)

			trainer.train()

	if args.wandb_sweep is not None:
		wandb.agent(args.wandb_sweep, lambda: runner(args, None), count=1)
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
		}

		runner(args, config)


class FinetuneDataset(Dataset):
	def __init__(self, tokenizer, text_encoder, device, dataset, only_subreddit, is_validation):
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.text_encoder = text_encoder
		self.device = device
		self.only_subreddit = only_subreddit
		self.is_validation = is_validation
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		row = self.dataset[idx % len(self.dataset)]
		title = row['title']
		subreddit = row['subreddit']
		latent = row['latent']

		# 10% dropping of the text conditioning
		if random.random() > 0.1:
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

		#if clip_penultimate:
		#	input_ids = [self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True)['hidden_states'][-2])[0] for input_id in input_ids]
		#else:
		#	input_ids = [self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True).last_hidden_state[0] for input_id in input_ids]

		#input_ids = torch.cat(prompt_ids, dim=0)
		#input_ids = torch.stack(tuple(input_ids))

		return {
			'latents': latents,
			'input_ids': input_ids,
		}


class MainTrainer:
	def __init__(self, config, output_dir, resume, device, device_batch_size, dataset, save_every, save_images_every, save_image_prompts, test_every, hf_repo_id):
		self.output_dir = output_dir
		self.resume = resume
		self.device = device
		self.dataset = dataset
		self.save_image_prompts = save_image_prompts
		self.hf_repo_id = hf_repo_id

		self.batch_size = config.batch_size
		self.adam_beta1 = config.adam_beta1
		self.adam_beta2 = config.adam_beta2
		self.adam_weight_decay = config.adam_weight_decay
		self.adam_epsilon = config.adam_epsilon
		self.max_grad_norm = config.max_grad_norm
		self.learning_rate = config.learning_rate
		self.train_text_encoder = config.train_text_encoder
		self.clip_penultimate = config.clip_penultimate
		self.warmup_samples = config.warmup_samples
		self.max_samples = config.max_samples
		self.use_8bit_adam = config.use_8bit_adam
		self.lr_scheduler_type = config.lr_scheduler_type
		self.allow_tf32 = config.allow_tf32
		self.only_subreddit = config.only_subreddit
		self.fp16 = config.fp16
		self.seed = config.seed
		self.xformers = config.xformers

		os.makedirs(self.output_dir, exist_ok=True)

		if self.xformers and not is_xformers_available():
			raise ValueError("xformers is not installed`")

		if self.allow_tf32:
			torch.backends.cuda.matmul.allow_tf32 = True
		
		# Calculate the batch size and gradient accumulation steps to get the target batch size
		self.device_batch_size = min(self.batch_size, device_batch_size)
		self.gradient_accumulation_steps = self.batch_size // self.device_batch_size
		self.test_every = test_every // self.batch_size
		self.save_every = save_every // self.batch_size
		self.save_images_every = save_images_every // self.batch_size

		assert self.batch_size == self.device_batch_size * self.gradient_accumulation_steps, f"batch_size {self.batch_size} must be divisible by device_batch_size {device_batch_size}"
	
	def load_models(self):
		model_path = "CompVis/stable-diffusion-v1-4"
		if self.resume is not None:
			model_path = self.resume
			self.resume_global_step = int(self.resume.split("-")[-1])
		
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
	
	def build_optimizer(self):
		# Optimizer
		if self.use_8bit_adam:
			import bitsandbytes
			optimizer_class = bitsandbytes.optim.AdamW8bit
		else:
			optimizer_class = torch.optim.AdamW
		
		params_to_optimize = self.unet.parameters() if not self.train_text_encoder else itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
		self.optimizer = optimizer_class(
			params_to_optimize,
			lr=self.learning_rate,
			betas=(self.adam_beta1, self.adam_beta2),
			weight_decay=self.adam_weight_decay,
			eps=self.adam_epsilon,
		)
	
	def build_datasets(self):
		self.source_dataset = load_dataset(self.dataset)
		self.source_dataset.set_format("torch")
		self.dataset = FinetuneDataset(self.tokenizer, self.text_encoder, self.device, self.source_dataset["train"], self.only_subreddit, False)
		self.validation_dataset = FinetuneDataset(self.tokenizer, self.text_encoder, self.device, self.source_dataset["validation"], self.only_subreddit, True)

		self.total_steps = self.max_samples // self.batch_size
		self.total_device_batches = self.total_steps * self.gradient_accumulation_steps
	
	def build_dataloader(self):
		self.dataloader = iter(torch.utils.data.DataLoader(self.dataset, batch_size=self.device_batch_size, shuffle=True, num_workers=4, collate_fn=self.dataset.collate_fn))

	def train(self):
		if self.seed is not None:
			torch.manual_seed(self.seed)
			np.random.seed(self.seed)
			random.seed(self.seed)

		scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
		self.load_models()
		self.build_optimizer()
		self.build_datasets()
		self.build_dataloader()
		self.init_git_repo()

		lr_scheduler = get_scheduler(
			self.lr_scheduler_type,
			optimizer=self.optimizer,
			num_warmup_steps=self.warmup_samples // self.batch_size,
			num_training_steps=None,
		)

		loss_sum = 0

		for device_step in tqdm(range(self.total_device_batches)):
			self.global_step = device_step // self.gradient_accumulation_steps

			self.unet.train()
			if self.train_text_encoder:
				self.text_encoder.train()
			
			# Reload the dataloader as needed
			try:
				batch = next(self.dataloader)
			except StopIteration:
				self.build_dataloader()
				batch = next(self.dataloader)
			
			# TODO: Is there a more efficient way to do this? As is, we have to load all that data for nothing
			if self.resume is not None and self.global_step <= self.resume_global_step:
				# NOTE: resume_global_step represents the state "global_step" was at when the checkpoint was saved.
				# NOTE: Hence, resume_global_step represents an optimizer step that has already been taken.
				# NOTE: Therefore, we use "<=" so global_step = resume_global_step + 1 at the end of this loop.
				continue
		
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

			# Forward pass
			with torch.autocast('cuda', enabled=self.fp16):
				model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

				# Loss
				# TODO: Not sure if this should be under autocast; PyTorch docs show an example with it under autocast, but other repos don't
				loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
				loss = loss / self.gradient_accumulation_steps

			# Backward pass
			scaler.scale(loss).backward()

			loss_sum += loss.detach().item()

			# Take a step if accumulation is complete
			if (device_step + 1) % self.gradient_accumulation_steps == 0:
				# Unscale gradients before clipping
				scaler.unscale_(self.optimizer)

				# Clip gradients
				torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
				if self.train_text_encoder:
					torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), self.max_grad_norm)
				
				# Take a step
				scaler.step(self.optimizer)
				scaler.update()
				lr_scheduler.step()
				self.optimizer.zero_grad()
				logs = {
					"train/loss": loss_sum,
					"train/lr": lr_scheduler.get_last_lr()[0],
					"train/step": self.global_step,
					"train/samples": self.global_step * self.batch_size,
				}
				wandb.log(logs, step=self.global_step)
				loss_sum = 0

				# Save checkpoint
				if (self.global_step + 1) % self.save_every == 0 or (self.global_step + 1) == self.total_steps:
					self.save_checkpoint()
				
				# Validation
				if self.test_every > 0 and (self.global_step + 1) % self.test_every == 0:
					self.do_validation()
				
				# Sample image generation
				if self.save_images_every > 0 and (self.global_step + 1) % self.save_images_every == 0:
					self.do_image_samples()

	def save_checkpoint(self):
		if self.repo is not None:
			path = self.output_dir
		else:
			path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
		
		pipeline = StableDiffusionPipeline(
			text_encoder=self.text_encoder,
			vae=AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", revision=None),
			unet=self.unet,
			tokenizer=self.tokenizer,
			scheduler=PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler", revision=None), # TODO: Why is this PNDMScheduler whereas from scratch it's DDPM?
			safety_checker=None,
			feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
		)
		print(f"Saving model checkpoint to {path}")

		if self.repo is not None:
			with self.repo.commit(commit_message=f"Training checkpoint {self.global_step}", blocking=False, auto_lfs_prune=True):
				pipeline.save_pretrained(path)
		else:
			pipeline.save_pretrained(path)

	def do_validation(self):
		self.unet.eval()
		self.text_encoder.eval()

		# Note: I'm using batch_size=1 here, because there was some reproducibility issue with batch_size > 1
		# TODO: Investigate this
		validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=self.validation_dataset.collate_fn)
		validation_loss = 0
		validation_steps = 0

		# Set seed for validation consistency
		old_torch_rng_state = torch.get_rng_state()
		torch.manual_seed(42)

		with torch.no_grad():
			for batch in validation_dataloader:
				latents = batch["latents"].to(self.device, dtype=self.weight_dtype)  # TODO: see the todo at the other instance of this

				noise = torch.randn_like(latents)
				bsz = latents.shape[0]
				timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
				timesteps = timesteps.long()

				noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

				encoder_hidden_states = [self.text_encoder(torch.asarray(input_id).to(self.device), output_hidden_states=True).last_hidden_state[0] for input_id in batch['input_ids']]
				encoder_hidden_states = torch.stack(tuple(encoder_hidden_states))

				with torch.autocast('cuda', enabled=self.fp16):
					model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

					loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

				validation_loss += loss.detach().item()
				validation_steps += 1
		
		# Restore PyTorch RNG state
		torch.set_rng_state(old_torch_rng_state)

		validation_loss /= validation_steps
		logs = {
			"validation/loss": validation_loss,
			"validation/samples": self.global_step * self.batch_size,
		}
		wandb.log(logs, step=self.global_step)

	def do_image_samples(self):
		self.unet.eval()
		self.text_encoder.eval()

		generator = torch.Generator(self.device).manual_seed(42)
		scheduler = EulerAncestralDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
		pipeline = StableDiffusionPipeline(
			text_encoder=self.text_encoder,
			vae=self.vae,
			unet=self.unet,
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
					images += [wandb.Image(img) for img in pipeline(batch_prompts, num_inference_steps=28, negative_prompt=negative_prompts, num_images_per_prompt=1, guidance_scale=11, generator=generator).images]
		
		wandb.log({"images/images": images}, step=self.global_step)

		del pipeline
		gc.collect()

	def init_git_repo(self):
		if self.hf_repo_id is None:
			self.repo = None
		else:
			self.repo = Repository(self.output_dir, clone_from=self.hf_repo_id, private=True)
			self.repo.git_pull()


if __name__ == "__main__":
	main()
