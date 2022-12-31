import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig
from datasets import load_dataset

from pathlib import Path
import itertools
import functools

import numpy as np
import wandb
import random
import argparse


parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--device', type=str, default="cuda:1", help='Device to use')
parser.add_argument('--wandb_sweep', type=str, default=None, help='Wandb sweep id')
parser.add_argument('--dataset', type=str, help='Dataset to use')


only_subreddit = True   # Only use the subreddit in the prompt, and ignore similarity score
LR_SCHEDULER_TYPE = "constant_with_warmup"

optimizer_class = torch.optim.AdamW


class FinetuneDataset(Dataset):
	def __init__(self, tokenizer, text_encoder, device, dataset, only_subreddit, is_validation):
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.text_encoder = text_encoder
		self.device = device
		self.only_subreddit = only_subreddit
		self.is_validation = is_validation
	
	def __len__(self):
		if self.is_validation:
			return len(self.dataset)
		return 99999999 #len(self.dataset)
	
	def __getitem__(self, idx):
		row = self.dataset[idx % len(self.dataset)]
		title = row['title']
		subreddit = row['subreddit']
		latent = row['latent']

		# 10% dropping of the text conditioning
		if random.random() > 0.1:
			if not only_subreddit:
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


def main(args, config=None):
	with wandb.init(config=config):
		config = wandb.config
		batch_size = min(config.batch_size, args.batch_size)
		gradient_accumulation_steps = config.batch_size // batch_size

		print(f"batch_size: {batch_size}")
		print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
		print(f"target_batch_size: {config.batch_size}")

		# Load models
		tokenizer = AutoTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer", revision=None, use_fast=False)
		text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", revision=None)
		#vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", revision=None)
		unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", revision=None)

		noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

		#unet.set_use_memory_efficient_attention_xformers(True)

		#vae.to("cuda:1")
		text_encoder.to(args.device)
		unet.to(args.device)

		if not config.train_text_encoder:
			text_encoder.requires_grad_(False)
		#vae.requires_grad_(False)

		# Load datasets
		source_dataset = load_dataset(args.dataset)
		source_dataset.set_format("torch")
		dataset = FinetuneDataset(tokenizer, text_encoder, args.device, source_dataset["train"], only_subreddit, False)
		validation_dataset = FinetuneDataset(tokenizer, text_encoder, args.device, source_dataset["validation"], only_subreddit, True)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)

		params_to_optimize = unet.parameters() if not config.train_text_encoder else itertools.chain(unet.parameters(), text_encoder.parameters())
		optimizer = optimizer_class(
			params_to_optimize,
			lr=config.learning_rate,
			betas=(config.adam_beta1, config.adam_beta2),
			weight_decay=config.adam_weight_decay,
			eps=config.adam_epsilon,
		)

		total_batches = config.max_train_samples // batch_size
		total_steps = total_batches // gradient_accumulation_steps

		# The configured warmup_steps is actually the number of warmup samples, so we divide by the number of samples per step
		warmup_steps = config.warmup_steps // config.batch_size

		lr_scheduler = get_scheduler(
			LR_SCHEDULER_TYPE,
			optimizer=optimizer,
			num_warmup_steps=warmup_steps,
			num_training_steps=total_steps,
		)

		unet.train()
		if config.train_text_encoder:
			text_encoder.train()
		loss_sum = 0
		global_step = 0
		epoch_samples = 0

		with tqdm(total=total_batches) as pbar:
			for step, batch in enumerate(dataloader):
				latents = batch["latents"].to(args.device)

				noise = torch.randn_like(latents)
				bsz = latents.shape[0]
				timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
				timesteps = timesteps.long()

				noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

				encoder_hidden_states = [text_encoder(torch.asarray(input_id).to(args.device), output_hidden_states=True).last_hidden_state[0] for input_id in batch['input_ids']]
				encoder_hidden_states = torch.stack(tuple(encoder_hidden_states))
				#encoder_hidden_states = text_encoder(batch["input_ids"].to('cuda:1'))[0]
				#encoder_hidden_states = batch["input_ids"]

				model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

				loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

				loss = loss / gradient_accumulation_steps
				loss.backward()

				loss_sum += loss.detach().item()
				pbar.update(1)
				epoch_samples += batch_size

				if (step + 1) % gradient_accumulation_steps == 0:
					torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
					if config.train_text_encoder:
						torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), config.max_grad_norm)
					optimizer.step()
					lr_scheduler.step()
					optimizer.zero_grad()
					global_step += 1
					logs = {
						"train/loss": loss_sum,
						"train/lr": lr_scheduler.get_last_lr()[0],
						"train/step": global_step,
						"train/samples": global_step * batch_size * gradient_accumulation_steps,
					}
					wandb.log(logs, step=global_step)
					loss_sum = 0
				
				if epoch_samples >= config.samples_per_epoch:
					epoch_samples = 0
					# Note I'm using batch_size=1 here, because there was some reproducibility issue with batch_size > 1 (I think...)
					validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=validation_dataset.collate_fn)
					validation_loss = 0
					validation_steps = 0

					# Set seed for validation consistency
					old_torch_rng_state = torch.get_rng_state()
					torch.manual_seed(42)

					# Validation
					with torch.no_grad():
						for batch in validation_dataloader:
							latents = batch["latents"].to(args.device)

							noise = torch.randn_like(latents)
							bsz = latents.shape[0]
							timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
							timesteps = timesteps.long()

							noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

							encoder_hidden_states = [text_encoder(torch.asarray(input_id).to(args.device), output_hidden_states=True).last_hidden_state[0] for input_id in batch['input_ids']]
							encoder_hidden_states = torch.stack(tuple(encoder_hidden_states))

							model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

							loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

							validation_loss += loss.detach().item()
							validation_steps += 1
					
					# Restore PyTorch RNG state
					torch.set_rng_state(old_torch_rng_state)

					validation_loss /= validation_steps
					logs = {
						"validation/loss": validation_loss,
						"validation/samples": global_step * batch_size * gradient_accumulation_steps,
					}
					wandb.log(logs, step=global_step)

				if global_step >= total_steps:
					break
		
		return unet, text_encoder



if __name__ == "__main__":
	args = parser.parse_args()

	if args.wandb_sweep is not None:
		wandb.agent(args.wandb_sweep, functools.partial(main, args), count=1)
	else:
		config = {
			"batch_size": args.batch_size,
			"gradient_accumulation_steps": 1,
			"adam_beta1": 0.9,
			"adam_beta2": 0.999,
			"adam_weight_decay": 0.01,
			"adam_epsilon": 1e-8,
			"max_grad_norm": 1.0,
			"learning_rate": 5e-6,
			"train_text_encoder": True,
			"clip_penultimate": False,
			"warmup_steps": 100,
			"max_train_samples": 8 * 1024,
			"use_8bit_adam": False,
			"samples_per_epoch": 256,
		}

		main(args, config=config)