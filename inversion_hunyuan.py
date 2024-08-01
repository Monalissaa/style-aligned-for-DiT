# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
from typing import Callable
from diffusers import HunyuanDiTPipeline
from diffusers.models.embeddings import get_2d_rotary_pos_embed
import torch
from tqdm import tqdm
import numpy as np


T = torch.Tensor
TN = T | None
InversionCallback = Callable[[HunyuanDiTPipeline, int, T, dict[str, T]], dict[str, T]]


def _get_text_embeddings(prompt: str, tokenizer, text_encoder, device):
    # Tokenize text and get embeddings
    text_inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            output_hidden_states=True,
        )

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    if prompt == '':
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        return negative_prompt_embeds, negative_pooled_prompt_embeds
    return prompt_embeds, pooled_prompt_embeds


def _encode_text_sdxl(model: HunyuanDiTPipeline, prompt: str) -> tuple[dict[str, T], T]:
    device = model._execution_device
    prompt_embeds, pooled_prompt_embeds, = _get_text_embeddings(prompt, model.tokenizer, model.text_encoder, device)
    prompt_embeds_2, pooled_prompt_embeds2, = _get_text_embeddings( prompt, model.tokenizer_2, model.text_encoder_2, device)
    prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)
    text_encoder_projection_dim = model.text_encoder_2.config.projection_dim
    add_time_ids = model._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), torch.float16,
                                           text_encoder_projection_dim).to(device)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds2, "time_ids": add_time_ids}
    return added_cond_kwargs, prompt_embeds


def _encode_text_sdxl_with_negative(model: HunyuanDiTPipeline, prompt: str) -> tuple[dict[str, T], T]:
    added_cond_kwargs, prompt_embeds = _encode_text_sdxl(model, prompt)
    added_cond_kwargs_uncond, prompt_embeds_uncond = _encode_text_sdxl(model, "")
    prompt_embeds = torch.cat((prompt_embeds_uncond, prompt_embeds, ))
    added_cond_kwargs = {"text_embeds": torch.cat((added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"])),
                         "time_ids": torch.cat((added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"])),}
    return added_cond_kwargs, prompt_embeds

def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def _encode_text_hunyuan_with_negative(model: HunyuanDiTPipeline, prompt: str) -> tuple[dict[str, T], T]:
    device = model._execution_device
    ## paramters
    num_images_per_prompt = 1
    batch_size = 1
    do_classifier_free_guidance = True


    (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
    ) = model.encode_prompt(
        prompt=prompt,
        device=device,
        dtype=model.transformer.dtype,
        num_images_per_prompt=num_images_per_prompt, # origin num_images_per_prompt
        do_classifier_free_guidance=do_classifier_free_guidance, # origin self.do_classifier_free_guidance
        # negative_prompt=negative_prompt,
        # prompt_embeds=prompt_embeds,
        # negative_prompt_embeds=negative_prompt_embeds,
        # prompt_attention_mask=prompt_attention_mask,
        # negative_prompt_attention_mask=negative_prompt_attention_mask,
        max_sequence_length=77,
        text_encoder_index=0,
    )
    (
        prompt_embeds_2,
        negative_prompt_embeds_2,
        prompt_attention_mask_2,
        negative_prompt_attention_mask_2,
    ) = model.encode_prompt(
        prompt=prompt,
        device=device,
        dtype=model.transformer.dtype,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        # negative_prompt=negative_prompt,
        # prompt_embeds=prompt_embeds_2,
        # negative_prompt_embeds=negative_prompt_embeds_2,
        # prompt_attention_mask=prompt_attention_mask_2,
        # negative_prompt_attention_mask=negative_prompt_attention_mask_2,
        max_sequence_length=256,
        text_encoder_index=1,
    )
    generator, eta = None, 0.0
    extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)
    height, width = 1024, 1024

    grid_height = height // 8 // model.transformer.config.patch_size
    grid_width = width // 8 // model.transformer.config.patch_size
    base_size = 512 // 8 // model.transformer.config.patch_size
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
    image_rotary_emb = get_2d_rotary_pos_embed(
        model.transformer.inner_dim // model.transformer.num_heads, grid_crops_coords, (grid_height, grid_width)
    )

    style = torch.tensor([0], device=device)

    original_size = (1024, 1024)
    # target_size = target_size or (height, width)
    target_size = (1024, 1024)
    crops_coords_top_left = (0, 0)

    add_time_ids = list(original_size + target_size + crops_coords_top_left)
    add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
        prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
        prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])
        add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
        style = torch.cat([style] * 2, dim=0)

    prompt_embeds = prompt_embeds.to(device=device)
    prompt_attention_mask = prompt_attention_mask.to(device=device)
    prompt_embeds_2 = prompt_embeds_2.to(device=device)
    prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)
    add_time_ids = add_time_ids.to(dtype=prompt_embeds.dtype, device=device).repeat(
        batch_size * num_images_per_prompt, 1
    )
    style = style.to(device=device).repeat(batch_size * num_images_per_prompt)
    model._num_timesteps = len(model.scheduler.timesteps)

    added_cond_kwargs = {
        'extra_step_kwargs':extra_step_kwargs, 
        'image_rotary_emb': image_rotary_emb,
        'style': style,
        'add_time_ids': add_time_ids,
        'prompt_attention_mask': prompt_attention_mask,
        'prompt_attention_mask_2': prompt_attention_mask_2,
        }

    # added_cond_kwargs, prompt_embeds = _encode_text_sdxl(model, prompt)
    # added_cond_kwargs_uncond, prompt_embeds_uncond = _encode_text_sdxl(model, "")
    # prompt_embeds = torch.cat((prompt_embeds_uncond, prompt_embeds, ))
    # added_cond_kwargs = {"text_embeds": torch.cat((added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"])),
    #                      "time_ids": torch.cat((added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"])),}
    return added_cond_kwargs, prompt_embeds, prompt_embeds_2



def _encode_image(model: HunyuanDiTPipeline, image: np.ndarray) -> T:
    model.vae.to(dtype=torch.float32)
    image = torch.from_numpy(image).float() / 255.
    image = (image * 2 - 1).permute(2, 0, 1).unsqueeze(0)
    latent = model.vae.encode(image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor
    model.vae.to(dtype=torch.float16)
    return latent


def _next_step(model: HunyuanDiTPipeline, model_output: T, timestep: int, sample: T) -> T:
    timestep, next_timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = model.scheduler.alphas_cumprod[int(timestep)] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[int(next_timestep)]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def _get_noise_pred(model: HunyuanDiTPipeline, latent: T, t: T, context: T, context_2: T, guidance_scale: float, added_cond_kwargs: dict[str, T]):
    device = model._execution_device
    # expand the latents if we are doing classifier free guidance
    latents_input = torch.cat([latent] * 2)
    latents_input = model.scheduler.scale_model_input(latents_input, t)

    # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
    t_expand = torch.tensor([t] * latents_input.shape[0], device=device).to(
        dtype=latents_input.dtype
    )

    noise_pred = model.transformer(
                    latents_input,
                    t_expand,
                    encoder_hidden_states=context,
                    text_embedding_mask=added_cond_kwargs['prompt_attention_mask'],
                    encoder_hidden_states_t5=context_2,
                    text_embedding_mask_t5=added_cond_kwargs['prompt_attention_mask_2'],
                    image_meta_size=added_cond_kwargs['add_time_ids'],
                    style=added_cond_kwargs['style'],
                    image_rotary_emb=added_cond_kwargs['image_rotary_emb'],
                    return_dict=False,
                )[0]
    noise_pred, _ = noise_pred.chunk(2, dim=1)

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # latents = next_step(model, noise_pred, t, latent)
    return noise_pred

def _ddim_loop(model: HunyuanDiTPipeline, z0, prompt, guidance_scale) -> T:
    all_latent = [z0]
    added_cond_kwargs, text_embedding, text_embedding_2 = _encode_text_hunyuan_with_negative(model, prompt)
    latent = z0.clone().detach().half()
    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        noise_pred = _get_noise_pred(model, latent, t, text_embedding, text_embedding_2, guidance_scale, added_cond_kwargs)
        latent = _next_step(model, noise_pred, t, latent)
        all_latent.append(latent)
    return torch.cat(all_latent).flip(0)


def make_inversion_callback(zts, offset: int = 0) -> [T, InversionCallback]:

    def callback_on_step_end(pipeline: HunyuanDiTPipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]:
        latents = callback_kwargs['latents']
        latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        return {'latents': latents}
    return  zts[offset], callback_on_step_end


@torch.no_grad()
def ddim_inversion(model: HunyuanDiTPipeline, x0: np.ndarray, prompt: str, num_inference_steps: int, guidance_scale,) -> T:
    z0 = _encode_image(model, x0)
    model.scheduler.set_timesteps(num_inference_steps, device=z0.device)
    # timesteps: [980 960 940 920 900 880 860 840 820 800 780 760 740 720 700 680 660 640 620 600 580 560 540 520 500 480 460 440 420 400 380 360 340 320 300 280 260 240 220 200 180 160 140 120 100 80 60 40 20 0]
    zs = _ddim_loop(model, z0, prompt, guidance_scale)
    return zs
