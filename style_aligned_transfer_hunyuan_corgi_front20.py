from diffusers import HunyuanDiTPipeline, DDIMScheduler
import torch
import mediapy
import math

import argparse
parser = argparse.ArgumentParser(description='Load a HunyuanDiT model with specific share layer setting.')

# 添加命令行参数
parser.add_argument('--specific_share_layer', type=int, default=-1, help='The specific share layer value.')

# 解析命令行参数
args = parser.parse_args()


scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.02, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=True, 
    prediction_type='v_prediction', steps_offset=1)



pipeline = HunyuanDiTPipeline.from_pretrained(
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True,
    scheduler=scheduler
).to("cuda")

# DDIM inversion

from diffusers.utils import load_image
import inversion_hunyuan as inversion
import numpy as np

src_style = "medieval painting"
src_prompt = 'A brown and white corgi sits on an orange background, looking up at the camera. The photo is close-up, centered, and head-up, showing the real corgi and creating a warm atmosphere.'
image_path = '/place/HunyuanDiT/dataset/dog/images/02.jpg'

num_inference_steps = 50
x0 = np.array(load_image(image_path).resize((1024, 1024)))
zts = inversion.ddim_inversion(pipeline, x0, src_prompt, num_inference_steps, 2)
# mediapy.show_image(x0, title="innput reference image", height=256)

import sa_handler_for_hunyuan as sa_handler
prompts = [
    src_prompt,
    "A brown and white corgi is swimming. The photo is close-up, centered, and head-up, showing the real corgi and creating a warm atmosphere.",
    "A brown and white corgi sits on an orange background, wearing a hat, looking up at the camera. The photo is close-up, centered, and head-up, showing the real corgi and creating a warm atmosphere.",
    "A brown and white corgi in a bucket, looking up at the camera. The photo is close-up, centered, and head-up, showing the real corgi and creating a warm atmosphere.",
    "A brown and white corgi sits, in the style of van gogh, looking up at the camera. The photo is close-up, centered, and head-up, creating a warm atmosphere.",
]

# some parameters you can adjust to control fidelity to reference
shared_score_shift = 0 # np.log(2)  # higher value induces higher fidelity, set 0 for no shift
shared_score_scale = 1.0  # higher value induces higher, set 1 for no rescale

# for very famouse images consider supressing attention to refference, here is a configuration example:
# shared_score_shift = np.log(1)
# shared_score_scale = 0.5

# for i in range(1, len(prompts)):
#     prompts[i] = f'{prompts[i]}, {src_style}.'

handler = sa_handler.Handler(pipeline)
sa_args = sa_handler.StyleAlignedArgs(
    share_group_norm=False, 
    share_layer_norm=False, 
    share_attention=True,
    adain_queries=False, 
    adain_keys=False, 
    adain_values=False,
    shared_score_shift=shared_score_shift, 
    shared_score_scale=shared_score_scale, 
    number_of_share_layer=29,
    not_share_layers=[1, 3]
    )
handler.register(sa_args)

zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)

g_cpu = torch.Generator(device='cpu')
g_cpu.manual_seed(10)

latents = torch.randn(len(prompts), 4, 128, 128, device='cpu', generator=g_cpu,
                      dtype=pipeline.transformer.dtype,).to('cuda:0')

latents[0] = zT

images_a = pipeline(prompts, latents=latents,
                    callback_on_step_end=inversion_callback,
                    num_inference_steps=num_inference_steps, guidance_scale=10.0).images

handler.remove()
# mediapy.show_images(images_a, titles=[p[:-(len(src_style) + 3)] for p in prompts])
prompts_simple = [
    'corgi',
    "is swimming",
    "wearing a hat",
    "in a bucket",
    "in the style of van gogh",
]


for i in range(len(images_a)):
    image = images_a[i]
    image.save('/place/style-aligned-for-DiT/results/recon_share_front20_except1_3/recon_recon_share_front29_except1_3_{}.png'.format(prompts_simple[i].replace(' ', '_')))