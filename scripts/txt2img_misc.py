"""
Miscellaneous operations, such as save latent and noise
"""
import os
import sys

if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    gpu = '5'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(f"os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu}'")
else:
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"Existing: CUDA_VISIBLE_DEVICES: {gpu}")
cur_dir = os.path.dirname(os.path.abspath(__file__))
prt_dir = os.path.dirname(cur_dir)  # parent dir
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
    print(f"sys.path.append({cur_dir})")
if prt_dir not in sys.path:
    sys.path.append(prt_dir)
    print(f"sys.path.append({prt_dir})")

import argparse
import torch
import time
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from utils import log_info, get_time_ttl_and_eta

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=f"{prt_dir}/configs/stable-diffusion/v2-inference.yaml")
    parser.add_argument("--model", type=str, choices=['plms', 'dpm', 'ddim'], default="ddim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--prompt", type=str, default="a bedroom with bright window")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="down-sampling factor, most often 8 or 16")
    parser.add_argument("--scale", type=float, default=9.0)
    # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    parser.add_argument("--ckpt", type=str, default="./checkpoints/v2-1_512-ema-pruned.ckpt")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--bf16", action='store_true', help="Use float16")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    opt = parser.parse_args()
    from torch.backends import cudnn
    cudnn.benchmark = True
    return opt

def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    log_info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        log_info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        log_info("missing keys:")
        log_info(m)
    if len(u) > 0 and verbose:
        log_info("unexpected keys:")
        log_info(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model

def save_latent_and_noise():
    opt = parse_args()
    seed_everything(opt.seed)
    log_info(f"opt: {opt}")

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    log_info(f"seed  : {opt.seed}")
    log_info(f"config: {config}")
    log_info(f"device: {device}")
    log_info(f"ckpt  : {opt.ckpt}")
    model = load_model_from_config(config, f"{opt.ckpt}", device)
    sampler = DDIMSampler(model, device=device)
    log_info(f"sampler = DDIMSampler(model, device={device})")

    batch_size = opt.batch_size
    steps      = opt.steps
    b_cnt      = opt.n_iter
    out_dir    = opt.output_dir
    os.makedirs(out_dir, exist_ok=True)
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    noise_dir  = os.path.join(out_dir, f"sd_{shape[0]}x{shape[1]}x{shape[2]}_bs{batch_size}_noise")
    latent_dir = os.path.join(out_dir, f"sd_{shape[0]}x{shape[1]}x{shape[2]}_bs{batch_size}_latent_steps{steps}")
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)
    log_info(f"batch_size  : {batch_size}")
    log_info(f"b_cnt       : {b_cnt}")
    log_info(f"C           : {opt.C}")
    log_info(f"H           : {opt.H}")
    log_info(f"W           : {opt.W}")
    log_info(f"f           : {opt.f}")
    log_info(f"latent shape: {shape}")
    log_info(f"steps       : {steps}")
    log_info(f"out_dir     : {out_dir}")
    log_info(f"noise_dir   : {noise_dir}")
    log_info(f"latent_dir  : {latent_dir}")

    precision_scope = autocast
    uc = model.get_learned_conditioning(batch_size * [""])
    prompts = batch_size * [opt.prompt]
    c = model.get_learned_conditioning(prompts)
    time_start = time.time()
    with torch.no_grad(), precision_scope(opt.device), model.ema_scope():
        for b_idx in range(b_cnt):
            start_noise = torch.randn((batch_size, *shape), device=opt.device)
            samples, _ = sampler.sample(S=steps,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc,
                                        eta=opt.ddim_eta,
                                        x_T=start_noise)
            elp, eta = get_time_ttl_and_eta(time_start, b_idx+1, b_cnt)
            log_info(f"B{b_idx:3d}/{b_cnt}. elp:{elp}, eta:{eta}")
            f_path = os.path.join(noise_dir, f"noise_batch{b_idx:04d}.pt")
            torch.save(start_noise, f_path)
            if b_idx == 0:
                log_info(f"  saved: {f_path}")
            f_path = os.path.join(latent_dir, f"latent_batch{b_idx:04d}.pt")
            torch.save(samples, f_path)
            if b_idx == 0:
                log_info(f"  saved: {f_path}")
        # for n_iter
    # with

def main():
    save_latent_and_noise()

if __name__ == "__main__":
    main()
