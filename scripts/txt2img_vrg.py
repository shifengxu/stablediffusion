"""
text to image, for VRG: variance reduction guidance
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
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config
from utils import log_info

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--todo", type=str, default="sample_track_schedule_regen_merge")
    parser.add_argument("--todo", type=str, default="sample_track_schedule_regen_merge")
    parser.add_argument("--config", type=str, default=f"{prt_dir}/configs/stable-diffusion/v2-inference.yaml")
    parser.add_argument("--model", type=str, choices=['plms', 'dpm', 'ddim'], default="ddim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps_arr", type=int, nargs='+', default=[10])
    parser.add_argument("--sch_lp_arr", type=float, nargs='+', default=[0.01], help='scheduler learning-portion')
    parser.add_argument("--fid_input1", type=str, default="./datasets_for_training/sd_gen_bedroom_512x512")
    parser.add_argument("--n_samples", type=int, default=22)
    parser.add_argument("--s_batch_init_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=".")
    # parser.add_argument("--prompt", type=str, default="a professional photograph of an astronaut riding a triceratops")
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
    opt = parser.parse_args()
    from torch.backends import cudnn
    cudnn.benchmark = True
    return opt

def get_model_and_sampler(opt):
    log_info(f"get_model_and_sampler()...")
    config = OmegaConf.load(f"{opt.config}")
    log_info(f"  config: {opt.config}")
    log_info(f"  device: {opt.device}")
    ckpt = opt.ckpt
    log_info(f"  Load ckpt from {ckpt}...")
    root_dict = torch.load(ckpt, map_location="cpu")
    log_info(f"  Load ckpt from {ckpt}...Done")
    log_info(f"  Create model...")
    model = instantiate_from_config(config.model)
    log_info(f"  Create model...Done")
    model.load_state_dict(root_dict["state_dict"], strict=False)
    device = opt.device
    if device == torch.device("cuda") or device == 'cuda':
        model.cuda()
        log_info(f"  model.cuda()")
    elif device == torch.device("cpu") or device == 'cpu':
        model.cpu()
        log_info(f"  model.cpu()")
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()

    # if opt.model == 'plms':
    #     sampler = PLMSSampler(model, device=device)
    #     log_info(f"sampler = PLMSSampler(model, device={device})")
    # elif opt.model == 'dpm':
    #     sampler = DPMSolverSampler(model, device=device)
    #     log_info(f"sampler = DPMSolverSampler(model, device={device})")
    # elif opt.model == 'ddim':
    #     sampler = DDIMSampler(model, device=device)
    #     log_info(f"sampler = DDIMSampler(model, device={device})")
    # else:
    #     raise ValueError(f"Invalid model: {opt.model}")
    from runner.ddim_vrg import DDIMSamplerVrg
    sampler = DDIMSamplerVrg(opt, model)
    log_info(f"  sampler = DDIMSamplerVrg(opt, model)")
    log_info(f"get_model_and_sampler()...Done")
    return model, sampler

def schedule(opt, model, old_file, new_file):
    from scheduler_vrg.vrg_scheduler import VrgScheduler
    sch = VrgScheduler(opt, model.alphas_cumprod)
    torch.set_grad_enabled(True)
    res = sch.schedule(f_path=old_file, output_file=new_file)
    torch.set_grad_enabled(False)
    return res

def sample_or_regen(opt, model, sampler, img_old_dir, trajectory_file, img_new_dir):
    """
    sample, or re-generate images by scheduled trajectory, or both.
    If img_old_dir is valid, then sample; if it is None, then not.
    If img_new_dir is valid, then regen; if it is None, then not.
    """
    def load_trajectory():
        with open(trajectory_file, 'r') as f:
            lines = f.readlines()
        _ab_arr, _ts_arr = [], [] # alpha_bar array, timestep array
        # line sample:
        #   # aacum : ts : alpha   ; coef    *weight     =numerator; numerator/aacum   =sub_var
        #   0.939064:  61: 0.942214; 0.036376* 334.118411=12.153861; 12.153861/0.939064= 12.942529
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            _ab, _ts = line.split(':')[0:2]
            _ab_arr.append(float(_ab))
            _ts_arr.append(int(_ts))
        return _ab_arr, _ts_arr

    log_info(f"sample_or_regen()...")
    log_info(f"  img_old_dir    : {img_old_dir}")
    log_info(f"  img_new_dir    : {img_new_dir}")

    if img_new_dir:
        log_info(f"  trajectory_file: {trajectory_file}")
        if not os.path.exists(trajectory_file):
            raise ValueError(f"File not exist: {trajectory_file}")
        ab_arr, ts_arr = load_trajectory()  # alpha_bar array
        ts_arr_desc = list(reversed(ts_arr))
        log_info(f"  ab_arr len: {len(ab_arr)}")
        log_info(f"  ab_arr[0] : {ab_arr[0]:.6f}  {ts_arr[0]:3d}")
        log_info(f"  ab_arr[1] : {ab_arr[1]:.6f}  {ts_arr[1]:3d}")
        log_info(f"  ab_arr[-2]: {ab_arr[-2]:.6f}  {ts_arr[-2]:3d}")
        log_info(f"  ab_arr[-1]: {ab_arr[-1]:.6f}  {ts_arr[-1]:3d}")
    else:
        ts_arr_desc = None

    assert opt.prompt is not None
    n_samples  = opt.n_samples
    batch_size = opt.batch_size
    batch_cnt = n_samples // batch_size
    if batch_cnt * batch_size < n_samples: batch_cnt += 1

    latent_c, latent_h, latent_w = opt.C, opt.H // opt.f, opt.W // opt.f
    log_info(f"  n_samples  : {n_samples}")
    log_info(f"  batch_size : {batch_size}")
    log_info(f"  batch_cnt  : {batch_cnt}")
    log_info(f"  opt.steps_arr: {opt.steps_arr}")
    log_info(f"  opt.C      : {opt.C}")
    log_info(f"  opt.H      : {opt.H}")
    log_info(f"  opt.W      : {opt.W}")
    log_info(f"  opt.f      : {opt.f}")
    log_info(f"  latent_h   : {latent_h}")
    log_info(f"  latent_w   : {latent_w}")

    latent_c, latent_h, latent_w = opt.C, opt.H // opt.f, opt.W // opt.f
    shape = [opt.C, latent_h, latent_w]

    def sample_batch_new(_ts, _c, _uc, _noise, _dir, _init_idx):
        samples, _ = sampler.sample2_batch(S=opt.steps_arr[0],
                                           ts_list_desc=_ts,
                                           conditioning=_c,
                                           unconditional_conditioning=_uc,
                                           unconditional_guidance_scale=opt.scale,
                                           x_T=_noise)

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        # if batch_size is 1, opt.f is 8, opt.H = opt.W = 512,
        # then shape:    [4,  64,  64]
        # sample    : [1, 4,  64,  64]
        # x_samples : [1, 3, 512, 512]

        path = None
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            path = os.path.join(img_new_dir, f"{_init_idx:05}.png")
            img.save(path)
            _init_idx += 1
        log_info(f"  Saved {len(x_samples)}: {path}")

    def sample_batch_old(_c, _uc, _noise, _dir, _init_idx):
        samples, _ = sampler.sample_batch(S=opt.steps_arr[0],
                                          conditioning=_c,
                                          batch_size=len(_noise),
                                          shape=shape,
                                          verbose=False,
                                          unconditional_guidance_scale=opt.scale,
                                          unconditional_conditioning=_uc,
                                          x_T=_noise)

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        # if batch_size is 1, opt.f is 8, opt.H = opt.W = 512,
        # then shape:    [4,  64,  64]
        # sample    : [1, 4,  64,  64]
        # x_samples : [1, 3, 512, 512]

        path = None
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            path = os.path.join(_dir, f"{_init_idx:05}.png")
            img.save(path)
            _init_idx += 1
        log_info(f"  Saved {len(x_samples)}: {path}")

    with torch.no_grad(), autocast(opt.device), model.ema_scope():
        for b_idx in range(0, batch_cnt):
            n = batch_size if b_idx < batch_cnt - 1 else n_samples - b_idx * batch_size
            prompts = n * [opt.prompt]
            c = model.get_learned_conditioning(prompts)
            uc = None if opt.scale == 1.0 else model.get_learned_conditioning(n * [""])
            start_code = torch.randn([n, latent_c, latent_h, latent_w], device=opt.device)
            init_idx = b_idx * batch_size + opt.s_batch_init_id
            if img_old_dir:
                sample_batch_old(c, uc, start_code, img_old_dir, init_idx)
            if img_new_dir:
                sample_batch_new(ts_arr_desc, c, uc, start_code, img_new_dir, init_idx)
        # for batch
    # with
    log_info(f"sample_or_regen()...Done")

def merge(opt, dir1, dir2, dir_merge, label1=None, label2=None):
    log_info(f"merge()...")
    log_info(f"  dir1     : {dir1}")
    log_info(f"  dir2     : {dir2}")
    log_info(f"  dir_merge: {dir_merge}")
    import cv2
    os.makedirs(dir_merge, exist_ok=True)
    label1 = label1 or f"{opt.steps_arr[0]} steps: Old"
    label2 = label2 or f"{opt.steps_arr[0]} steps: New"
    f_list = os.listdir(dir1)
    f_list.sort()  # file name list
    font_face, font_scale = 0, 0.7
    color_bgr, thickness = (0, 0, 255), 2
    for fn in f_list:
        f1 = os.path.join(dir1, fn)
        f2 = os.path.join(dir2, fn)
        f3 = os.path.join(dir_merge, fn)
        img1, img2 = cv2.imread(f1), cv2.imread(f2)
        cv2.putText(img1, label1, (5, 20), font_face, font_scale, color_bgr, thickness)
        cv2.putText(img2, label2, (5, 20), font_face, font_scale, color_bgr, thickness)
        img3 = np.concatenate([img1, img2], axis=1)
        cv2.imwrite(f3, img3)
        log_info(f"  saved: {f3}")
    log_info(f"merge()...Done")

def instance_gen_compare(opt):
    steps = opt.steps_arr[0]
    root_dir = opt.output_dir or f"./vrg_steps{steps:02d}_lp{opt.sch_lp_arr[0]:.4f}"
    log_info(f"root_dir: {root_dir}")
    os.makedirs(root_dir, exist_ok=True)
    file_old_trajectory = os.path.join(root_dir, f"ddim_steps{steps:02d}_trajectory_old.txt")
    file_new_trajectory = os.path.join(root_dir, f"ddim_steps{steps:02d}_trajectory_scheduled.txt")
    img_old_dir   = os.path.join(root_dir, f"img_by_old_trajectory_steps{steps:02d}")
    img_new_dir   = os.path.join(root_dir, f"img_by_new_trajectory_steps{steps:02d}")
    img_merge_dir = os.path.join(root_dir, f"img_merge_old_new_steps{steps:02d}")

    if opt.todo == 'merge': # only merge. no need to create model and sampler
        return merge(opt, img_old_dir, img_new_dir, img_merge_dir)

    model, sampler = get_model_and_sampler(opt)

    if 'track' in opt.todo:         # track current sampling trajectory.
        sampler.track_current_trajectory(steps=steps, file_path=file_old_trajectory)

    if 'schedule' in opt.todo:      # schedule current trajectory, and create new trajectory
        schedule(opt, model, file_old_trajectory, file_new_trajectory)

    if 'sample' in opt.todo or 'regen' in opt.todo:
        # sample with original trajectory
        # re-generate sample with scheduled trajectory
        old_dir, new_dir = None, None
        if 'sample' in opt.todo:
            old_dir = img_old_dir
            os.makedirs(old_dir, exist_ok=True)
        if 'regen' in opt.todo:
            new_dir = img_new_dir
            os.makedirs(new_dir, exist_ok=True)
        sample_or_regen(opt, model, sampler, old_dir, file_new_trajectory, new_dir)

    if 'merge' in opt.todo:         # merge images by old and new trajectories
        merge(opt, img_old_dir, img_new_dir, img_merge_dir)

def main():
    opt = parse_args()
    seed_everything(opt.seed)
    log_info(f"seed:{opt.seed}")
    log_info(f"opt:{opt}")
    log_info(f"txt2img_vrg -> todo: {opt.todo} ==================")
    if opt.todo == 'sample_compare_all':
        model, sampler = get_model_and_sampler(opt)
        sampler.sample_compare_all()
    else:
        instance_gen_compare(opt)


if __name__ == "__main__":
    main()
