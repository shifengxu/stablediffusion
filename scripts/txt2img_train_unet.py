import os
import sys
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
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from utils import log_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=str, default='latent_gen')
    # parser.add_argument("--todo", type=str, default='unet_train')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./download_dataset/coco_val2017_ltt")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument("--outdir", type=str, nargs="?", default="outputs/txt2img-samples")
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="downsampling factor, most often 8 or 16")
    parser.add_argument("--n_samples", type=int, default=1, help=" samples to produce for each prompt")
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
        default="./checkpoints/v2-1_512-ema-pruned.ckpt",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    # device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    args.device = device
    log_info(f"gpu_ids : {gpu_ids}")
    log_info(f"device  : {device}")

    return args

def main():
    opt = parse_args()
    if opt.seed:
        log_info(f"seed_everything({opt.seed})")
        seed_everything(opt.seed)
    log_info(f"cwd    : {os.getcwd()}")
    log_info(f"pid    : {os.getpid()}")
    log_info(f"host   : {os.uname().nodename}")
    log_info(f"opt    : -------------------------")
    log_info(f"seed   : {opt.seed}")
    if opt.seed:
        log_info(f"         seed_everything({opt.seed})")
        seed_everything(opt.seed)
    log_info(f"todo   : {opt.todo}")
    log_info(f"gpu_ids: {opt.gpu_ids}")
    log_info(f"device : {opt.device}")
    log_info(f"opt    : {opt}")

    # hard code config file temporarily
    abs_path = os.path.abspath(__file__)
    s_dir = os.path.dirname(abs_path)   # the scripts folder
    r_dir = os.path.dirname(s_dir)      # root folder
    cfg_file = os.path.join(r_dir, "configs/stable-diffusion/v2-inference.yaml")
    log_info(f"Loading: {cfg_file}")
    config = OmegaConf.load(cfg_file)
    log_info(f"config : -------------------------")
    log_info(f"f_path : {cfg_file}")
    log_info(f"config : {config}")
    log_info(f"config : `````````````````````````")

    if opt.todo == 'latent_gen':
        from runner.ddim_latent_gen import DDIMLatentGenerator
        runner = DDIMLatentGenerator(opt, config)
        runner.encode_prompt()
        runner.encode_image_to_latent()
    elif opt.todo == 'unet_train':
        from runner.ddim_trainer import DDIMTrainer
        runner = DDIMTrainer(opt, config)
        runner.run()
    else:
        raise ValueError(f"Invalid todo: {opt.todo}")

if __name__ == "__main__":
    main()
