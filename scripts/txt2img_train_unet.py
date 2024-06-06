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

import time
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from utils import log_info, get_time_ttl_and_eta

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=str, default='latent_gen')
    # parser.add_argument("--todo", type=str, default='unet_train')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument("--data_dir", type=str, default="./download_dataset/coco_val2017")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
        default="./checkpoints/v2-1_512-ema-pruned.ckpt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    args.device = device
    log_info(f"gpu_ids : {gpu_ids}")
    log_info(f"device  : {device}")

    return args

def load_model_from_config(config, ckpt):
    log_info(f"load_model_from_config()...")
    log_info(f"  ckpt  : {ckpt}")
    # config.model has target: ldm.models.diffusion.ddpm.LatentDiffusion
    log_info(f"  create model...")
    model = instantiate_from_config(config.model)

    log_info(f"  torch.load({ckpt})...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        log_info(f"  Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    log_info(f"  model.load_state_dict()...")
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0:
        log_info("missing keys:")
        log_info(m)
    if len(u) > 0:
        log_info("unexpected keys:")
        log_info(u)
    log_info(f"  model.eval()")
    model.eval()
    model.to(opt.device)
    log_info(f"load_model_from_config()...Done")
    return model

def load_model_unet_from_config(config, ckpt):
    log_info(f"load_model_unet_from_config()...")
    log_info(f"  ckpt  : {ckpt}")
    # config.model has target: ldm.models.diffusion.ddpm.LatentDiffusion
    log_info(f"  create model...")
    model = instantiate_from_config(config.model)

    log_info(f"  torch.load({ckpt})...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        log_info(f"  Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    log_info(f"  model.load_state_dict()...")
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0:
        log_info("missing keys:")
        log_info(m)
    if len(u) > 0:
        log_info("unexpected keys:")
        log_info(u)
    log_info(f"  model.eval()")
    model.eval()
    model.to(opt.device)

    del sd  # clear memory
    # In this context,
    # model                       is LatentDiffusion
    # model.model                 is DiffusionWrapper
    # model.model.diffusion_model is UNet, ldm.modules.diffusionmodules.openaimodel.UNetModel
    u_cfg = config.model.params.unet_config
    log_info(f"  new_unet setup ... target: {u_cfg['target']}")
    new_unet = instantiate_from_config(u_cfg)   # train this UNet
    model.model.diffusion_model = new_unet
    log_info(f"  new_unet setup... done")
    log_info(f"load_model_unet_from_config()...Done")

    return model, new_unet

def latent_gen(config):
    """"""
    log_info(f"latent_gen()...")
    data_dir = opt.data_dir
    # image_size = config.model.params.first_stage_config.params.ddconfig.resolution
    image_size = 512
    folder, basename = os.path.split(data_dir)
    dir_ltt = os.path.join(folder, f"{basename}_ltt")
    dir_dec = os.path.join(folder, f"{basename}_ltt_decode")
    dir_cmp = os.path.join(folder, f"{basename}_ori_decode_cmp")
    os.makedirs(dir_ltt, exist_ok=True)
    os.makedirs(dir_dec, exist_ok=True)
    os.makedirs(dir_cmp, exist_ok=True)
    log_info(f"image_size : {image_size}")
    log_info(f"data_dir   : {data_dir}")
    log_info(f"dir_ltt    : {dir_ltt}")
    log_info(f"dir_dec    : {dir_dec}")
    log_info(f"dir_cmp    : {dir_cmp}")

    from datasets import ImageDataset
    import torchvision.transforms as T
    import torch.utils.data as tu_data
    import torchvision.utils as tvu

    tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    ds = ImageDataset(data_dir, transform=tf)
    img_cnt = len(ds)
    dl = tu_data.DataLoader(ds, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    batch_cnt = len(dl)
    model = load_model_from_config(config, f"{opt.ckpt}")
    # model is LatentDiffusion

    def _save():
        f_path = None
        for img, img_id, ltt, img_dec in zip(img_batch, id_batch, ltt_batch, img_decode):
            stem = f"{img_id:012d}"
            f_path = os.path.join(dir_ltt, f"{stem}.npy")
            np.save(f_path, ltt.cpu().numpy())
            f_path = os.path.join(dir_dec, f"{stem}.png")
            tvu.save_image(img_dec, f_path)
            f_path = os.path.join(dir_cmp, f"{stem}.png")
            img2 = torch.cat([img, img_dec], dim=2)
            tvu.save_image(img2, f_path)
            # img2 = 255. * rearrange(img2.cpu().numpy(), 'c h w -> h w c')
            # img2 = Image.fromarray(img2.astype(np.uint8))
            # img2.save(f_path)
        # for save items
        return f_path

    log_info(f"img_cnt    : {img_cnt}")
    log_info(f"batch_size : {opt.batch_size}")
    log_info(f"batch_cnt  : {batch_cnt}")
    start_time = time.time()
    for b_idx, (img_batch, id_batch) in enumerate(dl):
        img_batch = img_batch.to(opt.device)
        img_batch = img_batch * 2.0 - 1.0
        encoder_posterior = model.encode_first_stage(img_batch)
        ltt_batch = model.get_first_stage_encoding(encoder_posterior)
        img_decode = model.decode_first_stage(ltt_batch)
        if b_idx == 0:
            log_info(f"img_batch : {img_batch.size()}")
            log_info(f"ltt_batch : {ltt_batch.size()}")
            log_info(f"img_decode: {img_decode.size()}")
        img_decode = torch.clamp((img_decode + 1.0) / 2.0, min=0.0, max=1.0)
        img_batch  = torch.clamp((img_batch + 1.0) / 2.0, min=0.0, max=1.0)
        f_path = _save()
        elp, eta = get_time_ttl_and_eta(start_time, b_idx+1, batch_cnt)
        log_info(f"B{b_idx:03d}, elp:{elp}, eta:{eta}. saved {len(img_batch)}: {f_path}")
    # for b_idx

def unet_train(config):
    device = opt.device
    model, unet = load_model_unet_from_config(config, f"{opt.ckpt}")
    # todo: unet
    sampler = DDIMSampler(model, device=device)
    log_info(f"sampler = DDIMSampler(model, device={device})")

    out_dir = opt.outdir
    os.makedirs(out_dir, exist_ok=True)
    batch_size = opt.n_samples
    log_info(f"out_dir    : {out_dir}")
    log_info(f"batch_size : {batch_size}")
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        prompt_batch = [batch_size * [prompt]]

    else:
        log_info(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            prompt_batch = f.read().splitlines()
            prompt_batch = [p for p in prompt_batch for _ in range(opt.repeat)]
            it1 = iter(prompt_batch)
            it2 = iter(lambda: tuple(islice(it1, batch_size)), ())
            prompt_batch = list(it2)

    start_code = None
    log_info(f"start_code : {start_code}")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), precision_scope(opt.device), model.ema_scope():
        all_samples = list()
        for n in range(opt.n_iter):      # Sampling
            for prompts in prompt_batch:
                x_samples = sample_by_prompts(sampler, start_code, prompts, batch_size, out_dir)
                all_samples.append(x_samples)
            # for prompt
        # for n_iter
    # with

    log_info(f"Your samples are ready and waiting for you here: \n{out_dir} \n \nEnjoy.")

def sample_by_prompts(sampler, start_code, prompts, batch_size, sample_path):
    global img_base_count
    model = sampler.model
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(batch_size * [""])
    else:
        uc = None
    if isinstance(prompts, tuple):
        prompts = list(prompts)
    c = model.get_learned_conditioning(prompts)
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    samples, _ = sampler.sample(S=opt.steps,
                                conditioning=c,
                                batch_size=opt.n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc,
                                eta=0.0,
                                x_T=start_code)

    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    # if batch_size is 1, opt.f is 8, opt.H = opt.W = 512,
    # then shape:    [4,  64,  64]
    # sample    : [1, 4,  64,  64]
    # x_samples : [1, 3, 512, 512]

    img_path = None
    for x_sample in x_samples:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img_path = os.path.join(sample_path, f"{img_base_count:05}.png")
        img.save(img_path)
        img_base_count += 1
    log_info(f"Saved {len(x_samples)} images: {img_path}")
    return x_samples

def main():
    seed_everything(opt.seed)
    log_info(f"opt    : -------------------------")
    log_info(f"seed   : {opt.seed}")
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
    log_info(f"config : {cfg_file}")
    log_info(f"config : {config}")

    if opt.todo == 'latent_gen':
        latent_gen(config)
    elif opt.todo == 'unet_train':
        unet_train(config)
    else:
        raise ValueError(f"Invalid todo: {opt.todo}")

if __name__ == "__main__":
    img_base_count = 0
    opt = parse_args()
    main()
