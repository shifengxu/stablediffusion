import json
import os
import time
import torch
import numpy as np
import torchvision.transforms as T
import torch.utils.data as tu_data
import torchvision.utils as tvu

from ldm.util import instantiate_from_config
from utils import log_info, get_time_ttl_and_eta
from datasets import ImageDataset


class DDIMLatentGenerator:
    def __init__(self, opt, config):
        log_info(f"DDIMLatentGenerator::__init__()...")
        self.opt = opt
        self.config = config
        self.device   = opt.device
        self.data_dir = opt.data_dir
        self.model = None # model is LatentDiffusion
        parent_folder, data_folder = os.path.split(self.data_dir)
        self.parent_folder, self.data_folder = parent_folder, data_folder
        log_info(f"data_dir : {self.data_dir}")
        log_info(f"DDIMLatentGenerator::__init__()...Done")

    def load_model_from_config(self):
        opt, config = self.opt, self.config
        ckpt = opt.ckpt
        log_info(f"DDIMLatentGenerator::load_model_from_config()...")
        log_info(f"  ckpt  : {ckpt}")
        # config.model has target: ldm.models.diffusion.ddpm.LatentDiffusion
        log_info(f"  create model...")
        model = instantiate_from_config(config.model)

        log_info(f"  torch.load({ckpt})...")
        tl_sd = torch.load(ckpt, map_location=opt.device)  # torch loaded state dict
        if "global_step" in tl_sd:
            log_info(f"  Global Step: {tl_sd['global_step']}")
        sd = tl_sd["state_dict"]
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
        log_info(f"  model.to({opt.device})")
        model = model.to(opt.device)
        log_info(f"DDIMLatentGenerator::load_model_from_config()...Done")
        return model

    def encode_prompt(self):
        log_info(f"DDIMLatentGenerator::encode_prompt()...")
        parent_folder, data_folder = self.parent_folder, self.data_folder

        json_file_name = f"{data_folder}_captions.json"  # coco_val2017_captions.json
        # the JSON file name is tricky. But now let's do it in this way.
        json_file_path = os.path.join(parent_folder, json_file_name)
        if not os.path.exists(json_file_path):
            raise ValueError(f"File not exist: {json_file_path}")
        log_info(f"Read : {json_file_path}")
        with open(json_file_path, 'r') as fptr:
            j_str = fptr.read()
        log_info(f"Parse: {json_file_path}")
        j_dict = json.loads(j_str)
        anno_arr = j_dict.get("annotations", [])
        anno_len = len(anno_arr)
        log_info(f"Found: {anno_len} annotations")

        dir_pen = os.path.join(parent_folder, f"{data_folder}_prompt_encode")
        os.makedirs(dir_pen, exist_ok=True)
        log_info(f"dir_pen    : {dir_pen}")
        b_sz = self.opt.batch_size
        prompt_list = []
        if self.model is None:
            self.model = self.load_model_from_config()
        model = self.model
        for i in range(0, anno_len, b_sz):
            j = i + b_sz
            if j > anno_len:
                j = anno_len
            anno_batch = anno_arr[i:j]
            # some prompt has trailing "\n", so strip() is necessary.
            prompt_batch = [a['caption'].strip() for a in anno_batch]
            ppt_enc_batch = model.get_learned_conditioning(prompt_batch, device=self.device)
            f_path = None
            for ppt_enc, anno, caption in zip(ppt_enc_batch, anno_batch, prompt_batch):
                img_id, ppt_id = anno['image_id'], anno['id']
                f_path = os.path.join(dir_pen, f"{img_id:012d}_{ppt_id:06d}.npy")
                np.save(f_path, ppt_enc.cpu().numpy())
                prompt_list.append(f"{img_id:012d}_{ppt_id:06d}\t{caption}")
            # for
            log_info(f"Save prompt encoding. {i:05d}~{j:05d} / {anno_len}. {f_path}")
        # for
        f_path = os.path.join(parent_folder, f"{data_folder}_prompt_encode.txt")
        with open(f_path, 'w') as fptr:
            [fptr.write(f"{p}\n") for p in prompt_list]
        # with
        log_info(f"Saved: {f_path}")

    @staticmethod
    def _save(img_batch, id_batch, ltt_batch, img_decode, dir_ltt, dir_dec, dir_cmp):
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

    def encode_image_to_latent(self, image_size=512):
        opt, config = self.opt, self.config
        log_info(f"DDIMLatentGenerator::encode_image_to_latent()...")
        # image_size = config.model.params.first_stage_config.params.ddconfig.resolution
        data_dir = self.data_dir
        log_info(f"image_size : {image_size}")
        log_info(f"data_dir   : {data_dir}")

        tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        ds = ImageDataset(data_dir, transform=tf)
        img_cnt = len(ds)
        dl = tu_data.DataLoader(ds, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        batch_cnt = len(dl)

        parent_folder, data_folder = self.parent_folder, self.data_folder
        dir_ltt = os.path.join(parent_folder, f"{data_folder}_ltt")
        dir_dec = os.path.join(parent_folder, f"{data_folder}_ltt_decode")
        dir_cmp = os.path.join(parent_folder, f"{data_folder}_ori_decode_cmp")
        os.makedirs(dir_ltt, exist_ok=True)
        os.makedirs(dir_dec, exist_ok=True)
        os.makedirs(dir_cmp, exist_ok=True)
        log_info(f"dir_ltt    : {dir_ltt}")
        log_info(f"dir_dec    : {dir_dec}")
        log_info(f"dir_cmp    : {dir_cmp}")
        log_info(f"img_cnt    : {img_cnt}")
        log_info(f"batch_size : {opt.batch_size}")
        log_info(f"batch_cnt  : {batch_cnt}")
        start_time = time.time()
        if self.model is None:
            self.model = self.load_model_from_config()
        model = self.model
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
            img_batch = torch.clamp((img_batch + 1.0) / 2.0, min=0.0, max=1.0)
            f_path = self._save(img_batch, id_batch, ltt_batch, img_decode, dir_ltt, dir_dec, dir_cmp)
            elp, eta = get_time_ttl_and_eta(start_time, b_idx + 1, batch_cnt)
            log_info(f"B{b_idx:03d}, elp:{elp}, eta:{eta}. saved {len(img_batch)}: {f_path}")
        # for b_idx

# class
