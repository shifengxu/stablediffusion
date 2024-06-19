import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import log_info


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.exists(root_dir):
            raise ValueError(f"Path not exist: {root_dir}")
        if not os.path.isdir(root_dir):
            raise ValueError(f"Path not dir: {root_dir}")
        name_arr = os.listdir(root_dir)
        name_arr.sort()
        self.img_path_arr = [os.path.join(root_dir, n) for n in name_arr]

    def __getitem__(self, index):
        img_path = self.img_path_arr[index]
        basename = os.path.basename(img_path)
        stem, ext = os.path.splitext(basename)
        img_id = int(stem)
        image = Image.open(img_path)
        img_rgb = image.convert("RGB")
        if self.transform:
            img_rgb = self.transform(img_rgb)
        img_np = np.array(img_rgb)
        return img_np, img_id

    def __len__(self):
        return len(self.img_path_arr)

class LatentPromptDataset(Dataset):
    def __init__(self, data_dir, data_limit=0):
        # data_dir is like: ./download_dataset/coco_val2017_ltt
        self.data_dir = data_dir
        self.data_limit = data_limit
        if not os.path.exists(data_dir):
            raise ValueError(f"Path not exist: {data_dir}")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Path not dir: {data_dir}")
        fn_list = os.listdir(data_dir)  # file name list: ['00001.npy', '00002.npy', '00003.npy']
        fns_list = [f[:-4] for f in fn_list if f.endswith('.npy')]
        # file name stem list: ['00001', '00002', '00003']
        fns_list.sort()
        log_info(f"LatentPromptDataset()...")
        log_info(f"  data_dir   : {data_dir}")
        log_info(f"  file cnt   : {len(fns_list)}")
        log_info(f"  file[0]    : {fns_list[0]}")
        log_info(f"  file[-1]   : {fns_list[-1]}")
        log_info(f"  data_limit : {data_limit}")
        if data_limit > 0:
            fns_list = fns_list[:data_limit]
        log_info(f"  train      : {len(fns_list)}")
        log_info(f"  train[0]   : {fns_list[0]}")
        log_info(f"  train[-1]  : {fns_list[-1]}")
        self.fns_list = fns_list

        # data dir: ./download_dataset/coco_val2017_ltt
        # pen dir : ./download_dataset/coco_val2017_prompt_encode
        parent_folder, dir_name = os.path.split(data_dir)
        dir_name = dir_name[:-4]
        pen_dir_name = f"{dir_name}_prompt_encode"
        pen_path = os.path.join(parent_folder, pen_dir_name)    # ./download_dataset/coco_val2017_prompt_encode
        self.pen_dir = pen_path
        log_info(f"  pen_path   : {self.pen_dir}")
        pen_list = os.listdir(pen_path) # prompt encoding file list
        pen_list.sort()
        log_info(f"  found prompt: {len(pen_list)}")

        fns_pen_list = []   # it will have the same length with fns_list
        len_fns, len_pen = len(fns_list), len(pen_list)
        i, j = 0, 0
        cur_ltt_pen_list = []   # current latent pen list
        no_prompt_count = 0
        while i < len_fns and j < len_pen:
            name1 = fns_list[i]
            name2 = pen_list[j][:12]
            # pen file name is like "000000581781_228278.npy". Its first 12 character is img id
            if name1 < name2:
                fns_pen_list.append(cur_ltt_pen_list)
                if len(cur_ltt_pen_list) == 0: no_prompt_count += 1
                cur_ltt_pen_list = []
                i += 1
            if name1 > name2:
                j += 1
                log_info(f"!!! this case should not happen: name1:{name1}, name2:{name2}")
            else:
                cur_ltt_pen_list.append(pen_list[j])
                j += 1
        # while
        while i < len_fns:
            fns_pen_list.append(cur_ltt_pen_list)
            if len(cur_ltt_pen_list) == 0: no_prompt_count += 1
            cur_ltt_pen_list = []
            i += 1
        log_info(f"  fns_pen_list: {len(fns_pen_list)}")
        log_info(f"  no_pen_count: {no_prompt_count}")
        sum_up = 0
        for pen_lst in fns_pen_list:
            sum_up += len(pen_lst)
        assert sum_up == len_pen, f"sum of fns_pen_list is {sum_up}, but original pen is {len_pen}"
        self.fns_pen_list = fns_pen_list
        lst2str = lambda lst: " ".join([f"{l}" for l in lst])
        log_info(f"  fns_pen_list sum: {sum_up}")
        log_info(f"  fns_list[0]     : {fns_list[0]}")
        log_info(f"  fns_pen_list[0] : {lst2str(fns_pen_list[0])}")
        log_info(f"  fns_list[-1]    : {fns_list[-1]}")
        log_info(f"  fns_pen_list[-1]: {lst2str(fns_pen_list[-1])}")
        log_info(f"LatentPromptDataset()...Done")

    def __getitem__(self, index):
        fns = self.fns_list[index] # file name step is like: '00001', '00002', '00003'
        ltt_id = int(fns)
        ltt = np.load(os.path.join(self.data_dir, f"{fns}.npy"))
        pen_list = self.fns_pen_list[index] # prompt encoding list
        pen_len = len(pen_list)
        pen_idx = random.randint(0, pen_len - 1)    # randint() include both start and end
        pen_fn = pen_list[pen_idx]  # pen file name, is like "000000581781_228278.npy"
        pen = np.load(os.path.join(self.pen_dir, pen_fn))
        stem, _ = pen_fn.split('.')
        img_id, prompt_id = stem.split('_')
        prompt_id = int(prompt_id)
        return ltt, pen, ltt_id, prompt_id

    def __len__(self):
        return len(self.fns_list)

class LatentPrompt1to1Dataset(Dataset):
    def __init__(self, data_dir, data_limit=0):
        # data_dir is like: ./download_dataset/coco_val2017_ltt
        self.data_dir = data_dir
        self.data_limit = data_limit
        if not os.path.exists(data_dir):
            raise ValueError(f"Path not exist: {data_dir}")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Path not dir: {data_dir}")
        fn_list = os.listdir(data_dir)  # file name list: ['00001.npy', '00002.npy', '00003.npy']
        fns_list = [f[:-4] for f in fn_list if f.endswith('.npy')]
        # file name stem list: ['00001', '00002', '00003']
        fns_list.sort()
        log_info(f"LatentPrompt1to1Dataset()...")
        log_info(f"  data_dir   : {data_dir}")
        log_info(f"  file cnt   : {len(fns_list)}")
        log_info(f"  file[0]    : {fns_list[0]}")
        log_info(f"  file[-1]   : {fns_list[-1]}")
        log_info(f"  data_limit : {data_limit}")
        if data_limit > 0:
            fns_list = fns_list[:data_limit]
        log_info(f"  train      : {len(fns_list)}")
        log_info(f"  train[0]   : {fns_list[0]}")
        log_info(f"  train[-1]  : {fns_list[-1]}")
        self.fns_list = fns_list

        # data dir: ./download_dataset/coco_val2017_ltt
        # pen dir : ./download_dataset/coco_val2017_prompt_encode
        parent_folder, dir_name = os.path.split(data_dir)
        dir_name = dir_name[:-4]
        pen_dir_name = f"{dir_name}_prompt_encode"
        pen_path = os.path.join(parent_folder, pen_dir_name)    # ./download_dataset/coco_val2017_prompt_encode
        self.pen_dir = pen_path
        log_info(f"  pen_path   : {self.pen_dir}")
        if not os.path.exists(pen_path):
            raise ValueError(f"prompt encoding dir not found: {pen_path}")
        pen_list = os.listdir(pen_path) # prompt encoding file list
        pen_list.sort()
        log_info(f"  found prompt: {len(pen_list)}")

        ltt_pen_map = {}
        for pen_fn in pen_list:
            # pen file name is like "000000581781_228278.npy". Its first 12 character is img id
            tmp, _ = pen_fn.split('.')
            ltt, pen = tmp.split('_')
            if ltt not in ltt_pen_map:
                ltt_pen_map[ltt] = pen_fn
        # for
        self.ltt_pen_map = ltt_pen_map
        log_info(f"  ltt_pen_map : {len(ltt_pen_map)}")
        assert len(self.fns_list) == len(self.ltt_pen_map), f"prompt files should match all latent files."
        for fns in self.fns_list:
            assert fns in self.ltt_pen_map, f"{fns} should in ltt_pen_map"
        f_path = "./latent_prompt_in_training.txt"
        with open(f_path, 'w') as fptr:
            fptr.write(f"# From: LatentPrompt1to1Dataset\n")
            fptr.write(f"# size: {len(self.ltt_pen_map)} \n")
            fptr.write(F"# host: {os.uname().nodename}")
            fptr.write(F"# cwd : {os.getcwd()}")
            fptr.write(F"# pid : {os.getpid()}")
            fptr.write(f"# latent         : {self.data_dir} \n")
            fptr.write(f"# prompt encoding: {self.pen_dir} \n")
            fptr.write(f"\n")
            fptr.write(f"# latent    : prompt encoding \n")
            [fptr.write(f"{ltt}: {pen}\n") for ltt, pen in self.ltt_pen_map.items()]
        log_info(f"  Saved: {f_path}")
        log_info(f"LatentPrompt1to1Dataset()...Done")

    def __getitem__(self, index):
        fns = self.fns_list[index] # file name step is like: '00001', '00002', '00003'
        ltt_id = int(fns)
        ltt = np.load(os.path.join(self.data_dir, f"{fns}.npy"))
        pen_fn = self.ltt_pen_map[fns]
        # pen file name is like "000000581781_228278.npy". Its first 12 character is img id
        pen = np.load(os.path.join(self.pen_dir, pen_fn))
        stem, _ = pen_fn.split('.')
        img_id, prompt_id = stem.split('_')
        prompt_id = int(prompt_id)
        return ltt, pen, ltt_id, prompt_id

    def __len__(self):
        return len(self.fns_list)
