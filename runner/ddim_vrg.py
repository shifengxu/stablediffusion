"""
SAMPLING ONLY, with VRG (Variance Reduction Guidance)
"""

import os
import torch
import numpy as np
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from utils import log_info


class DDIMSamplerVrg(object):
    def __init__(self, args, model):
        log_info(f"DDIMSamplerVrg::__init__()...")
        super().__init__()
        self.args = args
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.device = args.device or torch.device("cuda")
        log_info(f"  device     : {self.device}")
        log_info(f"  model      : {type(self.model).__name__}")
        log_info(f"  ddpm_num_timesteps : {self.ddpm_num_timesteps}")
        log_info(f"DDIMSamplerVrg::__init__()...Done")

    def sample_compare_all(self):
        """
        make samples, and compare (by FID) with 'fid_input1'
        loop all steps_arr, and sch_lp_arr
        """
        from utils import calc_fid
        import datetime

        args = self.args
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
        log_info(f"DDIMSamplerVrg::sample_compare_all()...")
        log_info(f"  output_dir : {args.output_dir}")
        log_info(f"  n_samples  : {args.n_samples}")
        log_info(f"  steps_arr  : {args.steps_arr}")
        log_info(f"  sch_lp_arr : {args.sch_lp_arr}")
        log_info(f"  fid_input1 : {args.fid_input1}")
        log_info(f"  gpu        : {gpu}")

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        img_gen_dir = os.path.join(output_dir, 'generated')
        os.makedirs(img_gen_dir, exist_ok=True)
        result_arr = [
            f"# time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"# host: {os.uname().nodename}",
            f"# cwd : {os.getcwd()}",
            f"# pid : {os.getpid()}",
            f"# n_samples  : {args.n_samples}",
            f"# fid_input1 : {args.fid_input1}",
            f"# img_gen_dir: {img_gen_dir}",
            f"# prompt     : {args.prompt}",
            f"# steps | vrg_scheduler_lp | FID",
        ]

        def save_result_arr():
            _path = os.path.join(output_dir, f"sample_compare_all.txt")
            with open(_path, 'w') as fptr:
                [fptr.write(f"{result}\n") for result in result_arr]

        # generate samples with old trajectory
        for steps in args.steps_arr:
            self.sample_by_vanilla(img_gen_dir)
            fid = calc_fid(gpu, True, args.fid_input1, img_gen_dir)
            result_arr.append(f"{steps:7d} \t vanilla \t {fid:7.3f}")
            save_result_arr()

            # track trajectory
            old_tj_file = os.path.join(output_dir, f"ddim_steps{steps:02d}_trajectory_old.txt")
            self.track_current_trajectory(steps, old_tj_file)
            for lp in args.sch_lp_arr:
                # schedule the current trajectory with lp, and make new trajectory
                new_tj_file = os.path.join(output_dir, f"ddim_steps{steps:02d}_trajectory_scheduled_lp{lp:.3f}.txt")
                self.schedule_trajectory_by_vrg(old_tj_file, new_tj_file)
                self.sample_by_trajectory(new_tj_file, img_gen_dir)
                fid = calc_fid(gpu, True, args.fid_input1, img_gen_dir)
                result_arr.append(f"{steps:7d} \t {lp:7.3f} \t {fid:7.3f}")
                save_result_arr()
            # for lp
        # for steps
        log_info(f"DDIMSamplerVrg::sample_compare_all()...Done")

    def sample_by_vanilla(self, img_gen_dir):
        """sample by vanilla approach, which uses original trajectory"""
        from torch import autocast
        from PIL import Image
        from einops import rearrange

        args = self.args
        log_info(f"DDIMSamplerVrg::sample_by_vanilla()...")
        log_info(f"  img_gen_dir : {img_gen_dir}")

        n_samples = args.n_samples
        batch_size = args.batch_size
        batch_cnt = n_samples // batch_size
        if batch_cnt * batch_size < n_samples: batch_cnt += 1

        latent_c, latent_h, latent_w = args.C, args.H // args.f, args.W // args.f
        shape = [args.C, latent_h, latent_w]
        log_info(f"  n_samples  : {n_samples}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  batch_cnt  : {batch_cnt}")
        log_info(f"  args.C     : {args.C}")
        log_info(f"  args.H     : {args.H}")
        log_info(f"  args.W     : {args.W}")
        log_info(f"  args.f     : {args.f}")
        log_info(f"  latent_h   : {latent_h}")
        log_info(f"  latent_w   : {latent_w}")
        log_info(f"  shape      : {shape}")


        def sample_by_vanilla_batch(_c, _uc, _noise, _init_idx):
            samples, _ = self.sample_batch(S=args.steps_arr[0],
                                           conditioning=_c,
                                           batch_size=len(_noise),
                                           shape=shape,
                                           verbose=False,
                                           unconditional_guidance_scale=args.scale,
                                           unconditional_conditioning=_uc,
                                           x_T=_noise)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            # if batch_size is 1, opt.f is 8, opt.H = opt.W = 512,
            # then shape:    [4,  64,  64]
            # sample    : [1, 4,  64,  64]
            # x_samples : [1, 3, 512, 512]

            path = None
            for x_sample in x_samples:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                path = os.path.join(img_gen_dir, f"{_init_idx:05}.png")
                img.save(path)
                _init_idx += 1
            log_info(f"  Saved {len(x_samples)}: {path}")

        with torch.no_grad(), autocast(args.device), self.model.ema_scope():
            for b_idx in range(0, batch_cnt):
                n = batch_size if b_idx < batch_cnt - 1 else n_samples - b_idx * batch_size
                prompts = n * [args.prompt]
                c = self.model.get_learned_conditioning(prompts)
                uc = None if args.scale == 1.0 else self.model.get_learned_conditioning(n * [""])
                start_code = torch.randn([n, latent_c, latent_h, latent_w], device=args.device)
                init_idx = b_idx * batch_size + args.s_batch_init_id
                sample_by_vanilla_batch(c, uc, start_code, init_idx)
            # for batch
        # with
        log_info(f"DDIMSamplerVrg::sample_by_vanilla()...Done")

    def sample_by_trajectory(self, trajectory_file, img_gen_dir):
        """"""
        from torch import autocast
        from PIL import Image
        from einops import rearrange

        def load_trajectory():
            with open(trajectory_file, 'r') as f:
                lines = f.readlines()
            _ab_arr, _ts_arr = [], []  # alpha_bar array, timestep array
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

        args = self.args
        log_info(f"DDIMSamplerVrg::sample_by_trajectory()...")
        log_info(f"  trajectory_file: {trajectory_file}")
        log_info(f"  img_gen_dir    : {img_gen_dir}")
        log_info(f"  prompt         : {args.prompt}")
        ab_arr, ts_arr = load_trajectory()  # alpha_bar array
        ts_arr_desc = list(reversed(ts_arr))
        step_count = len(ab_arr)
        log_info(f"  ab_arr len: {len(ab_arr)}")
        log_info(f"  ab_arr[0] : {ab_arr[0]:.6f}  {ts_arr[0]:3d}")
        log_info(f"  ab_arr[1] : {ab_arr[1]:.6f}  {ts_arr[1]:3d}")
        log_info(f"  ab_arr[-2]: {ab_arr[-2]:.6f}  {ts_arr[-2]:3d}")
        log_info(f"  ab_arr[-1]: {ab_arr[-1]:.6f}  {ts_arr[-1]:3d}")

        assert args.prompt is not None
        n_samples = args.n_samples
        batch_size = args.batch_size
        batch_cnt = n_samples // batch_size
        if batch_cnt * batch_size < n_samples: batch_cnt += 1

        latent_c, latent_h, latent_w = args.C, args.H // args.f, args.W // args.f
        log_info(f"  n_samples  : {n_samples}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  batch_cnt  : {batch_cnt}")
        log_info(f"  args.C     : {args.C}")
        log_info(f"  args.H     : {args.H}")
        log_info(f"  args.W     : {args.W}")
        log_info(f"  args.f     : {args.f}")
        log_info(f"  latent_c   : {latent_c}")
        log_info(f"  latent_h   : {latent_h}")
        log_info(f"  latent_w   : {latent_w}")

        def sample_batch(_ts, _c, _uc, _noise):
            samples, _ = self.sample2_batch(S=step_count,
                                            ts_list_desc=_ts,
                                            conditioning=_c,
                                            unconditional_conditioning=_uc,
                                            unconditional_guidance_scale=args.scale,
                                            x_T=_noise)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            # if batch_size is 1, args.f is 8, args.H = args.W = 512,
            # then shape:    [4,  64,  64]
            # sample    : [1, 4,  64,  64]
            # x_samples : [1, 3, 512, 512]
            return x_samples

        def save_batch(_x_samples, _init_idx):
            path = None
            for x in _x_samples:
                x = 255. * rearrange(x.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x.astype(np.uint8))
                path = os.path.join(img_gen_dir, f"{_init_idx:05}.png")
                img.save(path)
                _init_idx += 1
            log_info(f"  Saved {len(_x_samples)}: {path}")

        with torch.no_grad(), autocast(args.device), self.model.ema_scope():
            for b_idx in range(0, batch_cnt):
                n = batch_size if b_idx < batch_cnt - 1 else n_samples - b_idx * batch_size
                prompts = n * [args.prompt]
                c = self.model.get_learned_conditioning(prompts)
                uc = None if args.scale == 1.0 else self.model.get_learned_conditioning(n * [""])
                start_code = torch.randn([n, latent_c, latent_h, latent_w], device=args.device)
                init_idx = b_idx * batch_size + args.s_batch_init_id
                x_batch = sample_batch(ts_arr_desc, c, uc, start_code)
                save_batch(x_batch, init_idx)
            # for batch
        # with
        log_info(f"DDIMSamplerVrg::sample_by_trajectory()...Done")

    def schedule_trajectory_by_vrg(self, old_file, new_file):
        from scheduler_vrg.vrg_scheduler import VrgScheduler
        sch = VrgScheduler(self.args, self.model.alphas_cumprod)
        torch.set_grad_enabled(True)
        res = sch.schedule(f_path=old_file, output_file=new_file)
        torch.set_grad_enabled(False)
        return res

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_alphas_cumprod(self, ddim_num_steps, ddim_discretize="uniform", verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=0., verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = 0. * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def track_current_trajectory(self, steps, file_path):
        log_info(f"DDIMSamplerVrg::track_current_trajectory(steps={steps})...")
        log_info(f"  file_path: {file_path}")
        ddim_discr_method = "uniform"
        ts_arr = make_ddim_timesteps(ddim_discr_method=ddim_discr_method, num_ddim_timesteps=steps,
                                     num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=False)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        with open(file_path, 'w') as fptr:
            fptr.write(f"# steps   : {steps}\n")
            fptr.write(f"# class   : {self.__class__.__name__}\n")
            fptr.write(f"# ddpm_num_timesteps: {self.ddpm_num_timesteps}\n")
            fptr.write(f"# ddim_discr_method : {ddim_discr_method}\n")
            fptr.write(f"# alpha_bar\t: timestep\n")
            for i, ts in enumerate(ts_arr):
                ab = alphas_cumprod[ts]
                fptr.write(f"{ab:.8f}\t: {ts:3d}\n")
                log_info(f"  {i:02d}:  ab={ab:.8f}, ts={ts:3d}")
        # with
        log_info(f"  save file: {file_path}")
        log_info(f"DDIMSamplerVrg::track_current_trajectory(steps={steps})...Done")

    @torch.no_grad()
    def sample_batch(self,
                     S,
                     batch_size,
                     shape,
                     conditioning=None,
                     callback=None,
                     img_callback=None,
                     quantize_x0=False,
                     mask=None,
                     x0=None,
                     temperature=1.,
                     noise_dropout=0.,
                     score_corrector=None,
                     corrector_kwargs=None,
                     verbose=True,
                     x_T=None,
                     log_every_t=100,
                     unconditional_guidance_scale=1.,
                     unconditional_conditioning=None,
                     dynamic_threshold=None,
                     ucg_schedule=None,
                     ):
        self.make_alphas_cumprod(ddim_num_steps=S, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        if not hasattr(self, '_log_flag1_data_shape'):
            setattr(self, '_log_flag1_data_shape', True)
            print(f'Data shape for DDIM sampling is {size}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def sample2_batch(self,
                      S,
                      ts_list_desc,
                      conditioning=None,
                      img_callback=None,
                      quantize_x0=False,
                      mask=None,
                      x0=None,
                      temperature=1.,
                      noise_dropout=0.,
                      score_corrector=None,
                      corrector_kwargs=None,
                      x_T=None,
                      log_every_t=100,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      dynamic_threshold=None,
                      ucg_schedule=None,
                      ):
        # We have to use "S" (steps) here, and can not use len(ts_list_desc).
        # Because when steps=6, the ts_list_desc will have 7 elements:
        #   ts_list_desc = [997, 831, 665, 499, 333, 167, 1]
        # the reason is, when generating the timestep list, it is:
        #   c = num_ddpm_timesteps // num_ddim_timesteps
        #   ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        self.make_alphas_cumprod(ddim_num_steps=S, verbose=False)
        img = x_T
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # if steps is 20:
        # time_range: [951 901 851 801 751 701 651 601 551 501 451 401 351 301 251 201 151 101,  51   1]

        if hasattr(self, '_sample2_ts_log_done'):
            _sample2_ts_log_done = True
        else:
            _sample2_ts_log_done = False
            setattr(self, '_sample2_ts_log_done', True)
        b = len(x_T)
        total_steps = len(ts_list_desc)
        for i, timestep in enumerate(ts_list_desc):
            index = total_steps - i - 1
            ts = torch.full((b,), timestep, device=self.model.betas.device, dtype=torch.long)
            if not _sample2_ts_log_done: log_info(f"DDIMSamplerVrg::sample2(): ts[{i:02d}]: {timestep:3d}")

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == total_steps
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, conditioning, ts, index=index, use_original_steps=False,
                                      quantize_denoised=quantize_x0, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # if steps is 20, ddim_use_original_steps is False:
        # timesteps : [  1  51 101 151 201 251 301 351 401 451 501 551 601 651 701 751 801 851, 901 951]
        # time_range: [951 901 851 801 751 701 651 601 551 501 451 401 351 301 251 201 151 101,  51   1]

        if hasattr(self, '_ddim_sampling_ts_log_done'):
            _ddim_sampling_ts_log_done = True
        else:
            _ddim_sampling_ts_log_done = False
            setattr(self, '_ddim_sampling_ts_log_done', True)
        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            # print(f"index:{index}; step:{step}")
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if not _ddim_sampling_ts_log_done: log_info(f"DDIMSamplerVrg::ddim_sampling(): ts[{i:02d}]: {step:3d}")

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
