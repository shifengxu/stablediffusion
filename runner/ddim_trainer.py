import os
import time
import torch
import torch.utils.data as tu_data
from torch import autocast, optim

from ldm.util import instantiate_from_config
from runner.ema import ExponentialMovingAverage
from utils import log_info, get_time_ttl_and_eta
from datasets import LatentPromptDataset, LatentPrompt1to1Dataset


class DDIMTrainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = args.device
        self.lr = args.lr
        self.ema_rate = args.ema_rate
        self.shape = [args.C, args.H // args.f, args.W // args.f]
        self.uc_guidance_scale = args.scale
        log_info(f"DDIMTrainer::__init__()...")
        log_info(f"  device   : {self.device}")
        log_info(f"  lr       : {self.lr}")
        log_info(f"  ema_rate : {self.ema_rate}")
        log_info(f"  C        : {args.C}")
        log_info(f"  H        : {args.H}")
        log_info(f"  W        : {args.W}")
        log_info(f"  f        : {args.f}")
        log_info(f"  shape    : {self.shape}  <-- latent shape")
        model, unet = self.load_model_unet_from_config(config, args.ckpt)
        self.unet = unet
        self.optimizer = self.get_optimizer(unet.parameters())
        self.ema = None # don't set up EMA until training
        self.empty_prompt_encoding = model.get_learned_conditioning([""], device=self.device)
        # empty_prompt_encoding has size[1, 77, 1024]
        self.ddpm_num_timesteps = model.num_timesteps
        self.ab_list = model.alphas_cumprod # alpha_bar list
        self.noise_select_opt1 = True
        log_info(f"  uc_guidance_scale    : {self.uc_guidance_scale}")
        log_info(f"  empty_prompt_encoding: {self.empty_prompt_encoding.size()}")
        log_info(f"  ddpm_num_timesteps   : {self.ddpm_num_timesteps}")
        log_info(f"  ab_list length       : {len(self.ab_list)}")
        log_info(f"  ab_list[0]           : {self.ab_list[0]:.8f}")
        log_info(f"  ab_list[1]           : {self.ab_list[1]:.8f}")
        log_info(f"  ab_list[-2]          : {self.ab_list[-2]:.8f}")
        log_info(f"  ab_list[-1]          : {self.ab_list[-1]:.8f}")
        log_info(f"  noise_select_opt1    : {self.noise_select_opt1}")
        self.batch_counter = None
        self.batch_total = None
        self.eval_start_code_arr = []
        self.dataset = None

        # to save memory, we can delete the model (which is LatentDiffusion)
        # and it can save 1.7G memory in each GPU.
        # When batch size is 10, the memory comparison is as below:
        #   Tesla V100-PCIE-32GB | 64°C, 100 % | 28786 / 32768 MB | shifeng(28778M)     <-- del model
        #   Tesla V100-PCIE-32GB | 55°C, 100 % | 30450 / 32768 MB | shifeng(30442M)     <-- use model
        # If delete model, the improvement is marginal. But it introduces some risk,
        # as it may miss some intermediate operations.
        # Here, intermediate operations mean the logic between:
        #   LatentDiffusion -> DiffusionWrapper -> UNetModel
        # del model

        # By now, we use model (LatentDiffusion).
        self.model = model
        log_info(f"  self.model = model")
        log_info(f"DDIMTrainer::__init__()...Done")

    def load_model_unet_from_config(self, config, ckpt):
        log_info(f"load_model_unet_from_config()...")
        log_info(f"  ckpt  : {ckpt}")
        # config.model has target: ldm.models.diffusion.ddpm.LatentDiffusion
        log_info(f"  instantiate_from_config(config.model)")
        model = instantiate_from_config(config.model)

        log_info(f"  torch.load({ckpt})...")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            log_info(f"  pl_sd['global_step']: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        log_info(f"  model.load_state_dict()...")
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            log_info(f"  missing keys:")
            log_info(f"    {m}")
        if len(u) > 0:
            log_info(f"  unexpected keys:")
            log_info(f"    {u}")

        del sd  # clear memory
        u_cfg = config.model.params.unet_config
        log_info(f"  new_unet setup ... target: {u_cfg['target']}")
        # train this UNet
        new_unet = instantiate_from_config(u_cfg)
        # from ldm.modules.diffusionmodules.openaimodel import UNetModel
        # new_unet = UNetModel(
        #     use_checkpoint=True,
        #     use_fp16=True,
        #     image_size=32,
        #     in_channels=4,
        #     out_channels=4,
        #     model_channels=320,
        #     attention_resolutions=[4, 2, 1],
        #     num_res_blocks=2,
        #     channel_mult=[1, 2, 4, 4],
        #     num_head_channels=64,
        #     use_spatial_transformer=True,
        #     use_linear_in_transformer=True,
        #     transformer_depth=1,
        #     context_dim=1024,
        #     legacy=False
        # )
        new_unet.eval()
        model.model.diffusion_model = new_unet
        # In this context,
        # model                       : is LatentDiffusion
        # model.model                 : is DiffusionWrapper
        # model.model.diffusion_model : is ldm.modules.diffusionmodules.openaimodel.UNetModel
        log_info(f"  new_unet setup... done")
        log_info(f"  model.eval()")
        model.eval()
        log_info(f"  model.to({self.device})")
        model.to(self.device)
        log_info(f"load_model_unet_from_config()...Done")

        return model, new_unet

    def get_optimizer(self, parameters):
        lr = self.lr
        wd = 0.
        betas = (0.9, 0.999)
        eps = 0.00000001
        o = optim.Adam(parameters, lr=lr, weight_decay=wd, betas=betas, eps=eps, amsgrad=False)
        log_info(f"DDIMTrainer::get_optimizer()...")
        log_info(f"  optimizer   : {type(o).__name__}")
        log_info(f"  lr          : {lr}")
        log_info(f"  weight_decay: {wd}")
        log_info(f"  betas       : {betas}")
        log_info(f"  eps         : {eps}")
        log_info(f"DDIMTrainer::get_optimizer()...Done")
        return o

    def log_stats(self, latent, prompt_encoding, un_cond):
        l_var, l_mean = torch.var_mean(latent)
        p_var, p_mean = torch.var_mean(prompt_encoding)
        u_var, u_mean = torch.var_mean(un_cond)
        log_info(f"latent    : {latent.size()}")
        log_info(f"prompt_enc: {prompt_encoding.size()}")
        log_info(f"un_cond   : {un_cond.size()}")
        log_info(f"latent    : {latent.dtype}")
        log_info(f"prompt_enc: {prompt_encoding.dtype}")
        log_info(f"un_cond   : {un_cond.dtype}")
        log_info(f"latent    : mean:{l_mean:7.4f}, var:{l_var:.4f}")
        log_info(f"prompt_enc: mean:{p_mean:7.4f}, var:{p_var:.4f}")
        log_info(f"un_cond   : mean:{u_mean:7.4f}, var:{u_var:.4f}")

        C, H, W = self.shape
        bs, c, h, w = latent.size()
        assert C == c, f"DDIMTrainer.C is {C}, not match input latent {c}"
        assert H == h, f"DDIMTrainer.H is {H}, not match input latent {h}"
        assert W == w, f"DDIMTrainer.W is {W}, not match input latent {w}"

    def train(self):
        args, config, = self.args, self.config
        if args.prompt_per_latent == 'n':
            ds = LatentPromptDataset(args.data_dir)
        elif args.prompt_per_latent == '1':
            ds = LatentPrompt1to1Dataset(args.data_dir)
        else:
            raise ValueError(f"Invalid args.prompt_per_latent: {args.prompt_per_latent}")
        self.dataset = ds
        ltt_cnt = len(ds)
        batch_size = args.batch_size
        dl = tu_data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
        batch_cnt = len(dl)
        e_cnt = args.n_epochs
        self.batch_counter, self.batch_total = 0, batch_cnt * e_cnt

        log_itv = args.log_interval
        save_itv = args.save_ckpt_interval
        save_path = args.save_ckpt_path
        save_eval = args.save_ckpt_eval
        log_info(f"log_itv    : {log_itv}")
        log_info(f"save_itv   : {save_itv}")
        log_info(f"save_path  : {save_path}")
        log_info(f"save_eval  : {save_eval}")
        log_info(f"batch_size : {batch_size}")
        log_info(f"e_cnt      : {e_cnt}")
        log_info(f"batch_cnt  : {batch_cnt}")
        log_info(f"batch_total: {self.batch_total}")
        log_info(f"ltt_cnt    : {ltt_cnt}")
        log_info(f"noise_sss  : {args.noise_sss}")
        prompt = args.prompt
        assert prompt is not None

        self.ema = ExponentialMovingAverage(self.unet.parameters(), self.ema_rate)
        precision_scope = autocast
        time_start = time.time()
        with precision_scope("cuda"):
            # the precision_scope("cuda") is necessary. Without it, there will be error:
            #   Input type (c10::Half) and bias type (float) should be the same
            for e_idx in range(1, e_cnt + 1):  # Sampling
                log_info(f"E{e_idx:3d}/{e_cnt} ------------------------ lr:{self.lr:}")
                self.unet.train()
                loss_cnt, loss_sum, do_sum, dn_sum = 0, 0.0, 0.0, 0.0
                for b_idx, (ltt, pen, ltt_id, prompt_id) in enumerate(dl):
                    self.batch_counter += 1
                    ltt = ltt.to(args.device)
                    pen = pen.to(args.device)
                    batch_size = len(ltt)
                    uc = None if args.scale == 1.0 else torch.cat([self.empty_prompt_encoding] * batch_size)
                    if self.batch_counter == 1: self.log_stats(ltt, pen, uc)
                    loss, dist_old, dist_new = self.calc_loss(ltt, pen, uc)
                    loss_cnt += 1
                    loss_sum += loss.item()
                    do_sum += dist_old
                    dn_sum += dist_new
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    self.optimizer.step()
                    ema_decay = self.ema.update(self.unet.parameters())
                    if b_idx % log_itv == 0 or b_idx == batch_cnt - 1:
                        elp, eta = get_time_ttl_and_eta(time_start, self.batch_counter, self.batch_total)
                        msg = f"loss:{loss:.6f}, ema: {ema_decay:.6f}."
                        msg += f"dist:{dist_old:.6f}->{dist_new:.6f}. elp:{elp}, eta:{eta}"
                        log_info(f"E{e_idx}.B{b_idx:4d}/{batch_cnt}. {msg}")
                    del loss
                # for batch
                loss_avg = loss_sum / loss_cnt
                dist_old_avg = do_sum / loss_cnt
                dist_new_avg = dn_sum / loss_cnt
                log_info(f"E{e_idx:3d}/{e_cnt}. cnt:{loss_cnt:4d}, loss_avg:{loss_avg:.6f}, "
                         f"dist_old_avg:{dist_old_avg:.6f}, dist_new_avg:{dist_new_avg:.6f}")
                if e_idx % save_itv == 0 or e_idx == e_cnt:
                    self.save_ckpt(e_idx)
                    if save_eval: self.eval_ema(e_idx)
            # for n_epochs
        # with

    def save_ckpt(self, epoch):
        unet, ema = self.model, self.ema
        ckpt_path = self.args.save_ckpt_path
        save_ckpt_dir, base_name = os.path.split(ckpt_path)
        if not os.path.exists(save_ckpt_dir):
            log_info(f"os.makedirs({save_ckpt_dir})")
            os.makedirs(save_ckpt_dir)
        stem, ext = os.path.splitext(base_name)
        ckpt_path = os.path.join(save_ckpt_dir, f"{stem}_E{epoch:04d}{ext}")
        log_info(f"Save ckpt: {ckpt_path}...")
        pure_unet = unet
        if isinstance(pure_unet, torch.nn.DataParallel):
            # save pure model, not DataParallel.
            pure_unet = pure_unet.module
        saved_state = {
            'unet'  : pure_unet.state_dict(),
            'ema'   : ema.state_dict(),
            'epoch' : epoch,
        }
        log_info(f"  model : {type(pure_unet).__name__}")
        log_info(f"  ema   : {type(ema).__name__}")
        log_info(f"  epoch : {saved_state['epoch']}")
        torch.save(saved_state, ckpt_path)
        log_info(f"Save ckpt: {ckpt_path}...Done")

    def eval_ema(self, epoch):
        """generate images with EMA"""
        from ldm.models.diffusion.ddim import DDIMSampler
        import torchvision.utils as tvu

        save_path = self.args.sample_output_dir
        self.unet.eval()
        if not os.path.exists(save_path):
            log_info(f"os.makedirs({save_path})")
            os.makedirs(save_path)
        bs = self.args.batch_size
        dl = tu_data.DataLoader(self.dataset, batch_size=bs, shuffle=False, num_workers=1)
        sampler = DDIMSampler(self.model, device=self.device)
        sample_count = self.args.sample_count
        s_counter = 0
        for b_idx, (ltt, pen, ltt_id, prompt_id) in enumerate(dl):
            if s_counter >= sample_count: break
            # sample and save
            ltt, pen = ltt.to(self.device), pen.to(self.device)
            uc = torch.cat([self.empty_prompt_encoding] * bs)
            if b_idx >= len(self.eval_start_code_arr):
                start_code = torch.randn_like(ltt, device=self.device)
                self.eval_start_code_arr.append(start_code)
                if b_idx == 0: log_info(f"E{epoch}: self.eval_start_code_arr.append(start_code)")
            else:
                start_code = self.eval_start_code_arr[b_idx]
                if b_idx == 0: log_info(f"E{epoch}: start_code = self.eval_start_code_arr[{b_idx}]")
            with torch.no_grad():
                samples, _ = sampler.sample(S=20,
                                            conditioning=pen,
                                            batch_size=bs,
                                            shape=self.shape,
                                            verbose=False,
                                            unconditional_guidance_scale=self.uc_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=0.0,
                                            x_T=start_code)
            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            f_path = None
            for x, pid in zip(x_samples, prompt_id):
                folder = os.path.join(save_path, f"{pid:06d}")
                if not os.path.exists(folder):
                    os.makedirs(folder)
                f_path = os.path.join(folder, f"{pid:06d}_E{epoch:03d}.png")
                tvu.save_image(x, f_path)
                s_counter += 1
            log_info(f"Saved {s_counter:2d}/{sample_count} images: {f_path}")
        # for
        self.unet.train()

    def calc_loss(self, latent, conditioning, unconditional_conditioning):
        batch_size = len(latent)
        if self.noise_select_opt1:
            noise_gt, dist_old, dist_new = self.noise_select_from_set1(latent)  # noise ground truth
        else:
            noise_gt, dist_old, dist_new = self.noise_select_from_set(latent)   # noise ground truth
        ts = torch.randint(low=0, high=self.ddpm_num_timesteps, size=(batch_size,), device=self.device)
        ab_t = self.ab_list.index_select(0, ts).view(-1, 1, 1, 1)  # alpha_bar_t
        noisy_x_t = latent * ab_t.sqrt() + noise_gt * (1.0 - ab_t).sqrt()
        x_in = torch.cat([noisy_x_t] * 2)
        t_in = torch.cat([ts] * 2)
        c_in = torch.cat([unconditional_conditioning, conditioning])
        m_output = self.predict_by_diffusion_model(x_in, t_in, c_in)

        model_uc, model_c = m_output.chunk(2)
        e_t = model_uc + self.uc_guidance_scale * (model_c - model_uc)
        noise_pd = e_t
        loss = (noise_pd - noise_gt).square().mean()
        return loss, dist_old, dist_new

    def noise_select_from_set(self, latent):
        noise = torch.randn_like(latent, device=self.device)
        dist_old = (noise - latent).square().mean()
        ss_size = self.args.noise_sss
        for s_idx in range(2, ss_size+1):
            n2 = torch.randn_like(latent, device=self.device)
            d1 = (noise - latent).square().mean(dim=(1, 2, 3))
            d2 = (n2 - latent).square().mean(dim=(1, 2, 3))
            flag = torch.lt(d2, d1)
            noise[flag] = n2[flag]
            if self.batch_counter > 1: continue # only log once
            updated = flag.sum()
            if updated == 0: continue
            dist_new = (noise - latent).square().mean()
            log_info(f"ss:{s_idx:4d}/{ss_size}: updated {updated:2d}, new distance: {dist_new:.6f}")
        # for
        dist_new = (noise - latent).square().mean()
        return noise, dist_old, dist_new

    def noise_select_from_set1(self, latent_batch):
        noise_batch = torch.randn_like(latent_batch, device=self.device)
        bs, c, h, w = latent_batch.size()
        ss_size = self.args.noise_sss
        dist_old_sum, dist_new_sum = 0., 0.
        for i in range(bs):
            latent = latent_batch[i:i+1]
            n2 = torch.randn((ss_size, c, h, w), device=self.device)
            dist = (n2 - latent).square().mean(dim=(1, 2, 3))
            dist_old = dist[0]  # just take the first one as old distance
            dist_new, idx = torch.min(dist, dim=0)
            dist_old_sum += dist_old
            dist_new_sum += dist_new
            noise_batch[i, :] = n2[idx, :]
            if self.batch_counter == 1: # only log once
                log_info(f"ss:{ss_size} latent[{i:02d}]: distance:{dist_old:.6f}->{dist_new:.6f}")
        # for
        dist_old = dist_old_sum / bs
        dist_new = dist_new_sum / bs
        if self.batch_counter == 1:  # only log once
            dn2 = (noise_batch - latent_batch).square().mean()  # distance new version 2
            assert f"{dist_new:.6f}" == f"{dn2:.6f}", f"dist_new not match. {dist_new:.6f} == {dn2:.6f}"
            log_info(f"ss:{ss_size} dist_new calculation match {dist_new:.6f}. Ready to go.")
        return noise_batch, dist_old, dist_new

    def predict_by_diffusion_model(self, x_in, t_in, c_in):
        """
        predict noise by diffusion_model
        :param x_in: x input, which is the noisy x_t
        :param t_in: timestep input
        :param c_in: condition input
        :return:
        """

        # option 1: use self.model, which is LatentDiffusion
        m_output = self.model.apply_model(x_in, t_in, c_in)

        # option 2: use self.model.model, which is DiffusionWrapper
        # c_in = {'c_crossattn': [c_in]}
        # m_output = self.model.model(x_in, t_in, **c_in)

        # option 3: use self.model.model.diffusion_model, which is UNetModel
        #           And in this class, it is defined as "unet"
        # In this option, the memory improvement is marginal, around 1.7G on each GPU.
        # m_output = self.unet(x_in, t_in, context=c_in)

        return m_output

# class
