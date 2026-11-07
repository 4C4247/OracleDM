"""Gaussian diffusion with DDIM sampling, CFG and structural losses.

Pixel-space diffusion that renders an oracle glyph from a skeleton condition.
In addition to the standard noise-prediction objective, training adds a
timestep-weighted structural loss (edge + foreground occupancy + skeleton
centerline) that is active near x0, sharpening stroke detail and pinning the
generated glyph to the supplied skeleton. Sampling uses DDIM with optional
classifier-free guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


class SkeletonDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 w_edge=0.5, w_occ=0.5, w_skel=0.5, struct_power=2):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.w_edge = w_edge
        self.w_occ = w_occ
        self.w_skel = w_skel
        self.struct_power = struct_power

        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_ac", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_1mac", torch.sqrt(1.0 - alphas_cumprod).float())

        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))

    def q_sample(self, x0, t, noise):
        a = self.sqrt_ac[t][:, None, None, None]
        b = self.sqrt_1mac[t][:, None, None, None]
        return a * x0 + b * noise

    def predict_x0(self, x_t, t, eps):
        a = self.sqrt_ac[t][:, None, None, None]
        b = self.sqrt_1mac[t][:, None, None, None]
        return ((x_t - b * eps) / a).clamp(0.0, 1.0)

    def _sobel(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def _gauss_k(self, k, sigma, device, dtype):
        coords = torch.arange(k, device=device, dtype=dtype) - k // 2
        g1 = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g1 = g1 / g1.sum()
        return (g1[:, None] * g1[None, :]).view(1, 1, k, k)

    def _highpass(self, x, sigma=1.5, k=5):
        gk = self._gauss_k(k, sigma, x.device, x.dtype)
        return x - F.conv2d(x, gk, padding=k // 2)

    def edge_quality_loss(self, x0_pred, x0, hf_sigma=1.5, hf_k=5):
        """Crisp, correctly-placed edges without burrs (Stage3).

        Two terms on the binary line drawing:
          * Sobel boundary L1 aligns the predicted contour to the clean target
            (sharpening thin strokes to their true shape instead of chunks);
          * high-frequency residual L1 penalizes sub-pixel spiky texture
            (burrs) that the clean target does not have. Genuine strokes are
            lower-frequency at 128px and survive; 1-2px spikes do not.
        """
        edge = F.l1_loss(self._sobel(x0_pred), self._sobel(x0))
        hf = F.l1_loss(self._highpass(x0_pred, hf_sigma, hf_k),
                       self._highpass(x0, hf_sigma, hf_k))
        return edge + hf

    def structural_loss(self, x0_pred, x0, cond, t):
        """Edge + occupancy + confidence-gated skeleton-centerline loss."""
        w = (1.0 - t.float() / self.timesteps) ** self.struct_power
        w = w[:, None, None, None]

        # edge / stroke detail (whole image, incl. completion regions)
        edge = (w * (self._sobel(x0_pred) - self._sobel(x0)).abs()).mean()

        # soft foreground occupancy (logits form is autocast-safe)
        logits = (x0_pred - 0.5) * 10.0
        occ = (w * F.binary_cross_entropy_with_logits(
            logits, x0.clamp(0.0, 1.0), reduction="none")).mean()

        # skeleton centerline must be foreground, but ONLY where the skeleton is
        # reliable (conf gating). cond channels: [sk, conf, width, app].
        # Low-conf / erased regions are left free for the model to complete.
        sk = (cond[:, 0:1] > 0.5).float()
        conf = cond[:, 1:2].clamp(0.0, 1.0)
        wgt = w * sk * conf
        denom = wgt.sum().clamp(min=1.0)
        skel = (wgt * (1.0 - x0_pred)).sum() / denom

        return self.w_edge * edge + self.w_occ * occ + self.w_skel * skel

    def trajectory_loss(self, x0, cond, step=40, k_lo=1, k_hi=4,
                        t_anchor=40, w_x0=1.0, w_fill=1.0, subset=16,
                        w_edge=0.0):
        """Trajectory-consistency loss at the end of the DDIM chain.

        Single-step x0 supervision sees ideal Gaussian noise (q_sample), but at
        inference the last denoising steps are fed the model's OWN partly
        denoised states (exposure bias). Those states carry the high-frequency
        texture / pinholes that a one-step loss never sees. This method runs
        the model's own eta=0 DDIM chain (intermediate steps no-grad) from a
        low-t q-sampled state down to a low-t anchor, then supervises the
        final x0 estimate (with grad) on that model-produced state. It directly
        teaches the model to emit solid strokes along its own sampling path.

        Cheap by design: only a small `subset` of the batch is rolled, and the
        roll is short (k_lo..k_hi no-grad steps), because solidity is set in
        the very last few steps (the final output ~= x0_pred at t=40).
        """
        dev = x0.device
        r = min(subset, x0.shape[0])
        # random subset each step so every image is eventually covered
        idx = torch.randperm(x0.shape[0], device=dev)[:r]
        x0 = x0[idx]
        cond = cond[idx]
        b = x0.shape[0]
        k = int(torch.randint(k_lo, k_hi, (1,), device=dev).item())
        t_start = t_anchor + k * step
        ac = self.alphas_cumprod
        ts = torch.full((b,), t_start, device=dev, dtype=torch.long)
        x = self.q_sample(x0, ts, torch.randn_like(x0))

        # roll down the model's own chain (no grad) to the anchor state
        with torch.no_grad():
            cur = t_start
            while cur > t_anchor:
                nxt = cur - step
                tb = torch.full((b,), cur, device=dev, dtype=torch.long)
                eps = self.model(x, tb, cond)
                x0p = self.predict_x0(x, tb, eps)
                a_prev = ac[nxt] if nxt >= 0 else torch.tensor(1.0, device=dev)
                x = torch.sqrt(a_prev) * x0p + torch.sqrt(1.0 - a_prev) * eps
                cur = nxt

        # final anchored denoising step WITH grad -> the final clean estimate
        tb = torch.full((b,), t_anchor, device=dev, dtype=torch.long)
        eps = self.model(x, tb, cond)
        x0_pred = self.predict_x0(x, tb, eps)

        loss = x0_pred.sum() * 0.0
        if w_x0 > 0.0:
            loss = loss + w_x0 * F.l1_loss(x0_pred, x0)
        if w_fill > 0.0:
            fg = (x0 > 0.5).float()
            miss = (x0 - x0_pred).clamp(min=0.0)
            loss = loss + w_fill * (
                (miss * fg).sum() / fg.sum().clamp(min=1.0))
        # Stage3: clean, crisp edges on the model's own low-t path (burrs).
        if w_edge > 0.0:
            loss = loss + w_edge * self.edge_quality_loss(x0_pred, x0)
        return loss

    def training_loss(self, x0, cond, labels=None, glyph=None,
                      w_perc=0.0, w_rec=0.0, perc_max_t=300,
                      w_x0=0.0, w_fill=0.0, w_roll=0.0,
                      roll_k_lo=1, roll_k_hi=4, roll_subset=16,
                      roll_w_edge=0.0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps = self.model(x_t, t, cond)
        noise_loss = F.mse_loss(eps, noise)
        x0_pred = self.predict_x0(x_t, t, eps)
        struct_loss = self.structural_loss(x0_pred, x0, cond, t)
        total = noise_loss + struct_loss

        # All clean-image (x0) losses are only applied at low t, where the
        # predicted clean image is meaningful.
        if (glyph is not None and (w_perc > 0.0 or w_rec > 0.0)) \
                or w_x0 > 0.0 or w_fill > 0.0:
            low = t < perc_max_t
            if low.any():
                xp, xt = x0_pred[low], x0[low]

                # Stage2: domain perceptual + recognition consistency.
                if glyph is not None and w_perc > 0.0:
                    total = total + w_perc * glyph.perceptual(xp, xt)
                if glyph is not None and w_rec > 0.0 and labels is not None:
                    total = total + w_rec * glyph.recognition(
                        xp, labels[low])

                # Stage2.5: direct clean-image fidelity (constant-gradient L1,
                # unlike the saturating occupancy BCE) -> recovers PSNR and
                # drives the interior toward the solid target.
                if w_x0 > 0.0:
                    total = total + w_x0 * F.l1_loss(xp, xt)

                # Stage2.5: foreground coverage (solid strokes). One-sided,
                # normalized over target-foreground pixels, so it is NOT diluted
                # by the large background. It only fires where the TARGET is a
                # stroke, so it can fill holes but can never hallucinate.
                if w_fill > 0.0:
                    fg = (xt > 0.5).float()
                    miss = (xt - xp).clamp(min=0.0)
                    total = total + w_fill * (
                        (miss * fg).sum() / fg.sum().clamp(min=1.0))

        # Stage2.6: trajectory-consistency solidity (model's own DDIM path).
        # The trajectory uses fixed unit x0/fill weights; w_roll is its knob.
        if w_roll > 0.0:
            total = total + w_roll * self.trajectory_loss(
                x0, cond, k_lo=roll_k_lo, k_hi=roll_k_hi,
                w_x0=1.0, w_fill=1.0, subset=roll_subset,
                w_edge=roll_w_edge)
        return total

    def _eps(self, x, t, cond, guidance_scale):
        if guidance_scale is not None and guidance_scale != 1.0:
            zero = torch.zeros_like(cond)
            eps_c = self.model(x, t, cond)
            eps_u = self.model(x, t, zero)
            return eps_u + guidance_scale * (eps_c - eps_u)
        return self.model(x, t, cond)

    @torch.no_grad()
    def ddim_sample(self, cond, num_steps=50, guidance_scale=1.0,
                    eta=0.0, generator=None):
        device = cond.device
        b = cond.shape[0]
        size = cond.shape[-1]

        # evenly spaced timestep subset
        step = self.timesteps // num_steps
        timesteps = torch.arange(0, self.timesteps, step, device=device).flip(0)

        shape = (b, 1, size, size)
        x = torch.randn(shape, device=device, generator=generator)

        ac = self.alphas_cumprod
        for t in timesteps:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            eps = self._eps(x, t_batch, cond, guidance_scale)

            ac_t = ac[t]
            ac_prev = ac[t - step] if t - step >= 0 else torch.tensor(
                1.0, device=device)

            x0_pred = self.predict_x0(x, t_batch, eps)

            # DDIM deterministic (eta=0) update
            x = torch.sqrt(ac_prev) * x0_pred + torch.sqrt(
                1 - ac_prev) * eps
        return x.clamp(0.0, 1.0)
