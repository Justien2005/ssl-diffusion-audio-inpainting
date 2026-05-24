import os
import sys
from contextlib import contextmanager

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from official_cqtdiff_adapter import DiffusionParams


@contextmanager
def _prepend_path(path):
    path = os.path.abspath(path)
    already_present = path in sys.path
    if not already_present:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if not already_present:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


@contextmanager
def _numpy_clip_int_inf_compat():
    """Compatibility for cqt_nsgt_pytorch 0.0.8 under NumPy 2.x."""
    original_clip = np.clip

    def clip_compat(a, a_min=None, a_max=None, out=None, **kwargs):
        if out is not None and np.issubdtype(np.asarray(out).dtype, np.integer):
            try:
                max_is_inf = bool(np.all(np.isinf(a_max)))
            except TypeError:
                max_is_inf = False
            if max_is_inf:
                result = original_clip(a, a_min, None, **kwargs)
                np.copyto(out, np.asarray(result, dtype=out.dtype), casting="unsafe")
                return out
        return original_clip(a, a_min, a_max, out=out, **kwargs)

    np.clip = clip_compat
    try:
        yield
    finally:
        np.clip = original_clip


def _load_audio_inpainting_config(repo_dir, device):
    conf_dir = os.path.join(repo_dir, "conf")
    network_cfg = os.environ.get(
        "AUDIO_INPAINTING_NETWORK", "paper_1912_unet_cqt_oct_attention_44k_2"
    )
    exp_cfg = os.environ.get("AUDIO_INPAINTING_EXP", "musicnet44k_4s")
    diff_cfg = os.environ.get("AUDIO_INPAINTING_DIFF_PARAMS", "edm")

    paths = {
        "network": os.path.join(conf_dir, "network", f"{network_cfg}.yaml"),
        "exp": os.path.join(conf_dir, "exp", f"{exp_cfg}.yaml"),
        "diff_params": os.path.join(conf_dir, "diff_params", f"{diff_cfg}.yaml"),
    }
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Audio-inpainting config tidak ditemukan: " + ", ".join(missing))

    cfg = OmegaConf.create(
        {
            "network": OmegaConf.load(paths["network"]),
            "exp": OmegaConf.load(paths["exp"]),
            "diff_params": OmegaConf.load(paths["diff_params"]),
            "device": str(device),
            "model_dir": "experiments/audio_inpainting_musicnet",
        }
    )
    cfg.exp.sample_rate = int(cfg.exp.sample_rate)
    cfg.exp.audio_len = int(cfg.exp.audio_len)
    cfg.exp.resample_factor = int(cfg.exp.resample_factor)
    return cfg


def _find_audio_inpainting_weights(repo_dir):
    candidates = [
        os.environ.get("AUDIO_INPAINTING_CQTDIFF_WEIGHTS"),
        os.path.join(repo_dir, "experiments", "musicnet_44k_4s-560000.pt"),
        os.path.join(repo_dir, "experiments", "musicnet_44k_4s_560000.pt"),
        os.path.join(repo_dir, "musicnet_44k_4s-560000.pt"),
        os.path.join(repo_dir, "musicnet_44k_4s_560000.pt"),
    ]
    exp_dir = os.path.join(repo_dir, "experiments")
    if os.path.isdir(exp_dir):
        candidates.extend(
            os.path.join(exp_dir, name)
            for name in sorted(os.listdir(exp_dir))
            if name.endswith(".pt") and ("musicnet" in name.lower() or "44k_4s" in name.lower())
        )
    return next((path for path in candidates if path and os.path.exists(path)), None)


def _strip_module_prefix(state):
    return {str(k).replace("module.", "", 1): v for k, v in state.items()}


def _checkpoint_to_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint format tidak dikenali.")

    for key in ("ema", "network", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return _strip_module_prefix(value), key

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return _strip_module_prefix(checkpoint), "raw"

    raise KeyError(f"Tidak menemukan state dict model di checkpoint keys={list(checkpoint.keys())}")


class OfficialAudioInpaintingCQTDiffDecoder(nn.Module):
    """
    Adapter for the 44.1 kHz MusicNet CQTdiff+ model from audio-inpainting-diffusion.

    It preserves the same decoder interface used by code_final_run_v2.py:
    baseline calls inpaint(..., conditioning=None), while hybrid models pass SSL
    conditioning into the diffusion denoiser.
    """

    DIFFUSION_STEPS = int(os.environ.get("CQTDIFF_DIFFUSION_STEPS", "35"))
    DIFFUSION_SIGMA_MIN = float(os.environ.get("CQTDIFF_SIGMA_MIN", "1e-4"))
    DIFFUSION_SIGMA_MAX = float(os.environ.get("CQTDIFF_SIGMA_MAX", "1.0"))
    DIFFUSION_SCHURN = float(os.environ.get("CQTDIFF_SCHURN", "10"))
    DIFFUSION_SIGMA_DATA = float(os.environ.get("CQTDIFF_SIGMA_DATA", "0.063"))

    def __init__(self, device, target_sr, segment_samples, gap_durations_ms, audio_inpainting_dir):
        super().__init__()
        self.device_ref = torch.device(device)
        self.target_sr = int(target_sr)
        self.target_len = int(segment_samples)
        self.gap_durations_ms = list(gap_durations_ms)
        self.audio_inpainting_dir = os.path.abspath(audio_inpainting_dir)
        self.architecture_name = "ssl_conditioned_audio_inpainting_cqtdiffplus_musicnet44k_strongcond_v2"

        with _prepend_path(self.audio_inpainting_dir):
            from networks.unet_cqt_oct_with_projattention_adaLN_2 import (
                Unet_CQT_oct_with_attention,
            )

            self.args = _load_audio_inpainting_config(self.audio_inpainting_dir, self.device_ref)
            self.native_sr = int(self.args.exp.sample_rate)
            self.native_len = int(self.args.exp.audio_len)
            with _numpy_clip_int_inf_compat():
                self.backbone = Unet_CQT_oct_with_attention(self.args, self.device_ref).to(
                    self.device_ref
                )

        require_native = os.environ.get("CQTDIFF_REQUIRE_NATIVE_SHAPE", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if require_native and (self.target_sr != self.native_sr or self.target_len != self.native_len):
            raise RuntimeError(
                "Audio-inpainting MusicNet CQTdiff+ harus berjalan pada native checkpoint shape: "
                f"target_sr={self.target_sr}, target_len={self.target_len}, "
                f"native_sr={self.native_sr}, native_len={self.native_len}. "
                "Gunakan 44100 Hz dan 184184 samples, atau set "
                "CQTDIFF_REQUIRE_NATIVE_SHAPE=0 hanya untuk diagnosis."
            )

        weights_path = _find_audio_inpainting_weights(self.audio_inpainting_dir)
        if weights_path is None:
            raise FileNotFoundError(
                "Checkpoint MusicNet CQTdiff+ belum ditemukan. Download "
                "musicnet_44k_4s-560000.pt dari "
                "https://huggingface.co/Eloimoliner/audio-inpainting-diffusion "
                "atau set AUDIO_INPAINTING_CQTDIFF_WEIGHTS."
            )
        self.official_weights_path = os.path.abspath(weights_path)

        checkpoint = torch.load(weights_path, map_location=self.device_ref, weights_only=False)
        state, weight_type = _checkpoint_to_state_dict(checkpoint)
        self.official_weights_type = weight_type
        msg = self.backbone.load_state_dict(state, strict=False)
        print(f"Audio-inpainting CQTdiff+ {weight_type} weights loaded: {weights_path}")
        print(f"Audio-inpainting CQTdiff+ load_state_dict: {msg}")
        print(
            "Audio-inpainting MusicNet native setup: "
            f"sr={self.native_sr} Hz, len={self.native_len} samples, "
            f"duration={self.native_len / self.native_sr:.3f}s"
        )

        train_backbone = os.environ.get("CQTDIFF_TRAIN_BACKBONE", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(
            "Audio-inpainting CQTdiff+ backbone training: "
            f"{'enabled' if train_backbone else 'disabled'} ({trainable:,}/{total:,})"
        )

        self.diff_params = DiffusionParams(
            sigma_data=self.DIFFUSION_SIGMA_DATA,
            sigma_min=self.DIFFUSION_SIGMA_MIN,
            sigma_max=self.DIFFUSION_SIGMA_MAX,
            ro=13,
            Schurn=self.DIFFUSION_SCHURN,
            Snoise=1.0,
            Stmin=0,
            Stmax=50,
        )
        print(
            f"Diffusion sampling config: T={self.DIFFUSION_STEPS}, "
            f"sigma=[{self.DIFFUSION_SIGMA_MIN}, {self.DIFFUSION_SIGMA_MAX}], "
            f"Schurn={self.DIFFUSION_SCHURN}, sigma_data={self.DIFFUSION_SIGMA_DATA}"
        )

        cutoff = float(os.environ.get("CQTDIFF_FINAL_LOWPASS_HZ", "20000"))
        cutoff = min(cutoff, self.native_sr / 2 - 100.0)
        lpf_coeffs = scipy.signal.firwin(
            numtaps=100, cutoff=cutoff, width=1, window="kaiser", fs=self.native_sr
        )
        self.register_buffer("_lpf_kernel", torch.FloatTensor(lpf_coeffs).unsqueeze(0).unsqueeze(0))

        self.n_fft = int(os.environ.get("CQTDIFF_ADAPTER_N_FFT", 4096))
        self.hop_length = int(os.environ.get("CQTDIFF_ADAPTER_HOP", 1024))
        self.feature_dim = int(os.environ.get("CQTDIFF_ADAPTER_FEATURE_DIM", 256))
        self.freq_bins = self.n_fft // 2 + 1
        self.sigma = float(os.environ.get("CQTDIFF_ADAPTER_SIGMA", "0.1"))
        self.condition_gate_init = float(os.environ.get("CQTDIFF_CONDITION_GATE_INIT", "-1.0"))
        self.cond_residual_scale = float(os.environ.get("CQTDIFF_COND_RESIDUAL_SCALE", "2.0"))

        self.feature_encoder = nn.Sequential(
            nn.Linear(self.freq_bins * 2 + 1, 512),
            nn.SiLU(),
            nn.Linear(512, self.feature_dim),
            nn.SiLU(),
        )
        self.spec_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.SiLU(),
            nn.Linear(512, self.freq_bins * 2),
        )
        self.condition_gate = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.condition_gate[-1].weight)
        nn.init.constant_(self.condition_gate[-1].bias, self.condition_gate_init)

        self.sigma_scale_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.sigma_scale_net[-1].weight)
        nn.init.zeros_(self.sigma_scale_net[-1].bias)
        print(
            "CQT SSL conditioning config: "
            f"gate_init={self.condition_gate_init:g} "
            f"(initial_gate={torch.sigmoid(torch.tensor(self.condition_gate_init)).item():.3f}), "
            f"residual_scale={self.cond_residual_scale:g}"
        )

    def _pad_or_crop(self, x, length):
        if x.shape[-1] < length:
            return F.pad(x, (0, length - x.shape[-1]))
        if x.shape[-1] > length:
            start = (x.shape[-1] - length) // 2
            return x[..., start : start + length]
        return x

    def _to_native(self, audio):
        x = audio.squeeze(1) if audio.dim() == 3 else audio
        x = x.float()
        if self.target_sr != self.native_sr:
            x = torchaudio.functional.resample(x, self.target_sr, self.native_sr)
        return self._pad_or_crop(x, self.native_len)

    def _from_native(self, audio, target_len):
        x = audio.squeeze(1) if audio.dim() == 3 else audio
        if self.native_sr != self.target_sr:
            x = torchaudio.functional.resample(x, self.native_sr, self.target_sr)
        return self._pad_or_crop(x, target_len)

    def _target_mask_to_native_keep(self, user_mask):
        gap = user_mask.float()
        if self.target_sr != self.native_sr:
            gap = torchaudio.functional.resample(gap, self.target_sr, self.native_sr)
        gap = self._pad_or_crop(gap, self.native_len)
        return (gap < 0.5).float()

    def _apply_cqt_dc_filter(self, x):
        try:
            return self.backbone.CQTransform.apply_hpf_DC(x)
        except Exception:
            return x

    def _conditioning_to_waveform(self, conditioning, target_len, return_stats=False):
        if conditioning is None:
            return (None, {}) if return_stats else None
        conditioning = conditioning.to(self.device_ref, dtype=torch.float32)
        if conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1)

        pred_spec_features = self.decode_features(conditioning).permute(0, 2, 1).contiguous()
        with torch.autocast(device_type=self.device_ref.type, enabled=False):
            bsz, _, n_frames = pred_spec_features.shape
            pred_pairs = (
                pred_spec_features.float()
                .reshape(bsz, self.freq_bins, 2, n_frames)
                .permute(0, 1, 3, 2)
                .contiguous()
            )
            recon_spec = torch.view_as_complex(pred_pairs)
            window = torch.hann_window(self.n_fft, device=pred_spec_features.device)
            wave = torch.istft(
                recon_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                length=target_len,
            )

        pooled = conditioning.mean(dim=1)
        gate = torch.sigmoid(self.condition_gate(pooled)).clamp(0.0, 1.0)
        conditioned_wave = gate * wave
        scaled_conditioned_wave = self.cond_residual_scale * conditioned_wave
        if return_stats:
            stats = {
                "condition_gate_mean": gate.detach().mean(),
                "conditioning_wave_rms": torch.sqrt(wave.detach().pow(2).mean().clamp_min(1e-12)),
                "conditioned_residual_rms": torch.sqrt(
                    conditioned_wave.detach().pow(2).mean().clamp_min(1e-12)
                ),
                "conditioned_residual_scaled_rms": torch.sqrt(
                    scaled_conditioned_wave.detach().pow(2).mean().clamp_min(1e-12)
                ),
            }
            return scaled_conditioned_wave, stats
        return scaled_conditioned_wave

    def _conditioned_model(self, conditioning, target_len, return_stats=False):
        if return_stats:
            cond_wave_target, cond_stats = self._conditioning_to_waveform(
                conditioning, target_len, return_stats=True
            )
        else:
            cond_wave_target = self._conditioning_to_waveform(conditioning, target_len)
            cond_stats = {}
        if cond_wave_target is None:
            return (self.backbone, cond_stats) if return_stats else self.backbone

        cond_wave_native = self._to_native(cond_wave_target)
        sigma_scale_fn = self.sigma_scale_net

        def model(x, sigma):
            backbone_trainable = any(param.requires_grad for param in self.backbone.parameters())
            if backbone_trainable:
                base = self.backbone(x, sigma)
            else:
                self.backbone.eval()
                with torch.no_grad():
                    base = self.backbone(x, sigma)
            residual = cond_wave_native.to(device=x.device, dtype=x.dtype)
            if residual.shape[0] != x.shape[0]:
                residual = residual.expand(x.shape[0], -1)

            sigma_val = sigma
            if sigma_val.dim() == 0:
                sigma_val = sigma_val.view(1, 1)
            elif sigma_val.dim() == 1:
                sigma_val = sigma_val.unsqueeze(-1)
            scale = 1.0 + sigma_scale_fn(sigma_val)
            return base + scale * residual

        return (model, cond_stats) if return_stats else model

    def _run_diffusion_sampling(self, y, mask, T=None, show_progress=True, conditioning=None):
        if T is None:
            T = self.DIFFUSION_STEPS
        device = y.device
        shape = y.shape
        denoiser_model = self._conditioned_model(conditioning, target_len=self.target_len)

        t = self.diff_params.create_schedule(T).to(device)
        x = self.diff_params.sample_prior(shape, t[0]).to(device)
        gamma = self.diff_params.get_gamma(t).to(device)

        self.backbone.eval()
        with torch.no_grad():
            for i in tqdm(range(T), desc="Diffusion inpainting", leave=False, disable=not show_progress):
                if gamma[i] == 0:
                    t_hat = t[i]
                    x_hat = x
                else:
                    t_hat = t[i] + gamma[i] * t[i]
                    epsilon = torch.randn(shape, device=device) * self.diff_params.Snoise
                    x_hat = x + ((t_hat**2 - t[i] ** 2) ** 0.5) * epsilon

                denoised = self.diff_params.denoiser(x_hat, denoiser_model, t_hat.unsqueeze(-1))
                denoised = self._apply_cqt_dc_filter(denoised)
                denoised = mask * y + (1.0 - mask) * denoised

                score = (denoised - x_hat) / t_hat**2
                d = -t_hat * score
                h = t[i + 1] - t_hat

                if t[i + 1] != 0:
                    x_prime = x_hat + h * d
                    denoised_prime = self.diff_params.denoiser(
                        x_prime, denoiser_model, t[i + 1].unsqueeze(-1)
                    )
                    denoised_prime = self._apply_cqt_dc_filter(denoised_prime)
                    denoised_prime = mask * y + (1.0 - mask) * denoised_prime

                    score_prime = (denoised_prime - x_prime) / t[i + 1] ** 2
                    d_prime = -t[i + 1] * score_prime
                    x = x_hat + h * (0.5 * d + 0.5 * d_prime)
                else:
                    x = x_hat + h * d

        return x.detach()

    def _apply_lowpass(self, x):
        return F.conv1d(
            x.unsqueeze(1),
            self._lpf_kernel.to(x.device),
            padding="same",
        ).squeeze(1)

    def _inpaint_diffusion(self, masked_audio, mask, T=None, show_progress=True, conditioning=None):
        bsz = masked_audio.shape[0]
        target_len = masked_audio.shape[-1]
        if os.environ.get("CQTDIFF_REQUIRE_NATIVE_SHAPE", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            if target_len != self.native_len or mask.shape[-1] != self.native_len:
                raise RuntimeError(
                    f"Input harus native_len={self.native_len}; "
                    f"audio={target_len}, mask={mask.shape[-1]}."
                )

        mask_bool = mask[0].bool()
        gap_indices = torch.where(mask_bool)[0]
        if len(gap_indices) == 0:
            out = masked_audio.detach().cpu().numpy()
            return out[0] if bsz == 1 else out

        gap_start_orig = gap_indices[0].item()
        gap_end_orig = gap_indices[-1].item() + 1

        audio_native = self._to_native(masked_audio)
        crop_offset = 0
        sr_scale = self.native_sr / self.target_sr
        gap_start_nat = int(round(gap_start_orig * sr_scale)) - crop_offset
        gap_end_nat = int(round(gap_end_orig * sr_scale)) - crop_offset
        gap_start_nat = max(0, gap_start_nat)
        gap_end_nat = min(self.native_len, gap_end_nat)

        cqtdiff_mask = torch.ones(
            (bsz, self.native_len), device=masked_audio.device, dtype=torch.float32
        )
        cqtdiff_mask[..., gap_start_nat:gap_end_nat] = 0.0
        y = cqtdiff_mask * audio_native

        x_hat = self._run_diffusion_sampling(
            y, cqtdiff_mask, T=T, show_progress=show_progress, conditioning=conditioning
        )
        x_hat = self._apply_lowpass(x_hat)

        gap_native = x_hat[..., gap_start_nat:gap_end_nat].clone()
        gap_target = self._from_native(gap_native, gap_end_orig - gap_start_orig)

        output = masked_audio.clone()
        output[..., gap_start_orig:gap_end_orig] = gap_target
        out = output.detach().cpu().numpy()
        return out[0] if bsz == 1 else out

    _DETERMINISTIC_SIGMA_STEPS = 8

    def diffusion_loss(self, clean_audio, masked_audio, mask, conditioning=None, deterministic_sigma=False):
        if os.environ.get("CQTDIFF_REQUIRE_NATIVE_SHAPE", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            if clean_audio.shape[-1] != self.native_len or masked_audio.shape[-1] != self.native_len:
                raise RuntimeError(
                    f"Training harus native_len={self.native_len}; "
                    f"clean={clean_audio.shape[-1]}, masked={masked_audio.shape[-1]}."
                )
        clean_native = self._to_native(clean_audio).float()
        masked_native = self._to_native(masked_audio).float()
        keep_mask = self._target_mask_to_native_keep(mask).to(clean_native.device)
        gap_mask = 1.0 - keep_mask

        batch_size = clean_native.shape[0]
        if deterministic_sigma:
            log_sigma_grid = torch.linspace(
                np.log(self.DIFFUSION_SIGMA_MIN),
                np.log(self.DIFFUSION_SIGMA_MAX),
                steps=self._DETERMINISTIC_SIGMA_STEPS,
                device=clean_native.device,
            )
            idx = torch.arange(batch_size, device=clean_native.device) % self._DETERMINISTIC_SIGMA_STEPS
            sigma = torch.exp(log_sigma_grid[idx]).unsqueeze(1)
        else:
            sigma = torch.exp(
                torch.empty(batch_size, 1, device=clean_native.device).uniform_(
                    np.log(self.DIFFUSION_SIGMA_MIN), np.log(self.DIFFUSION_SIGMA_MAX)
                )
            )
        noise = torch.randn_like(clean_native) * sigma
        x_noisy = keep_mask * masked_native + gap_mask * (clean_native + noise)

        denoiser_model, cond_stats = self._conditioned_model(
            conditioning, target_len=clean_audio.shape[-1], return_stats=True
        )
        denoised = self.diff_params.denoiser(x_noisy, denoiser_model, sigma.squeeze(1))
        denoised = self._apply_cqt_dc_filter(denoised)
        denoised = keep_mask * masked_native + gap_mask * denoised

        denom = gap_mask.sum(dim=1).clamp_min(1.0)
        per_sample_gap_l1 = ((denoised - clean_native).abs() * gap_mask).sum(dim=1) / denom
        gap_loss = per_sample_gap_l1.mean()
        full_loss = (denoised - clean_native).abs().mean(dim=1).mean()

        pred_rms = torch.sqrt(((denoised * gap_mask).pow(2).sum(dim=1) / denom).clamp_min(1e-10))
        target_rms = torch.sqrt(((clean_native * gap_mask).pow(2).sum(dim=1) / denom).clamp_min(1e-10))
        energy_loss = F.l1_loss(torch.log(pred_rms + 1e-5), torch.log(target_rms + 1e-5))
        loss = gap_loss + 0.1 * full_loss + 0.05 * energy_loss
        return {
            "loss": loss,
            "gap_loss": gap_loss,
            "full_loss": full_loss,
            "energy_loss": energy_loss,
            **cond_stats,
        }

    def inpaint(self, masked_audio, mask, conditioning=None, diffusion_audio=None):
        if masked_audio.dim() == 1:
            masked_audio = masked_audio.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        masked_audio = masked_audio.to(self.device_ref, dtype=torch.float32)
        mask = mask.to(self.device_ref)
        return self._inpaint_diffusion(masked_audio, mask, conditioning=conditioning)

    def _backbone_predict(self, audio):
        with torch.autocast(device_type=self.device_ref.type, enabled=False):
            native = self._to_native(audio).float()
            sigma = torch.full(
                (native.shape[0], 1), self.sigma, device=native.device, dtype=torch.float32
            )
            backbone_trainable = any(param.requires_grad for param in self.backbone.parameters())
            if backbone_trainable:
                pred = self.backbone(native, sigma)
            else:
                self.backbone.eval()
                with torch.no_grad():
                    pred = self.backbone(native, sigma)
            return self._from_native(pred.float(), audio.shape[-1])

    def _sample_to_frame_mask(self, mask, n_frames):
        pooled = F.avg_pool1d(
            mask.float().unsqueeze(1),
            kernel_size=self.hop_length,
            stride=self.hop_length,
            ceil_mode=True,
        ).squeeze(1)
        if pooled.shape[1] < n_frames:
            pooled = F.pad(pooled, (0, n_frames - pooled.shape[1]))
        elif pooled.shape[1] > n_frames:
            pooled = pooled[:, :n_frames]
        return pooled

    def get_features(self, x, mask=None):
        with torch.autocast(device_type=self.device_ref.type, enabled=False):
            x = x.squeeze(1) if x.dim() == 3 else x
            x = x.float()
            backbone_pred = self._backbone_predict(x)
            base = torch.where(mask.bool(), backbone_pred, x) if mask is not None else backbone_pred

            window = torch.hann_window(self.n_fft, device=base.device)
            spec = torch.stft(
                base.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            )
            spec_ri = torch.view_as_real(spec).permute(0, 2, 1, 3).contiguous()
            spec_features = spec_ri.reshape(spec.shape[0], spec.shape[2], self.freq_bins * 2)
            if mask is None:
                frame_mask = torch.zeros(
                    spec_features.shape[0],
                    spec_features.shape[1],
                    device=spec_features.device,
                    dtype=spec_features.dtype,
                )
            else:
                frame_mask = self._sample_to_frame_mask(mask, spec_features.shape[1]).to(
                    spec_features.dtype
                )
            return self.feature_encoder(
                torch.cat([spec_features, frame_mask.unsqueeze(-1)], dim=-1).float()
            )

    def decode_features(self, features):
        return self.spec_decoder(features)

    def forward(self, x, mask=None, conditioning=None):
        features = self.get_features(x, mask)
        if conditioning is not None:
            if conditioning.dim() == 2:
                conditioning = conditioning.unsqueeze(1)
            features = features + conditioning
        return self.decode_features(features)


def build_cqtdiff_decoder(
    device, target_sr, segment_samples, gap_durations_ms, cqt_diff_dir=None, audio_inpainting_dir=None
):
    repo_dir = (
        audio_inpainting_dir
        or os.environ.get("AUDIO_INPAINTING_DIR")
        or os.path.join(os.getcwd(), "external", "audio-inpainting-diffusion")
    )
    return OfficialAudioInpaintingCQTDiffDecoder(
        device=device,
        target_sr=target_sr,
        segment_samples=segment_samples,
        gap_durations_ms=gap_durations_ms,
        audio_inpainting_dir=repo_dir,
    ).to(device)
