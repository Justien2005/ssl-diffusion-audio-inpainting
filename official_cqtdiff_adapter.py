import os
import sys
from contextlib import contextmanager

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


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


# ============================================================
# Diffusion Parameters (Karras VE-SDE Elucidating)
# Replika ringan dari src/sde.py di repo CQTdiff
# ============================================================

class DiffusionParams:
    """Karras VE-SDE noise schedule + preconditioning buat sampling."""

    def __init__(
        self,
        sigma_data=0.057,
        sigma_min=1e-4,
        sigma_max=1.0,
        ro=13,
        Schurn=5,
        Snoise=1.0,
        Stmin=0,
        Stmax=50,
    ):
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ro = ro
        self.Schurn = Schurn
        self.Snoise = Snoise
        self.Stmin = Stmin
        self.Stmax = Stmax

    def create_schedule(self, nb_steps):
        """Buat jadwal noise level dari sigma_max ke 0."""
        i = torch.arange(0, nb_steps + 1)
        t = (
            self.sigma_max ** (1 / self.ro)
            + i / (nb_steps - 1) * (self.sigma_min ** (1 / self.ro) - self.sigma_max ** (1 / self.ro))
        ) ** self.ro
        t[-1] = 0
        return t

    def sample_prior(self, shape, sigma):
        if torch.is_tensor(sigma):
            return torch.randn(shape, device=sigma.device, dtype=sigma.dtype) * sigma
        return torch.randn(shape) * sigma

    def get_gamma(self, t):
        """Stochasticity parameter per timestep."""
        N = t.shape[0]
        gamma = torch.zeros_like(t)
        indexes = torch.logical_and(t > self.Stmin, t < self.Stmax)
        gamma[indexes] = min(self.Schurn / N, 2 ** 0.5 - 1)
        return gamma

    def cskip(self, sigma):
        return self.sigma_data ** 2 * (sigma ** 2 + self.sigma_data ** 2) ** -1

    def cout(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** (-0.5)

    def cin(self, sigma):
        return (self.sigma_data ** 2 + sigma ** 2) ** (-0.5)

    def cnoise(self, sigma):
        return (1 / 4) * torch.log(sigma + 1e-44)

    def denoiser(self, x, model, sigma):
        """Full denoiser step: preconditioning + model forward."""
        sigma = sigma.unsqueeze(-1)
        return self.cskip(sigma) * x + self.cout(sigma) * model(self.cin(sigma) * x, self.cnoise(sigma))


# ============================================================
# Helper functions
# ============================================================

def _load_cqtdiff_config(cqt_diff_dir, device):
    try:
        from omegaconf import OmegaConf
    except Exception as exc:
        raise RuntimeError(
            "CQTdiff membutuhkan omegaconf/hydra-core. Install requirements repo CQTdiff terlebih dahulu."
        ) from exc

    cfg_path = os.path.join(cqt_diff_dir, "conf", "conf.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config CQTdiff tidak ditemukan: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg.device = str(device)
    cfg.log = False
    cfg.restore = False
    cfg.save_model = False
    return cfg




def _find_cqtdiff_weights(cqt_diff_dir):
    candidates = [
        os.environ.get("CQTDIFF_WEIGHTS"),
        os.path.join(cqt_diff_dir, "experiments", "cqt", "cqt_weights.pt"),
    ]
    weights_dir = os.path.join(cqt_diff_dir, "experiments", "cqt")
    if os.path.isdir(weights_dir):
        candidates.extend(
            os.path.join(weights_dir, name)
            for name in sorted(os.listdir(weights_dir))
            if name.startswith("weights-") and name.endswith(".pt")
        )
    return next((path for path in candidates if path and os.path.exists(path)), None)


def _strip_module_prefix(state):
    return {str(k).replace("module.", "", 1): v for k, v in state.items()}


def _load_ema_weights_from_checkpoint(weights_path, device, live_model=None):
    """
    Load EMA weights dari checkpoint CQT-Diff.
    EMA lebih stabil buat inference dibanding raw training weights.

    Parameter live_model diperlukan untuk menentukan key mana yang
    merupakan parameter (trainable) vs buffer (batch norm stats dll).
    EMA weights cuma ada buat parameter, bukan buffer.
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint format tidak dikenali: {weights_path}")

    if "ema_weights" not in checkpoint or "model" not in checkpoint:
        state = checkpoint.get("model", checkpoint)
        return _strip_module_prefix(state), "raw"

    model_state = checkpoint["model"]
    ema_list = checkpoint["ema_weights"]

    # Cari tahu key mana yang parameter (trainable) vs buffer
    if live_model is not None:
        param_keys = {n for n, _ in live_model.named_parameters()}
    else:
        param_keys = None

    dic_ema = {}
    ema_idx = 0

    for key in model_state.keys():
        clean_key = str(key).replace("module.", "", 1)

        is_param = False
        if param_keys is not None:
            is_param = clean_key in param_keys or key in param_keys
        else:
            is_param = model_state[key].is_floating_point()

        if is_param and ema_idx < len(ema_list):
            dic_ema[clean_key] = ema_list[ema_idx]
            ema_idx += 1
        else:
            dic_ema[clean_key] = model_state[key]

    weight_type = "ema" if ema_idx > 0 else "raw"
    print(f"  EMA weights mapped: {ema_idx}/{len(ema_list)} params, "
          f"{len(dic_ema) - ema_idx} buffers from raw checkpoint")
    return dic_ema, weight_type


# ============================================================
# Main Adapter
# ============================================================

class OfficialCQTDiffHybridDecoder(nn.Module):
    """
    Adapter di atas official CQTdiff U-Net.

    Dua mode inferensi:
    - Baseline (conditioning=None): multi-step reverse diffusion sampling
      menggunakan pipeline resmi CQT-Diff (T langkah Heun sampler)
    - Hybrid (conditioning!=None): reconstruction via learned STFT head

    Training (hybrid path): get_features -> FiLM -> decode_features tetap
    pakai STFT reconstruction head.
    """

    DIFFUSION_STEPS = int(os.environ.get("CQTDIFF_DIFFUSION_STEPS", "35"))
    DIFFUSION_XI = float(os.environ.get("CQTDIFF_DIFFUSION_XI", "0"))
    DIFFUSION_SIGMA_MIN = float(os.environ.get("CQTDIFF_SIGMA_MIN", "1e-4"))
    DIFFUSION_SIGMA_MAX = float(os.environ.get("CQTDIFF_SIGMA_MAX", "1.0"))

    def __init__(self, device, target_sr, segment_samples, gap_durations_ms, cqt_diff_dir):
        super().__init__()
        self.device_ref = torch.device(device)
        self.target_sr = int(target_sr)
        self.target_len = int(segment_samples)
        self.gap_durations_ms = list(gap_durations_ms)
        self.cqt_diff_dir = os.path.abspath(cqt_diff_dir)

        with _prepend_path(self.cqt_diff_dir):
            from src.models.unet_cqt import Unet_CQT

            self.args = _load_cqtdiff_config(self.cqt_diff_dir, self.device_ref)
            self.native_sr = int(self.args.sample_rate)
            self.native_len = int(self.args.audio_len)
            self.backbone = Unet_CQT(self.args, self.device_ref).to(self.device_ref)

        # ---- Load EMA weights (lebih baik daripada raw training weights) ----
        weights_path = _find_cqtdiff_weights(self.cqt_diff_dir)
        if weights_path is None:
            raise FileNotFoundError(
                "Checkpoint CQT-Diff+ asli belum ditemukan. Jalankan "
                "external/CQTdiff/download_weights_and_examples.sh atau set CQTDIFF_WEIGHTS."
            )

        ema_state, weight_type = _load_ema_weights_from_checkpoint(
            weights_path, self.device_ref, live_model=self.backbone
        )
        msg = self.backbone.load_state_dict(ema_state, strict=False)
        print(f"CQT-Diff+ {weight_type} weights loaded: {weights_path}")
        print(f"CQT-Diff+ load_state_dict: {msg}")

        # Freeze backbone by default
        train_backbone = os.environ.get("CQTDIFF_TRAIN_BACKBONE", "0").lower() in {"1", "true", "yes"}
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        trainable_backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_backbone_params = sum(p.numel() for p in self.backbone.parameters())
        print(
            "CQT-Diff+ backbone training: "
            f"{'enabled' if train_backbone else 'disabled'} "
            f"({trainable_backbone_params:,}/{total_backbone_params:,} trainable params)"
        )

        # ---- Diffusion sampling parameters (sesuai scripts/sampling_inpainting.sh) ----
        self.diff_params = DiffusionParams(
            sigma_data=0.057,
            sigma_min=self.DIFFUSION_SIGMA_MIN,
            sigma_max=self.DIFFUSION_SIGMA_MAX,
            ro=13,
            Schurn=5,
            Snoise=1.0,
            Stmin=0,
            Stmax=50,
        )
        print(
            f"Diffusion sampling config: T={self.DIFFUSION_STEPS}, "
            f"sigma=[{self.DIFFUSION_SIGMA_MIN}, {self.DIFFUSION_SIGMA_MAX}], "
            f"xi={self.DIFFUSION_XI}"
        )

        # ---- Low-pass filter buat buang artifact Nyquist setelah diffusion ----
        lpf_coeffs = scipy.signal.firwin(
            numtaps=100, cutoff=10000, width=1, window="kaiser", fs=self.native_sr
        )
        self.register_buffer(
            "_lpf_kernel",
            torch.FloatTensor(lpf_coeffs).unsqueeze(0).unsqueeze(0),
        )

        # ---- STFT reconstruction head (tetap dipakai buat hybrid training) ----
        self.n_fft = int(os.environ.get("CQTDIFF_ADAPTER_N_FFT", 2048))
        self.hop_length = int(os.environ.get("CQTDIFF_ADAPTER_HOP", 512))
        self.feature_dim = int(os.environ.get("CQTDIFF_ADAPTER_FEATURE_DIM", 256))
        self.freq_bins = self.n_fft // 2 + 1
        self.sigma = float(os.environ.get("CQTDIFF_ADAPTER_SIGMA", "0.1"))

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

    # ================================================================
    # Utility: resample + pad/crop
    # ================================================================

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

    # ================================================================
    # Diffusion sampling loop (Heun 2nd-order, sesuai src/sampler.py)
    # ================================================================

    def _run_diffusion_sampling(self, y, mask, T=None, show_progress=True):
        """
        Multi-step reverse diffusion inpainting (2nd-order Heun sampler).

        y    : (B, native_len) observed audio, zeros di area gap
        mask : (B, native_len) float, 1=keep 0=gap (konvensi CQT-Diff)
        T    : jumlah langkah diffusion (default dari DIFFUSION_STEPS)
        """
        if T is None:
            T = self.DIFFUSION_STEPS
        device = y.device
        shape = y.shape

        t = self.diff_params.create_schedule(T).to(device)
        x = self.diff_params.sample_prior(shape, t[0]).to(device)
        gamma = self.diff_params.get_gamma(t).to(device)

        self.backbone.eval()
        with torch.no_grad():
            for i in tqdm(range(T), desc="Diffusion inpainting", leave=False, disable=not show_progress):
                # Stochastic injection (Langevin-style)
                if gamma[i] == 0:
                    t_hat = t[i]
                    x_hat = x
                else:
                    t_hat = t[i] + gamma[i] * t[i]
                    epsilon = torch.randn(shape, device=device) * self.diff_params.Snoise
                    x_hat = x + ((t_hat ** 2 - t[i] ** 2) ** 0.5) * epsilon

                # Denoise + data consistency (replacement method)
                # mask=1 → observed (pakai y), mask=0 → gap (pakai prediksi denoiser)
                denoised = self.diff_params.denoiser(x_hat, self.backbone, t_hat.unsqueeze(-1))
                denoised = mask * y + (1.0 - mask) * denoised

                score = (denoised - x_hat) / t_hat ** 2
                d = -t_hat * score
                h = t[i + 1] - t_hat

                if t[i + 1] != 0:
                    # 2nd order Heun correction
                    x_prime = x_hat + h * d
                    denoised_prime = self.diff_params.denoiser(
                        x_prime, self.backbone, t[i + 1].unsqueeze(-1)
                    )
                    denoised_prime = mask * y + (1.0 - mask) * denoised_prime

                    score_prime = (denoised_prime - x_prime) / t[i + 1] ** 2
                    d_prime = -t[i + 1] * score_prime
                    x = x_hat + h * (0.5 * d + 0.5 * d_prime)
                else:
                    # Last step: 1st order Euler
                    x = x_hat + h * d

        return x.detach()

    def _apply_lowpass(self, x):
        """FIR low-pass filter buat buang artifact Nyquist setelah diffusion."""
        return F.conv1d(
            x.unsqueeze(1),
            self._lpf_kernel.to(x.device),
            padding="same",
        ).squeeze(1)

    # ================================================================
    # Inpainting: baseline (diffusion) vs hybrid (reconstruction)
    # ================================================================

    def _inpaint_diffusion(self, masked_audio, mask, T=None, show_progress=True):
        """
        Baseline inpainting: multi-step reverse diffusion.

        masked_audio : (B, target_len) audio di target_sr, zeros di gap
        mask         : (B, target_len) bool, True=gap
        """
        B = masked_audio.shape[0]
        target_len = masked_audio.shape[-1]

        # Cari batas gap di domain asli (target_sr)
        mask_bool = mask[0].bool()
        gap_indices = torch.where(mask_bool)[0]
        if len(gap_indices) == 0:
            out = masked_audio.detach().cpu().numpy()
            return out[0] if B == 1 else out

        gap_start_orig = gap_indices[0].item()
        gap_end_orig = gap_indices[-1].item() + 1

        # Resample ke native SR
        if self.target_sr != self.native_sr:
            audio_native = torchaudio.functional.resample(
                masked_audio, self.target_sr, self.native_sr
            )
        else:
            audio_native = masked_audio.clone()
        nat_len = audio_native.shape[-1]

        # Center crop ke native_len (model expectation)
        if nat_len > self.native_len:
            crop_offset = (nat_len - self.native_len) // 2
            audio_crop = audio_native[..., crop_offset : crop_offset + self.native_len]
        elif nat_len < self.native_len:
            crop_offset = 0
            audio_crop = F.pad(audio_native, (0, self.native_len - nat_len))
        else:
            crop_offset = 0
            audio_crop = audio_native

        # Map gap boundaries ke domain native yang sudah di-crop
        sr_scale = self.native_sr / self.target_sr
        gap_start_nat = int(round(gap_start_orig * sr_scale)) - crop_offset
        gap_end_nat = int(round(gap_end_orig * sr_scale)) - crop_offset
        gap_start_nat = max(0, gap_start_nat)
        gap_end_nat = min(self.native_len, gap_end_nat)

        # Buat mask CQT-Diff: 1=keep, 0=gap (kebalikan dari user convention)
        cqtdiff_mask = torch.ones(
            (B, self.native_len), device=masked_audio.device, dtype=torch.float32
        )
        cqtdiff_mask[..., gap_start_nat:gap_end_nat] = 0.0

        # y = observed audio (gap sudah nol)
        y = cqtdiff_mask * audio_crop

        # Jalankan multi-step reverse diffusion
        x_hat = self._run_diffusion_sampling(y, cqtdiff_mask, T=T, show_progress=show_progress)
        x_hat = self._apply_lowpass(x_hat)

        # Ambil konten gap dari hasil diffusion
        gap_native = x_hat[..., gap_start_nat:gap_end_nat].clone()

        # Resample gap content balik ke target SR
        if self.target_sr != self.native_sr:
            gap_target = torchaudio.functional.resample(
                gap_native, self.native_sr, self.target_sr
            )
        else:
            gap_target = gap_native

        # Sesuaikan panjang ke ukuran gap asli
        gap_len_orig = gap_end_orig - gap_start_orig
        if gap_target.shape[-1] > gap_len_orig:
            gap_target = gap_target[..., :gap_len_orig]
        elif gap_target.shape[-1] < gap_len_orig:
            gap_target = F.pad(gap_target, (0, gap_len_orig - gap_target.shape[-1]))

        # Taruh gap content ke audio asli
        output = masked_audio.clone()
        output[..., gap_start_orig:gap_end_orig] = gap_target

        out = output.detach().cpu().numpy()
        return out[0] if B == 1 else out

    def _inpaint_reconstruction(self, masked_audio, mask, conditioning,
                                diffusion_audio=None):
        """
        Hybrid inpainting: two-stage (diffusion + FiLM refinement).

        Stage 1: Multi-step diffusion (sama dgn baseline) → initial recon
                 Bisa di-skip kalau diffusion_audio sudah disediakan.
        Stage 2: FiLM-conditioned spec_decoder → iSTFT → refined gap
                 Blend 50/50 dengan diffusion output.

        conditioning: (B, T_frames, feature_dim) — sudah di-FiLM-kan di luar
        diffusion_audio: (B, T) tensor — opsional, skip stage 1 kalau ada
        """
        # Stage 1: Diffusion inpainting
        if diffusion_audio is not None:
            diffusion_tensor = diffusion_audio.to(self.device_ref)
        else:
            diffusion_result = self._inpaint_diffusion(masked_audio, mask)
            if isinstance(diffusion_result, np.ndarray):
                diffusion_tensor = torch.from_numpy(diffusion_result).float()
                if diffusion_tensor.dim() == 1:
                    diffusion_tensor = diffusion_tensor.unsqueeze(0)
                diffusion_tensor = diffusion_tensor.to(self.device_ref)
            else:
                diffusion_tensor = diffusion_result

        # Stage 2: Refine gap pakai FiLM-conditioned spectrogram decoder
        conditioning = conditioning.to(self.device_ref, dtype=torch.float32)
        if conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1)

        pred_spec_features = self.decode_features(conditioning).permute(0, 2, 1).contiguous()

        device_type = self.device_ref.type
        with torch.autocast(device_type=device_type, enabled=False):
            bsz, _, n_frames = pred_spec_features.shape
            pred_pairs = (
                pred_spec_features.float()
                .reshape(bsz, self.freq_bins, 2, n_frames)
                .permute(0, 1, 3, 2)
                .contiguous()
            )
            recon_spec = torch.view_as_complex(pred_pairs)
            window = torch.hann_window(self.n_fft, device=masked_audio.device)
            refined = torch.istft(
                recon_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                length=masked_audio.shape[-1],
            )

        # Blend: gap pakai rata-rata diffusion + refinement, observed pakai asli
        gap_blend = 0.5 * diffusion_tensor + 0.5 * refined
        output = torch.where(mask.bool(), gap_blend, masked_audio)
        out = output.detach().cpu().numpy()
        return out[0] if output.shape[0] == 1 else out

    def inpaint(self, masked_audio, mask, conditioning=None, diffusion_audio=None):
        """
        Entry point inpainting.

        conditioning=None  -> baseline: multi-step reverse diffusion
        conditioning!=None -> hybrid: diffusion + FiLM refinement (two-stage)
        diffusion_audio    -> opsional, skip stage 1 diffusion kalau sudah ada
        """
        if masked_audio.dim() == 1:
            masked_audio = masked_audio.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        masked_audio = masked_audio.to(self.device_ref, dtype=torch.float32)
        mask = mask.to(self.device_ref)

        if conditioning is not None:
            return self._inpaint_reconstruction(
                masked_audio, mask, conditioning, diffusion_audio=diffusion_audio
            )
        return self._inpaint_diffusion(masked_audio, mask)

    # ================================================================
    # Feature extraction (tetap buat hybrid FiLM training)
    # ================================================================

    def _backbone_predict(self, audio):
        """Single-pass backbone prediction (buat feature extraction, bukan inpainting)."""
        device_type = self.device_ref.type
        with torch.autocast(device_type=device_type, enabled=False):
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
        """Extract STFT-based features (buat hybrid training dengan FiLM)."""
        device_type = self.device_ref.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = x.squeeze(1) if x.dim() == 3 else x
            x = x.float()
            backbone_pred = self._backbone_predict(x)
            if mask is not None:
                base = torch.where(mask.bool(), backbone_pred, x)
            else:
                base = backbone_pred

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


def build_cqtdiff_decoder(device, target_sr, segment_samples, gap_durations_ms, cqt_diff_dir):
    return OfficialCQTDiffHybridDecoder(
        device=device,
        target_sr=target_sr,
        segment_samples=segment_samples,
        gap_durations_ms=gap_durations_ms,
        cqt_diff_dir=cqt_diff_dir,
    ).to(device)
