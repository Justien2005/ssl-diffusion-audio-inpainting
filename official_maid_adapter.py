import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak bisa memuat module {module_name} dari {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_repo_dir(explicit_dir=None):
    candidates = [
        explicit_dir,
        os.environ.get("MIDI2PERFORMANCE_DIR"),
        os.environ.get("DDPM_MIDI2PERFORMANCE_DIR"),
        Path(__file__).resolve().parent / "external" / "DDPM-Midi2Performance-Model",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        repo_dir = Path(candidate).expanduser().resolve()
        if (repo_dir / "main" / "models" / "diffusion" / "unet_openai.py").exists():
            return repo_dir
    raise FileNotFoundError(
        "Repo DDPM-Midi2Performance-Model tidak ditemukan. Set MIDI2PERFORMANCE_DIR "
        "atau letakkan repo di external/DDPM-Midi2Performance-Model."
    )


def _parse_channel_mult(value):
    if isinstance(value, str):
        return tuple(int(v) for v in value.split(",") if v)
    return tuple(value)


class DDPMMidi2PerformanceDecoder(nn.Module):
    """
    Adapter MAID untuk pipeline audio inpainting.

    Backbone denoiser dan DDPM scheduler diambil dari repository
    DDPM-Midi2Performance-Model. Kode lokal hanya menjembatani audio waveform,
    FiLM SSL conditioning, dan interface training/evaluasi pipeline ini.
    """

    def __init__(
        self,
        device,
        repo_dir=None,
        target_sr=44100,
        n_mels=128,
        feature_dim=512,
        n_fft=2048,
        hop_length=512,
        model_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0.0,
        beta_1=1e-4,
        beta_2=0.02,
        n_timesteps=1000,
        checkpoint_path=None,
    ):
        super().__init__()
        self.repo_dir = _find_repo_dir(repo_dir)
        self.target_sr = int(target_sr)
        self.n_mels = int(n_mels)
        self.feature_dim = int(feature_dim)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_timesteps = int(n_timesteps)

        diffusion_dir = self.repo_dir / "main" / "models" / "diffusion"
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            unet_mod = _load_module("ddpm_m2p_unet_openai", diffusion_dir / "unet_openai.py")
            ddpm_mod = _load_module("ddpm_m2p_ddpm", diffusion_dir / "ddpm.py")
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        backbone = unet_mod.SuperResModel(
            in_channels=1,
            model_channels=int(model_channels),
            out_channels=1,
            num_res_blocks=int(num_res_blocks),
            channel_mult=_parse_channel_mult(channel_mult),
            use_checkpoint=False,
            dropout=float(dropout),
            dims=2,
        )
        self.diffusion = ddpm_mod.DDPM(
            backbone,
            beta_1=float(beta_1),
            beta_2=float(beta_2),
            T=self.n_timesteps,
            var_type="fixedsmall",
        )

        self.feature_pool = nn.Sequential(
            nn.Linear(self.n_mels, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
        )
        self.condition_to_mel = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.n_mels),
        )
        self.register_buffer("_hann_window", torch.empty(0), persistent=False)
        self.register_buffer("_mel_basis", torch.empty(0), persistent=False)

        self._load_pretrained_if_available(checkpoint_path)
        self.to(device)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def decoder(self):
        return self.diffusion.decoder

    def _get_hann_window(self, device):
        if self._hann_window.numel() != self.n_fft or self._hann_window.device != device:
            self._hann_window = torch.hann_window(self.n_fft, device=device)
        return self._hann_window

    def _get_mel_basis(self, sr, device):
        expected_shape = (self.n_mels, self.n_fft // 2 + 1)
        if self._mel_basis.shape != expected_shape or self._mel_basis.device != device:
            self._mel_basis = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=0.0,
                f_max=sr / 2,
                n_mels=self.n_mels,
                sample_rate=sr,
                norm="slaney",
                mel_scale="slaney",
            ).to(device=device, dtype=torch.float32).T.contiguous()
        return self._mel_basis

    def _candidate_checkpoints(self, checkpoint_path):
        candidates = [
            checkpoint_path,
            os.environ.get("MAID_M2P_CHECKPOINT"),
            os.environ.get("DDPM_M2P_CHECKPOINT"),
        ]
        models_dir = self.repo_dir / "Models"
        if models_dir.exists():
            for pattern in ("*.ckpt", "*.pt", "*.pth"):
                candidates.extend(str(path) for path in sorted(models_dir.glob(pattern)))
        return [Path(path).expanduser().resolve() for path in candidates if path]

    def _load_pretrained_if_available(self, checkpoint_path=None):
        ckpt = next((path for path in self._candidate_checkpoints(checkpoint_path) if path.exists()), None)
        if ckpt is None:
            msg = (
                "DDPM-Midi2Performance checkpoint tidak ditemukan. "
                "Set MAID_M2P_CHECKPOINT/DDPM_M2P_CHECKPOINT ke checkpoint resmi, "
                "atau set ALLOW_RANDOM_MAID=1 hanya untuk smoke test non-paper."
            )
            allow_random = os.environ.get("ALLOW_RANDOM_MAID", "0").lower() in {"1", "true", "yes"}
            if allow_random:
                print(f"WARNING: {msg} MAID memakai arsitektur asli dan dilatih dari awal.")
                return
            raise FileNotFoundError(msg)
            return

        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError(f"Format checkpoint MAID tidak dikenali: {ckpt}")

        prefixes = [
            "target_network.decoder.",
            "online_network.decoder.",
            "diffusion.decoder.",
            "decoder.",
            "model.",
        ]
        stripped = {}
        for key, value in state.items():
            name = str(key)
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            stripped[name] = value

        decoder_state = self.decoder.state_dict()
        matched_keys = [
            name for name, value in stripped.items()
            if name in decoder_state and tuple(decoder_state[name].shape) == tuple(value.shape)
        ]
        if not matched_keys:
            raise RuntimeError(
                f"Checkpoint MAID tidak memiliki key yang cocok dengan decoder saat ini: {ckpt}"
            )

        msg = self.decoder.load_state_dict(stripped, strict=False)
        print(f"DDPM-Midi2Performance pretrained checkpoint diload: {ckpt}")
        print(f"MAID matched pretrained keys: {len(matched_keys)} / {len(decoder_state)}")
        print(f"MAID backbone load_state_dict: {msg}")

    def audio_to_mel_batch(self, audio_batch, sr=None, return_ref=False):
        sr = int(sr or self.target_sr)
        if isinstance(audio_batch, torch.Tensor):
            audio = audio_batch.to(self.device, dtype=torch.float32)
        else:
            audio = torch.as_tensor(audio_batch, dtype=torch.float32, device=self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        with torch.autocast(device_type="cuda" if audio.device.type == "cuda" else "cpu", enabled=False):
            window = self._get_hann_window(audio.device)
            spec = torch.stft(
                audio.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            )
            power = spec.abs().pow(2)
            mel_basis = self._get_mel_basis(sr, audio.device)
            mel_power = torch.einsum("mf,bft->bmt", mel_basis, power).clamp_min(1e-10)
            mel_db_raw = 10.0 * torch.log10(mel_power)
            ref_db = mel_db_raw.amax(dim=(1, 2), keepdim=True)
            mel_db = torch.clamp(mel_db_raw - ref_db, min=-80.0)
            mel_norm = torch.clamp((mel_db + 40.0) / 40.0, min=-1.0, max=1.0)
        if return_ref:
            return mel_db, mel_norm, ref_db
        return mel_db, mel_norm

    def mask_to_frame_mask(self, mask_batch, frame_count):
        if isinstance(mask_batch, torch.Tensor):
            mask_t = mask_batch.to(self.device).bool()
        else:
            mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
        pooled = F.max_pool1d(
            mask_t.float().unsqueeze(1),
            kernel_size=self.hop_length,
            stride=self.hop_length,
            ceil_mode=True,
        ).squeeze(1)
        if pooled.shape[1] < frame_count:
            pooled = F.pad(pooled, (0, frame_count - pooled.shape[1]))
        elif pooled.shape[1] > frame_count:
            pooled = pooled[:, :frame_count]
        return pooled.bool()

    def get_features(self, x):
        # FIX 5: ganti global mean pooling → per-frame features (B, T, feature_dim).
        # Sebelumnya mel_norm.mean(dim=-1) menghilangkan seluruh info temporal,
        # sehingga FiLM memberi conditioning yang sama untuk semua frame.
        # Sekarang setiap frame punya representasinya sendiri agar model tahu
        # posisi dan konteks lokal di sekitar gap.
        _, mel_norm = self.audio_to_mel_batch(x)   # (B, n_mels, T)
        mel_norm_t = mel_norm.permute(0, 2, 1)      # (B, T, n_mels)
        return self.feature_pool(mel_norm_t)         # (B, T, feature_dim)

    def _conditioning_image(self, masked_mel_norm, conditioning=None):
        cond = masked_mel_norm.unsqueeze(1)  # (B, 1, mel, T)
        if conditioning is None:
            return cond
        # FIX 5 lanjutan: conditioning sekarang bisa (B, feature_dim) global
        # atau (B, T, feature_dim) per-frame. Keduanya ditangani di sini.
        if conditioning.dim() == 2:
            # global vector → broadcast ke semua frame
            bias = self.condition_to_mel(conditioning.float())          # (B, n_mels)
            bias = torch.tanh(bias).unsqueeze(-1).expand(-1, -1, masked_mel_norm.shape[-1])
        else:
            # per-frame (B, T, feature_dim) → per-frame bias
            bias = self.condition_to_mel(conditioning.float())          # (B, T, n_mels)
            bias = torch.tanh(bias).permute(0, 2, 1)                   # (B, n_mels, T)
        return torch.clamp(cond + 0.25 * bias.unsqueeze(1), min=-1.0, max=1.0)

    def _pad_frames_for_unet(self, mel_norm):
        frame_count = mel_norm.shape[-1]
        stride = 2 ** (len(self.decoder.channel_mult) - 1)
        pad_frames = (stride - frame_count % stride) % stride
        if pad_frames:
            mel_norm = F.pad(mel_norm, (0, pad_frames), value=-1.0)
        return mel_norm, frame_count

    def diffusion_loss(self, clean_audio, masked_audio, mask=None, conditioning=None):
        _, clean_mel_norm = self.audio_to_mel_batch(clean_audio)
        _, masked_mel_norm = self.audio_to_mel_batch(masked_audio)
        clean_mel_norm, _ = self._pad_frames_for_unet(clean_mel_norm)
        masked_mel_norm, _ = self._pad_frames_for_unet(masked_mel_norm)
        target = clean_mel_norm.unsqueeze(1)
        cond = self._conditioning_image(masked_mel_norm, conditioning=conditioning)

        t = torch.randint(0, self.n_timesteps, size=(target.size(0),), device=target.device)
        eps = torch.randn_like(target)
        eps_pred = self.diffusion(target, eps, t, y=None, cond=cond)

        full_loss = F.l1_loss(eps_pred, eps)
        if mask is None:
            gap_loss = full_loss
        else:
            frame_mask = self.mask_to_frame_mask(mask, target.shape[-1]).unsqueeze(1).unsqueeze(1)
            expanded_mask = frame_mask.expand_as(target)
            if expanded_mask.any():
                gap_loss = F.l1_loss(eps_pred[expanded_mask], eps[expanded_mask])
            else:
                gap_loss = full_loss
        loss = 0.7 * gap_loss + 0.3 * full_loss  # FIX 4: dari (gap + 0.1*full) → (0.7*gap + 0.3*full) agar model tidak mengabaikan konsistensi keseluruhan audio
        return {"loss": loss, "gap_loss": gap_loss, "full_loss": full_loss}

    def predict_mel_norm(self, masked_audio, conditioning=None):
        # FIX 2: ganti t=0 shortcut dengan proper DDPM sampling agar
        # konsisten dengan cara inpaint() bekerja saat inference.
        # t=0 sebelumnya menyebabkan training-inference mismatch.
        _, masked_mel_norm = self.audio_to_mel_batch(masked_audio)
        masked_mel_norm, frame_count = self._pad_frames_for_unet(masked_mel_norm)
        cond = self._conditioning_image(masked_mel_norm, conditioning=conditioning)
        n_steps = int(os.environ.get("MAID_M2P_INFERENCE_STEPS", 50))
        n_steps = max(1, min(n_steps, self.n_timesteps))
        x_t = torch.randn_like(cond)
        with torch.no_grad():
            samples = self.diffusion.sample(x_t, y=None, cond=cond, n_steps=n_steps, checkpoints=[n_steps])
        pred = torch.clamp(samples[str(n_steps)].squeeze(1), min=-1.0, max=1.0)[..., :frame_count]
        return pred.permute(0, 2, 1)

    def _mel_db_to_audio_tensor(self, mel_db_pred, target_len):
        if mel_db_pred.dim() == 2:
            mel_db_pred = mel_db_pred.unsqueeze(0)
        mel_power = torch.pow(10.0, mel_db_pred.float() / 10.0).clamp_min(1e-10)
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.target_sr,
            f_min=0.0,
            f_max=self.target_sr / 2,
            norm="slaney",
            mel_scale="slaney",
        ).to(mel_power.device)
        griffinlim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=128,  # FIX 1: dinaikkan dari 32 → 128 untuk kualitas phase reconstruction lebih baik
            power=1.0,
        ).to(mel_power.device)
        linear_power = inverse_mel(mel_power).clamp_min(1e-10)
        audio = griffinlim(linear_power.sqrt())
        if audio.shape[-1] < target_len:
            audio = F.pad(audio, (0, target_len - audio.shape[-1]))
        elif audio.shape[-1] > target_len:
            audio = audio[..., :target_len]
        return audio

    def inpaint(self, masked_audio, mask, conditioning=None, n_steps=50):
        if masked_audio.dim() == 1:
            masked_audio = masked_audio.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        masked_audio = masked_audio.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device).bool()

        _, masked_mel_norm, ref_db = self.audio_to_mel_batch(masked_audio, return_ref=True)
        masked_mel_padded, frame_count = self._pad_frames_for_unet(masked_mel_norm)
        cond = self._conditioning_image(masked_mel_padded, conditioning=conditioning)

        # FIX 3: gap-preserving replacement method.
        # Di setiap langkah DDPM sampling, paksa frame di luar gap kembali ke
        # nilai noisy dari audio asli. Ini mencegah model "mengubah" bagian audio
        # yang seharusnya tidak disentuh dan membuat boundary gap lebih natural.
        frame_mask_padded = self.mask_to_frame_mask(mask, masked_mel_padded.shape[-1]).unsqueeze(1)  # (B,1,T)
        known_mel = masked_mel_padded.unsqueeze(1)  # (B,1,mel,T)

        n_steps = int(os.environ.get("MAID_M2P_INFERENCE_STEPS", n_steps))
        n_steps = max(1, min(n_steps, self.n_timesteps))

        # Hitung alpha_bar untuk setiap timestep (dibutuhkan untuk noisy known frames)
        betas = self.diffusion.betas  # (T,)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0).to(self.device)  # (T,)

        x_t = torch.randn_like(cond)
        step_size = self.n_timesteps // n_steps
        timesteps = list(reversed(range(0, self.n_timesteps, step_size)))[:n_steps]

        for i, t_val in enumerate(timesteps):
            t_tensor = torch.full((x_t.shape[0],), t_val, device=self.device, dtype=torch.long)

            # Satu langkah denoising dari backbone
            with torch.no_grad():
                eps_pred = self.decoder(x_t, t_tensor, y=None, cond=cond)
            beta_t = betas[t_val]
            alpha_t = alphas[t_val]
            alpha_bar_t = alpha_bar[t_val]
            x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            if t_val > 0:
                noise = torch.randn_like(x_t)
                x_t = alpha_bar[t_val - step_size].sqrt() * x0_pred + (1 - alpha_bar[t_val - step_size]).sqrt() * noise
            else:
                x_t = x0_pred

            # Replacement: paksa frame non-gap kembali ke nilai noisy dari known mel
            if t_val > 0:
                t_prev = max(t_val - step_size, 0)
                ab_prev = alpha_bar[t_prev]
                known_noisy = ab_prev.sqrt() * known_mel + (1 - ab_prev).sqrt() * torch.randn_like(known_mel)
            else:
                known_noisy = known_mel
            x_t = torch.where(frame_mask_padded.unsqueeze(2).unsqueeze(2).expand_as(x_t), x_t, known_noisy)

        pred_mel_norm = torch.clamp(x_t.squeeze(1), min=-1.0, max=1.0)[..., :frame_count]
        output_mel_norm = torch.where(
            self.mask_to_frame_mask(mask, pred_mel_norm.shape[-1]).unsqueeze(1),
            pred_mel_norm,
            masked_mel_norm,
        )
        output_mel_db = output_mel_norm * 40.0 - 40.0 + ref_db
        reconstructed = self._mel_db_to_audio_tensor(output_mel_db, masked_audio.shape[-1])
        output = torch.where(mask, reconstructed, masked_audio)
        if output.shape[0] == 1:
            return output[0].detach().cpu().numpy()
        return output.detach().cpu().numpy()


def build_maid_decoder(
    device,
    target_sr,
    segment_samples,
    gap_durations_ms,
    midi2performance_dir=None,
    checkpoint_path=None,
):
    return DDPMMidi2PerformanceDecoder(
        device=device,
        repo_dir=midi2performance_dir,
        target_sr=target_sr,
        checkpoint_path=checkpoint_path,
    )
