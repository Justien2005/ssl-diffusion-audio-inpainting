import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MAIDProxyDecoder(nn.Module):
    """
    MAID-inspired proxy decoder.

    The user explicitly allows MAID to remain a proxy while CQT-Diff+ and
    AudioMAE use their original repos. This module keeps the same interface as
    the final pipeline expects for training and evaluation.
    """

    def __init__(self, device, target_sr=44100, n_mels=128, feature_dim=512, n_fft=2048, hop_length=512):
        super().__init__()
        self.target_sr = int(target_sr)
        self.n_mels = int(n_mels)
        self.feature_dim = int(feature_dim)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(self.n_mels, 512), nn.SiLU()),
            nn.Sequential(nn.Linear(512, 512), nn.SiLU()),
            nn.Sequential(nn.Linear(512, self.feature_dim), nn.SiLU()),
        ])
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(self.feature_dim, 512), nn.SiLU()),
            nn.Sequential(nn.Linear(512, 512), nn.SiLU()),
            nn.Sequential(nn.Linear(512, self.n_mels)),
        ])
        self.feature_pool = nn.Sequential(nn.Linear(self.n_mels, self.feature_dim), nn.SiLU())
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def audio_to_mel_batch(self, audio_batch, sr=None):
        sr = int(sr or self.target_sr)
        if isinstance(audio_batch, torch.Tensor):
            audio = audio_batch.to(self.device, dtype=torch.float32)
        else:
            audio = torch.as_tensor(audio_batch, dtype=torch.float32, device=self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        power = spec.abs().pow(2)
        mel_basis = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=0.0,
            f_max=sr / 2,
            n_mels=self.n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        ).to(audio.device, dtype=audio.dtype).T
        mel_power = torch.einsum("mf,bft->bmt", mel_basis, power).clamp_min(1e-10)
        mel_db_raw = 10.0 * torch.log10(mel_power)
        ref_db = mel_db_raw.amax(dim=(1, 2), keepdim=True)
        mel_db = torch.clamp(mel_db_raw - ref_db, min=-80.0)
        mel_norm = (mel_db + 40.0) / 40.0
        return mel_db, mel_norm

    def mask_to_frame_mask(self, mask_batch, frame_count):
        mask_t = mask_batch.to(self.device).bool() if isinstance(mask_batch, torch.Tensor) else torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)
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
        _, mel_norm = self.audio_to_mel_batch(x)
        pooled = mel_norm.mean(dim=-1)
        return self.feature_pool(pooled)

    def predict_mel_norm(self, masked_audio, conditioning=None):
        _, mel_norm = self.audio_to_mel_batch(masked_audio)
        x = mel_norm.permute(0, 2, 1)
        for block in self.encoder_blocks:
            x = block(x)
        if conditioning is not None:
            x = x + conditioning.unsqueeze(1)
        for block in self.decoder_blocks:
            x = block(x)
        return x

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
            n_iter=16,
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
        base_mel_db, _ = self.audio_to_mel_batch(masked_audio)
        pred_mel_norm = self.predict_mel_norm(masked_audio, conditioning=conditioning)
        pred_mel_db = pred_mel_norm.permute(0, 2, 1) * 40.0 - 40.0
        frame_mask = self.mask_to_frame_mask(mask, pred_mel_norm.shape[1]).unsqueeze(1)
        output_mel_db = torch.where(frame_mask, pred_mel_db, base_mel_db)
        reconstructed = self._mel_db_to_audio_tensor(output_mel_db, masked_audio.shape[-1])
        output = torch.where(mask, reconstructed, masked_audio)
        if output.shape[0] == 1:
            return output[0].detach().cpu().numpy()
        return output.detach().cpu().numpy()


def build_maid_decoder(device, target_sr, segment_samples, gap_durations_ms):
    return MAIDProxyDecoder(device=device, target_sr=target_sr)
