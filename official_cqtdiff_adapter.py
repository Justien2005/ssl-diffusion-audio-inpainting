import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


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


def _patch_cqtdiff_numpy_clip(cqt_diff_dir):
    """Patch old CQTdiff NSGT code for NumPy 2.x casting rules."""
    patch_path = os.path.join(cqt_diff_dir, "src", "nsgt", "nsgfwin_sl.py")
    if not os.path.exists(patch_path):
        return

    with open(patch_path, "r", encoding="utf-8") as f:
        text = f.read()

    old = "    np.clip(M, min_win, np.inf, out=M)\n"
    new = "    M = np.clip(M.astype(float), min_win, np.inf).astype(int)\n"
    if old in text and new not in text:
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new))
        print(f"Patched CQTdiff NumPy clip compatibility: {patch_path}")


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


class OfficialCQTDiffHybridDecoder(nn.Module):
    """
    Adapter around the official CQTdiff U-Net.

    The official backbone is loaded from external/CQTdiff and can be frozen by
    default. A small trainable reconstruction head exposes the get_features /
    decode_features / inpaint interface used by the hybrid FiLM pipeline.
    """

    def __init__(self, device, target_sr, segment_samples, gap_durations_ms, cqt_diff_dir):
        super().__init__()
        self.device_ref = torch.device(device)
        self.target_sr = int(target_sr)
        self.target_len = int(segment_samples)
        self.gap_durations_ms = list(gap_durations_ms)
        self.cqt_diff_dir = os.path.abspath(cqt_diff_dir)
        _patch_cqtdiff_numpy_clip(self.cqt_diff_dir)

        with _prepend_path(self.cqt_diff_dir):
            from src.models.unet_cqt import Unet_CQT

            self.args = _load_cqtdiff_config(self.cqt_diff_dir, self.device_ref)
            self.native_sr = int(self.args.sample_rate)
            self.native_len = int(self.args.audio_len)
            self.backbone = Unet_CQT(self.args, self.device_ref).to(self.device_ref)

        weights_path = _find_cqtdiff_weights(self.cqt_diff_dir)
        if weights_path is None:
            raise FileNotFoundError(
                "Checkpoint CQT-Diff+ asli belum ditemukan. Jalankan "
                "external/CQTdiff/download_weights_and_examples.sh atau set CQTDIFF_WEIGHTS."
            )

        payload = torch.load(weights_path, map_location=self.device_ref, weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError(f"Format checkpoint CQTdiff tidak dikenali: {weights_path}")
        msg = self.backbone.load_state_dict(_strip_module_prefix(state), strict=False)
        print(f"CQT-Diff+ original weights loaded: {weights_path}")
        print(f"CQT-Diff+ load_state_dict: {msg}")

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

    def _pad_or_crop(self, x, length):
        if x.shape[-1] < length:
            return F.pad(x, (0, length - x.shape[-1]))
        if x.shape[-1] > length:
            start = (x.shape[-1] - length) // 2
            return x[..., start:start + length]
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

    def _backbone_predict(self, audio):
        device_type = self.device_ref.type
        with torch.autocast(device_type=device_type, enabled=False):
            native = self._to_native(audio).float()
            sigma = torch.full((native.shape[0], 1), self.sigma, device=native.device, dtype=torch.float32)
            if any(param.requires_grad for param in self.backbone.parameters()):
                pred = self.backbone(native, sigma)
            else:
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
                frame_mask = self._sample_to_frame_mask(mask, spec_features.shape[1]).to(spec_features.dtype)
            return self.feature_encoder(torch.cat([spec_features, frame_mask.unsqueeze(-1)], dim=-1).float())

    def decode_features(self, features):
        return self.spec_decoder(features)

    def forward(self, x, mask=None, conditioning=None):
        features = self.get_features(x, mask)
        if conditioning is not None:
            if conditioning.dim() == 2:
                conditioning = conditioning.unsqueeze(1)
            features = features + conditioning
        return self.decode_features(features)

    def inpaint(self, masked_audio, mask, conditioning=None):
        if masked_audio.dim() == 1:
            masked_audio = masked_audio.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        masked_audio = masked_audio.to(self.device_ref, dtype=torch.float32)
        mask = mask.to(self.device_ref).bool()

        features = self.get_features(masked_audio.float(), mask)
        if conditioning is not None:
            conditioning = conditioning.to(self.device_ref, dtype=features.dtype)
            if conditioning.dim() == 2:
                conditioning = conditioning.unsqueeze(1)
            features = features + conditioning
        pred_spec_features = self.decode_features(features).permute(0, 2, 1).contiguous()

        device_type = self.device_ref.type
        with torch.autocast(device_type=device_type, enabled=False):
            bsz, _, n_frames = pred_spec_features.shape
            pred_pairs = pred_spec_features.float().reshape(
                bsz, self.freq_bins, 2, n_frames
            ).permute(0, 1, 3, 2).contiguous()
            recon_spec = torch.view_as_complex(pred_pairs)
            window = torch.hann_window(self.n_fft, device=masked_audio.device)
            reconstructed = torch.istft(
                recon_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                length=masked_audio.shape[-1],
            )

        output = torch.where(mask, reconstructed, masked_audio)
        if output.shape[0] == 1:
            return output[0].detach().cpu().numpy()
        return output.detach().cpu().numpy()


def build_cqtdiff_decoder(device, target_sr, segment_samples, gap_durations_ms, cqt_diff_dir):
    return OfficialCQTDiffHybridDecoder(
        device=device,
        target_sr=target_sr,
        segment_samples=segment_samples,
        gap_durations_ms=gap_durations_ms,
        cqt_diff_dir=cqt_diff_dir,
    ).to(device)
