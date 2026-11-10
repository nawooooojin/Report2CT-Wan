"""Adapters for MAISI, MedSynV2, Wan 2.1 and WF-VAE-L on the 448^3 grid."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from vae_compare.data import drop_z, pad_z

VAES = ("maisi", "medsyn2", "wan2.1", "wf-vae")
HU_CLIP = 1000.0
MAISI_AE = {"norm_float16": True, "num_splits": 2, "dim_split": 1}
MAISI_SW_ROI = (80, 80, 80)
MAISI_SW_OVERLAP = 0.4
MEDSYN2_HU = (-1024.0, 600.0)
MEDSYN2_CHUNK = 80
MEDSYN2_SW_ROI = (16, 32, 32)
MEDSYN2_SW_OVERLAP = 0.1
WAN_DEFAULT = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class Vae:
    """A VAE with prep, encode and decode functions."""

    name: str
    prep: Callable
    enc: Callable
    dec: Callable


def build_maisi(config: str, ckpt: str, device: torch.device) -> Vae:
    """Build the MAISI autoencoder."""
    from monai.bundle import ConfigParser
    from monai.inferers.inferer import SlidingWindowInferer
    from monai.transforms import ScaleIntensityRanged

    parser = ConfigParser()
    parser.read_config(config)
    parser["autoencoder_def"].update(MAISI_AE)
    ae = parser.get_parsed_content("autoencoder_def")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    ae.load_state_dict(sd)
    ae = ae.requires_grad_(False).eval().to(device)
    scale = ScaleIntensityRanged(keys="image", a_min=-HU_CLIP, a_max=HU_CLIP, b_min=0.0, b_max=1.0, clip=True)

    def prep(x_hu: np.ndarray) -> torch.Tensor:
        x01 = scale({"image": x_hu[None]})["image"].numpy()[0]
        return torch.from_numpy(np.ascontiguousarray(x01))[None, None].to(device).half()

    def enc(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device.type, dtype=torch.float16):
            z_mu = ae.encode(x)[0]
        return z_mu.float()

    def dec(z: torch.Tensor) -> np.ndarray:
        inferer = SlidingWindowInferer(roi_size=list(MAISI_SW_ROI), sw_batch_size=1, progress=False, mode="gaussian",
                                       overlap=MAISI_SW_OVERLAP, sw_device=device, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = inferer(network=ae.decode_stage_2_outputs, inputs=z.float())
        vol = out[0, 0].float().cpu().numpy()
        hu = np.clip(vol * 2 * HU_CLIP - HU_CLIP, -HU_CLIP, HU_CLIP)
        return np.ascontiguousarray(hu.transpose(2, 0, 1)).astype(np.float32)

    return Vae("maisi", prep, enc, dec)


def build_medsyn2(repo: str, ckpt: str, device: torch.device) -> Vae:
    """Build the MedSynV2 VAE from its repository."""
    from monai.inferers.inferer import SlidingWindowInferer

    sys.path.insert(0, str(Path(repo) / "opensora" / "osp"))
    from opensora.models.ae.videobase import CausalVAEModel

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = CausalVAEModel(**{**(ck.get("hyper_parameters") or {}), "loss_type": "torch.nn.Identity"})
    model.load_state_dict({k: v for k, v in ck["state_dict"].items() if not k.startswith("loss.")}, strict=True)
    model = model.to(device).half().requires_grad_(False).eval()
    lo, hi = MEDSYN2_HU
    inferer = SlidingWindowInferer(roi_size=list(MEDSYN2_SW_ROI), sw_batch_size=1, mode="gaussian",
                                   overlap=MEDSYN2_SW_OVERLAP, sw_device=device, device=torch.device("cpu"))

    def prep(x_hu: np.ndarray) -> torch.Tensor:
        u = ((np.clip(x_hu, lo, hi) - lo) / (hi - lo) * 2 - 1).astype(np.float32)
        u = np.ascontiguousarray(u[::-1].transpose(2, 1, 0))
        return torch.from_numpy(u).half()[None, None].repeat(1, 3, 1, 1, 1).to(device)

    def enc(x: torch.Tensor) -> torch.Tensor:
        zs = [model.encode(x[:, :, k:k + MEDSYN2_CHUNK]).mode() for k in range(0, x.shape[2], MEDSYN2_CHUNK)]
        return torch.cat(zs, dim=2)

    def dec(z: torch.Tensor) -> np.ndarray:
        y = inferer(network=model.decode, inputs=z.half())
        y = y[0, 0].float().clamp(-1, 1).numpy()
        hu = np.clip((y + 1) / 2 * (hi - lo) + lo, -HU_CLIP, HU_CLIP)
        ras = hu.transpose(2, 1, 0)[::-1]
        return np.ascontiguousarray(ras.transpose(2, 0, 1)).astype(np.float32)

    return Vae("medsyn2", prep, enc, dec)


def build_wan21(path: str, device: torch.device) -> Vae:
    """Build the Wan 2.1 VAE from diffusers."""
    from diffusers import AutoencoderKLWan

    kw = {} if (Path(path) / "config.json").is_file() else {"subfolder": "vae"}
    vae = AutoencoderKLWan.from_pretrained(path, torch_dtype=torch.float32, **kw)
    vae.requires_grad_(False).eval().to(device)
    c = len(vae.config.latents_mean)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(1, c, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device).view(1, c, 1, 1, 1)

    def prep(x_hu: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(pad_z(x_hu).transpose(2, 0, 1)).astype(np.float32))

    def enc(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x.to(device, torch.float32), -HU_CLIP, HU_CLIP) / HU_CLIP
        x = x[:, None].repeat(1, 3, 1, 1).permute(1, 0, 2, 3)[None]
        z = vae.encode(x).latent_dist.mode()
        return ((z - mean) / std).cpu()

    def dec(z: torch.Tensor) -> np.ndarray:
        z = z.to(device, torch.float32) * std + mean
        y = vae.decode(z).sample[0].mean(0)
        return drop_z((torch.clamp(y, -1.0, 1.0) * HU_CLIP).cpu().float().numpy())

    return Vae("wan2.1", prep, enc, dec)


def build_wfvae(repo: str, ckpt_dir: str, device: torch.device) -> Vae:
    """Build WF-VAE-L from its repository."""
    sys.path.insert(0, str(repo))
    from causalvideovae.model import ModelRegistry

    model = ModelRegistry.get_model("WFVAE").from_pretrained(str(ckpt_dir))
    model = model.to(torch.bfloat16).requires_grad_(False).eval().to(device)
    model.enable_tiling()

    def prep(x_hu: np.ndarray) -> torch.Tensor:
        thw = np.ascontiguousarray(pad_z(x_hu).transpose(2, 0, 1)).astype(np.float32)
        x = (torch.from_numpy(thw).clamp(-HU_CLIP, HU_CLIP) / HU_CLIP)[:, None].repeat(1, 3, 1, 1)
        return x.permute(1, 0, 2, 3)[None].contiguous().to(device=device, dtype=torch.bfloat16)

    def enc(x: torch.Tensor) -> torch.Tensor:
        return model.encode(x).latent_dist.mode()

    def dec(z: torch.Tensor) -> np.ndarray:
        y = model.decode(z).sample.float()
        y = y[0] if y.dim() == 5 else y
        return drop_z((y.mean(0).clamp(-1, 1) * HU_CLIP).cpu().numpy())

    return Vae("wf-vae", prep, enc, dec)


def add_args(ap: argparse.ArgumentParser) -> None:
    """Add the VAE choice and weight paths to a parser."""
    ap.add_argument("--vae", required=True, choices=VAES)
    ap.add_argument("--maisi-config", help="MAISI config json that defines autoencoder_def")
    ap.add_argument("--maisi-ckpt", help="MAISI autoencoder checkpoint")
    ap.add_argument("--medsyn2-repo", help="clone of the MedSynV2 repository")
    ap.add_argument("--medsyn2-ckpt", help="MedSynV2 vae.ckpt")
    ap.add_argument("--wan-path", default=WAN_DEFAULT, help="Wan 2.1 Diffusers repo id, local snapshot or its vae folder")
    ap.add_argument("--wfvae-repo", help="clone of the WF-VAE repository")
    ap.add_argument("--wfvae-ckpt", help="WF-VAE-L 16-channel checkpoint folder")
    ap.add_argument("--device", default="cuda")


def build(args: argparse.Namespace) -> Vae:
    """Build the VAE named by --vae."""
    device = torch.device(args.device)
    if args.vae == "maisi":
        return build_maisi(args.maisi_config, args.maisi_ckpt, device)
    if args.vae == "medsyn2":
        return build_medsyn2(args.medsyn2_repo, args.medsyn2_ckpt, device)
    if args.vae == "wan2.1":
        return build_wan21(args.wan_path, device)
    return build_wfvae(args.wfvae_repo, args.wfvae_ckpt, device)
