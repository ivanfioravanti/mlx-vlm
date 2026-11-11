"""Few-step LoRA adapters and the base flow-matching head for H3-Image."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import HfFileSystem, constants, hf_hub_download
from mlx import nn

from .download import SOURCE_REPO_ID

_LORA_KEY = re.compile(
    r"^(?:transformer\.|diffusion_model\.)?(.+?)\.(lora_A|lora_B|alpha)"
    r"(?:\.[A-Za-z_]\w*)?(?:\.weight)?$"
)
# The base head lives only in the official native checkpoint. The diffusers and
# MLX layouts carry the six step-distilled PDD heads instead.
_HEAD_FILE = "transformer/model-00001-of-00012.safetensors"
_HEAD_KEYS = {
    "final_layer.video_out.weight": "weight",
    "final_layer.video_out.bias": "bias",
}


class LoRALinear(nn.Module):
    """A frozen (possibly quantized) linear plus a low-rank update."""

    def __init__(self, base, lora_a, lora_b, scale=1.0):
        super().__init__()
        self.base = base
        dtype = getattr(base, "scales", base.weight).dtype
        self.lora_a = lora_a.T.astype(dtype)
        self.lora_b = (lora_b.astype(mx.float32) * scale).T.astype(dtype)

    def __call__(self, x):
        y = self.base(x)
        update = (x.astype(self.lora_a.dtype) @ self.lora_a) @ self.lora_b
        return y + update.astype(y.dtype)


def resolve_lora(path):
    """A local safetensors file, or ``org/repo/file.safetensors`` on the Hub."""
    local = Path(str(path)).expanduser()
    if local.is_file():
        return local
    parts = str(path).split("/")
    if len(parts) < 3 or not parts[-1].endswith(".safetensors"):
        raise ValueError(
            "lora_path must be a local .safetensors file or a Hub reference "
            f"shaped org/repo/file.safetensors, got {path!r}"
        )
    return Path(hf_hub_download("/".join(parts[:2]), "/".join(parts[2:])))


def parse_lora(weights):
    """Group diffusers/PEFT LoRA tensors as {module: {lora_A, lora_B, alpha}}."""
    modules = {}
    for key, value in weights.items():
        match = _LORA_KEY.match(key)
        if match:
            modules.setdefault(match.group(1), {})[match.group(2)] = value
    return modules


def apply_lora(transformer, path, scale=1.0):
    """Wrap every targeted linear of ``transformer`` in place. Returns the count."""
    modules = parse_lora(mx.load(str(resolve_lora(path))))
    applied = 0
    for name, parts in modules.items():
        if "lora_A" not in parts or "lora_B" not in parts:
            continue
        *parents, leaf = name.split(".")
        try:
            parent = transformer
            for part in parents:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            base = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        except (AttributeError, IndexError, KeyError):
            raise ValueError(f"LoRA targets an unknown H3-Image module: {name}")
        rank = parts["lora_A"].shape[0]
        alpha = float(parts["alpha"].item()) if "alpha" in parts else rank
        wrapped = LoRALinear(
            base, parts["lora_A"], parts["lora_B"], scale * alpha / rank
        )
        if leaf.isdigit():
            parent[int(leaf)] = wrapped
        else:
            setattr(parent, leaf, wrapped)
        applied += 1
    if not applied:
        raise ValueError(f"No LoRA weights matched the H3-Image transformer: {path}")
    mx.eval(transformer.parameters())
    return applied


def _read_remote_tensors(repo, filename, names):
    """Range-read a few tensors from a Hub safetensors shard as float32."""
    dtypes = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}
    result = {}
    with HfFileSystem().open(f"{repo}/{filename}", "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        for name in names:
            meta = header[name]
            start, end = meta["data_offsets"]
            handle.seek(8 + size + start)
            raw = np.frombuffer(handle.read(end - start), dtypes[meta["dtype"]])
            if meta["dtype"] == "BF16":
                raw = (raw.astype(np.uint32) << 16).view(np.float32)
            result[name] = raw.astype(np.float32).reshape(meta["shape"])
    return result


def load_flow_head(transformer, repo=SOURCE_REPO_ID):
    """Attach the checkpoint's base output head, used by few-step flow sampling."""
    cache = (
        Path(constants.HF_HOME) / "mlx_vlm" / "minimax_h3_image_flow_head.safetensors"
    )
    if cache.is_file():
        weights = mx.load(str(cache))
    else:
        tensors = _read_remote_tensors(repo, _HEAD_FILE, _HEAD_KEYS)
        weights = {_HEAD_KEYS[key]: mx.array(value) for key, value in tensors.items()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(cache), weights)
    reference = transformer.pdd_proj_out[0]
    if weights["weight"].shape != reference.weight.shape:
        raise ValueError("Base flow head does not match this H3-Image transformer")
    head = nn.Linear(*reference.weight.shape[::-1])
    head.weight = weights["weight"].astype(reference.weight.dtype)
    head.bias = weights["bias"].astype(reference.bias.dtype)
    transformer.proj_out = head
    mx.eval(head.parameters())
    return head
