from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

DEFAULT_MAX_CPU_MEMORY_GIB = 8
DEFAULT_MAX_DISK_MEMORY_GIB = 96
DEFAULT_DISK_FREE_MIN_GIB = 128
SAFE_MAX_CPU_MEMORY_GIB = 12


def parse_gib_value(value: object) -> int:
    if isinstance(value, int):
        return max(1, value)
    spec = str(value).strip().lower()
    if spec.endswith("gib"):
        spec = spec[:-3]
    elif spec.endswith("gb"):
        spec = spec[:-2]
    if not spec:
        raise ValueError(f"invalid GiB value: {value!r}")
    return max(1, int(float(spec)))


def validate_full_model_load_policy(
    script_name: str,
    *,
    offload_enabled: bool,
    max_cpu_memory_gib: int,
    max_disk_memory_gib: int,
    allow_unsafe_cpu_memory: bool,
    allow_no_disk_offload: bool,
) -> None:
    if not offload_enabled and not allow_no_disk_offload:
        raise SystemExit(
            f"{script_name}: refusing to load the full HF model without disk offload; "
            "enable offload or pass the explicit unsafe override"
        )

    if max_cpu_memory_gib > SAFE_MAX_CPU_MEMORY_GIB and not allow_unsafe_cpu_memory:
        raise SystemExit(
            f"{script_name}: refusing CPU memory budget {max_cpu_memory_gib} GiB; "
            f"safe ceiling is {SAFE_MAX_CPU_MEMORY_GIB} GiB. "
            "Pass the explicit unsafe override only with operator approval"
        )

    if max_disk_memory_gib < max_cpu_memory_gib:
        raise SystemExit(
            f"{script_name}: disk offload budget {max_disk_memory_gib} GiB must be >= "
            f"CPU budget {max_cpu_memory_gib} GiB"
        )


def require_disk_free(paths: list[Path], min_free_gib: int) -> None:
    checked: set[str] = set()
    for path in paths:
        probe = Path(path)
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        resolved = str(probe.resolve())
        if resolved in checked:
            continue
        checked.add(resolved)
        free_gib = shutil.disk_usage(probe).free / (1024 ** 3)
        if free_gib < min_free_gib:
            raise SystemExit(
                f"refusing to run: {probe} has {free_gib:.1f} GiB free, needs at least {min_free_gib} GiB"
            )


def fresh_offload_folder(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root / f"run_{stamp}_{os.getpid()}"


def prepare_model_load_kwargs(
    *,
    torch_dtype,
    offload_enabled: bool,
    offload_folder: Path,
    max_cpu_memory_gib: int,
    max_disk_memory_gib: int,
    local_files_only: bool,
):
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    actual_offload_folder = None
    if offload_enabled:
        actual_offload_folder = fresh_offload_folder(offload_folder)
        actual_offload_folder.mkdir(parents=True, exist_ok=False)
        model_kwargs.update({
            "device_map": "auto",
            "max_memory": {
                "cpu": f"{max(1, max_cpu_memory_gib)}GiB",
                "disk": f"{max(1, max_disk_memory_gib)}GiB",
            },
            "offload_folder": str(actual_offload_folder),
            "offload_state_dict": True,
        })
    else:
        model_kwargs["device_map"] = "cpu"
    return model_kwargs, actual_offload_folder