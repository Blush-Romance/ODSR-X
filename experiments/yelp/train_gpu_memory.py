"""Run unchanged training code while recomputing candidate activations on GPU."""
from pathlib import Path
import runpy
import sys
import time

import torch
from torch.utils.checkpoint import checkpoint


def install_gpu_recomputation():
    from ODSR import ODSR
    original = ODSR._direct_history_correction

    def correction(self, users, items, final_item, aux, detach_backbone=False):
        if not torch.is_grad_enabled():
            return original(self, users, items, final_item, aux, detach_backbone)
        return checkpoint(original, self, users, items, final_item, aux,
                          detach_backbone, use_reentrant=False,
                          preserve_rng_state=True)

    ODSR._direct_history_correction = correction
    return original


def install_batch_progress():
    original = torch.optim.Adam.step
    steps = 0
    started = time.monotonic()

    def step(self, *args, **kwargs):
        nonlocal steps
        result = original(self, *args, **kwargs)
        steps += 1
        if steps == 1 or steps % 50 == 0:
            print("local_batch_progress: optimizer_steps={} elapsed={:.1f}s peak_cuda_allocated={:.0f}MiB".format(
                steps, time.monotonic() - started,
                torch.cuda.max_memory_allocated() / 1024**2), flush=True)
        return result

    torch.optim.Adam.step = step


if __name__ == "__main__":
    source = Path(__file__).resolve().parent / "source"
    sys.path.insert(0, str(source))
    sys.argv[0] = str(source / "main.py")
    install_gpu_recomputation()
    install_batch_progress()
    print("local_memory_mode: GPU candidate activation recomputation; no CPU offload; original batch size", flush=True)
    runpy.run_path(str(source / "main.py"), run_name="__main__")
