#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cek environment untuk RVL-DiffGrid.

Menampilkan versi paket inti + memverifikasi apakah build PyTorch Anda
benar-benar mendukung GPU Anda (penting untuk RTX 5090 / Blackwell = sm_120).

Jalankan:  python tools/check_env.py
"""
import importlib
import platform
import sys


def ver(mod_name, attr="__version__"):
    try:
        m = importlib.import_module(mod_name)
        return getattr(m, attr, "(no __version__)")
    except Exception as e:  # noqa
        return f"NOT INSTALLED ({type(e).__name__})"


def main():
    print("=" * 60)
    print("RVL-DiffGrid — environment check")
    print("=" * 60)
    print(f"Python           : {platform.python_version()} ({sys.executable})")
    print(f"OS               : {platform.platform()}")

    # ---- paket inti ----
    print("\n[ Core packages ]")
    for name, mod in [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("open_clip_torch", "open_clip"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("tqdm", "tqdm"),
        ("PyYAML", "yaml"),
    ]:
        print(f"  {name:18s}: {ver(mod)}")

    # ---- pengecekan GPU / CUDA ----
    print("\n[ CUDA / GPU ]")
    try:
        import torch
        print(f"  torch CUDA build : {torch.version.cuda}")
        print(f"  cuDNN            : {torch.backends.cudnn.version()}")
        avail = torch.cuda.is_available()
        print(f"  cuda.is_available: {avail}")

        if avail:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)   # mis. (12, 0) utk RTX 5090
            cap_str = f"sm_{cap[0]}{cap[1]}"
            arch_list = torch.cuda.get_arch_list()         # arsitektur yang DIDUKUNG build ini
            print(f"  Device           : {name}")
            print(f"  Compute capab.   : {cap_str}  {cap}")
            print(f"  Build arch_list  : {arch_list}")

            # verifikasi: apakah build mendukung GPU ini?
            supported = (
                cap_str in arch_list
                or any(a.replace("sm_", "") == f"{cap[0]}{cap[1]}" for a in arch_list)
            )
            if supported:
                print(f"\n  ✅ Build PyTorch mendukung {cap_str}. GPU siap dipakai.")
            else:
                print(
                    f"\n  ❌ Build PyTorch ini TIDAK menyertakan kernel {cap_str}.\n"
                    f"     RTX 5090 = sm_120 butuh torch>=2.7 dengan CUDA 12.8 (wheel cu128).\n"
                    f"     Perbaiki dengan:\n"
                    f"       pip uninstall -y torch torchvision\n"
                    f"       pip install torch torchvision --index-url "
                    f"https://download.pytorch.org/whl/cu128"
                )
        else:
            print("  (CUDA tidak tersedia — cek driver / instalasi torch)")
    except Exception as e:  # noqa
        print(f"  torch belum terpasang dengan benar: {e}")

    print("\nSelesai.")


if __name__ == "__main__":
    main()
