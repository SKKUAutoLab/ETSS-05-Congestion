import io
from pathlib import Path
import numpy as np
import torch
import torch.storage

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")

def main():
    src_dir = Path("data/ShanghaiTech_SAM/part_A/train_stage4_0")
    npy_files = sorted(src_dir.glob("*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    for path in npy_files:
        data = np.load(path, allow_pickle=True).item()
        person_map = data["person_map"]
        np.save(path, person_map)
        print(f"converted {path}")
    print(f"done: converted {len(npy_files)} files")

if __name__ == "__main__":
    main()
