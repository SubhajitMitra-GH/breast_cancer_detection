from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = Path("data/PROCESSED/CBIS-DDSM")
OUTPUT_DIR = Path("data/FINAL/CBIS-DDSM")

TARGET_SIZE = 224

for label in ["BENIGN", "MALIGNANT"]:
    (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

def resize_with_padding(img, target=224):
    h, w = img.shape
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    padded = np.zeros((target, target), dtype=np.uint8)

    y_off = (target - nh) // 2
    x_off = (target - nw) // 2
    padded[y_off:y_off+nh, x_off:x_off+nw] = resized

    return padded

corrupt = 0
processed = 0

for label in ["BENIGN", "MALIGNANT"]:
    imgs = list((INPUT_DIR / label).glob("*.jpg"))

    for img_path in tqdm(imgs, desc=f"Processing {label}"):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Unreadable image")

            img = resize_with_padding(img)
            out_path = OUTPUT_DIR / label / img_path.name
            cv2.imwrite(str(out_path), img)
            processed += 1

        except Exception:
            corrupt += 1

print("\nDONE")
print(f"Processed images: {processed}")
print(f"Corrupt images skipped: {corrupt}")
