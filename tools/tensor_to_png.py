"""
Mengubah dump tensor NCHW float32 (hasil debugReadback) menjadi PNG supaya bisa DILIHAT.

Pemakaian:
  python tensor_to_png.py scrfd.bin 640 640 scrfd.png scrfd      # normalisasi (x-127.5)/128
  python tensor_to_png.py arc.bin   112 112 arc.png   arcface    # normalisasi (x-127.5)/127.5
  python tensor_to_png.py pad.bin    80  80 pad.png   pad01      # nilai 0..1  (x/255)
  python tensor_to_png.py pad.bin    80  80 pad.png   pad255     # nilai 0..255 (BGR atau RGB sesuai shader)
Butuh: pip install numpy pillow
"""
import sys
import numpy as np
from PIL import Image

path, w, h, out, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
data = np.fromfile(path, dtype=np.float32)
assert data.size == 3 * h * w, f"ukuran file {data.size} != 3*{h}*{w}"
chw = data.reshape(3, h, w)

if mode == "scrfd":
    img = chw * 128.0 + 127.5
elif mode == "arcface":
    img = chw * 127.5 + 127.5
elif mode == "pad01":
    img = chw * 255.0
elif mode == "pad255":
    img = chw
else:
    raise SystemExit("mode tidak dikenal")

img = np.clip(img, 0, 255).astype(np.uint8).transpose(1, 2, 0)  # CHW -> HWC
Image.fromarray(img, "RGB").save(out)
print(f"tersimpan {out}  min={chw.min():.4f} max={chw.max():.4f} mean={chw.mean():.4f}")
if mode == "scrfd":
    # Padding hitam ternormalisasi = (0-127.5)/128 = -0.99609375. Harus muncul di KANAN dan BAWAH.
    print("nilai pojok kiri-atas  :", chw[:, 0, 0])
    print("nilai pojok kanan-bawah:", chw[:, -1, -1], "(harus sekitar -0.996 jika ada padding)")
