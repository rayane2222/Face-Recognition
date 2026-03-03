import cv2
import glob
import numpy as np
import os

img_dir = "calib_images"

paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    paths += glob.glob(os.path.join(img_dir, ext))

print(f"Images trouvées : {len(paths)}")

imgs = []
for p in paths[:100]:
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    imgs.append(img)

# imgs : liste de (128,128,3)
x = np.stack(imgs, axis=0)      # (N,128,128,3)
x = np.transpose(x, (0, 3, 1, 2))  # (N,3,128,128)

np.savez("blazeface_calib.npz", input=x)

print("NPZ créé :", x.shape)
