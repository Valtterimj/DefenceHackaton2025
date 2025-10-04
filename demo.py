import os, cv2, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def add_gaussian_noise(img, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# =========================
# 1) LOAD + BASIC VIEW
# =========================
data = os.path.join(os.getcwd(), 'data')
map_path = os.path.join(data, 'ICEYE.png')

full = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
# Adjust crop if needed
# r0, r1, c0, c1 = 1900, 2100, 1900, 2100
r0, r1, c0, c1 = 2100, 2200, 2200, 2300
r1 = min(r1, full.shape[0]); c1 = min(c1, full.shape[1])
crop = full[r0:r1, c0:c1]

crop = cv2.GaussianBlur(crop, (3,3), sigmaX=0.5)

map_path = os.path.join(data, 'map_village.jpg')
crop_path = os.path.join(data, 'frame_0.jpg')

full = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(full, cmap='gray'); plt.title("Satellite"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(crop, cmap='gray'); plt.title("Drone"); plt.axis("off")
plt.tight_layout(); plt.show()

def clahe(img, clip=2.0, tiles=(8,8)):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(img)

full_eq = clahe(full)
crop_eq = clahe(crop)

# =========================
# 2) EDGES (CANNY)
#  
# =========================
def auto_canny(img, sigma=0.33):
    v = np.median(img)
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lo, hi, L2gradient=True)

full_blur = cv2.GaussianBlur(full_eq, (3,3), 0)
crop_blur = cv2.GaussianBlur(crop_eq, (3,3), 0)

edges_full = auto_canny(full_blur)
edges_crop = auto_canny(crop_blur)



plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(edges_full, cmap='gray'); plt.title("Satellite (Canny edges)"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(edges_crop, cmap='gray'); plt.title("Drone (Canny edges)"); plt.axis("off")
plt.tight_layout(); plt.show()

# =========================
# 3) STRAIGHT LINES (HoughLinesP)
#   
# =========================
lines_full = cv2.HoughLinesP(edges_full, 1, np.pi/180, threshold=60,
                             minLineLength=60, maxLineGap=10)
lines_crop = cv2.HoughLinesP(edges_crop, 1, np.pi/180, threshold=40,
                             minLineLength=25, maxLineGap=8)

def overlay_lines(gray, lines, color=(0,255,0), th=2):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            cv2.line(vis, (x1,y1), (x2,y2), color, th, cv2.LINE_AA)
    return vis

full_lines_vis = overlay_lines(full_eq, lines_full)
crop_lines_vis = overlay_lines(crop_eq, lines_crop)

full_lines_vis_rgb = cv2.cvtColor(full_lines_vis, cv2.COLOR_BGR2RGB)
crop_lines_vis_rgb = cv2.cvtColor(crop_lines_vis, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(full_lines_vis_rgb); plt.title("Satellite + Hough lines"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(crop_lines_vis_rgb); plt.title("Drone + Hough lines"); plt.axis("off")
plt.tight_layout(); plt.show()

# =========================
# 4) LINE MASKS + TEMPLATE MATCH
#   
# =========================
mask_full = np.zeros_like(full, dtype=np.uint8)
mask_crop = np.zeros_like(crop, dtype=np.uint8)

if lines_full is not None:
    for x1,y1,x2,y2 in lines_full[:,0,:]:
        cv2.line(mask_full, (x1,y1), (x2,y2), 255, 2, cv2.LINE_AA)
else:
    mask_full = edges_full.copy()

if lines_crop is not None:
    for x1,y1,x2,y2 in lines_crop[:,0,:]:
        cv2.line(mask_crop, (x1,y1), (x2,y2), 255, 2, cv2.LINE_AA)
else:
    mask_crop = edges_crop.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask_full = cv2.dilate(mask_full, kernel, iterations=1)
mask_crop = cv2.dilate(mask_crop, kernel, iterations=1)

h, w = mask_crop.shape
H, W = mask_full.shape
if h >= H or w >= W:
    raise ValueError("Crop must be smaller than the full image.")

res = cv2.matchTemplate(mask_full, mask_crop, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
print(f"[matchTemplate] score: {max_val:.3f}, top-left: {top_left}, size: ({w},{h})")

# =========================
# 5) RESULT VISUALIZATION
#  
# =========================

full_box_bgr = cv2.cvtColor(full_eq, cv2.COLOR_GRAY2BGR)
cv2.rectangle(full_box_bgr, top_left, bottom_right, (0,0,255), 3)  # red in BGR
full_box_rgb = cv2.cvtColor(full_box_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(full_box_rgb); plt.title("Detected location on the satellite image"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(crop, cmap='gray'); plt.title("Drone"); plt.axis("off")
plt.tight_layout(); plt.show()
