import os, cv2, numpy as np
import matplotlib.pyplot as plt

# ============================================================
#                 PREPROCESSING HELPERS
# ============================================================

def clahe(img, clip=2.0, tiles=(8,8)):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(img)

def auto_canny(img):
    return cv2.Canny(img, 75, 115, L2gradient=True)

def preprocess_edges(gray):
    """
    Standardize, smooth, canny edges, light dilation to thicken thin lines.
    Args:
        gray: Grayscale image (uint8).
    Returns:
        eq    : CLAHE-equalized image (for display / later stages).
        edges : Dilated Canny edge map (uint8, 0/255).
    """
    eq   = clahe(gray, 2.0, (8,8))
    blur = cv2.GaussianBlur(eq, (3,3), 0.6)
    edges = auto_canny(blur)
    # edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1) # Better without dilation
    return eq, edges

# ============================================================
#                 COARSE CANDIDATE GENERATION
# ============================================================

def multiscale_match_edges(full_edge, crop_edge, scales=None, topk=5, return_maps=False):
    """
    Coarse search: multi-scale template matching on *edge maps*.

    For each scale s, we resize the template (crop_edge) and slide it over
    full_edge using cv2.matchTemplate (TM_CCOEFF_NORMED). We take up to 3 best
    peaks per scale with a simple local suppression, then collect the global
    top-K candidates across scales.

    Args:
        full_edge  : Satellite edge map (uint8).
        crop_edge  : Drone edge map (uint8).
        scales     : Iterable of scales for the template. If None, use linspace 0.75..1.35.
        topk       : Keep best K candidates overall.
        return_maps: If True, attach the raw response map (only used for debugging).

    Returns:
        List of tuples:
          (score, (x, y), (w, h), scale)             if return_maps=False
          (score, (x, y), (w, h), scale, res, (tw,th)) if return_maps=True
        where (x,y) is top-left on the full image, (w,h) = template size at that scale.
    """
    if scales is None:
        scales = np.linspace(0.75, 1.35, 13)

    H, W = full_edge.shape
    candidates = []

    for s in scales:
        tw = max(8, int(crop_edge.shape[1] * s))
        th = max(8, int(crop_edge.shape[0] * s))
        tmpl = cv2.resize(crop_edge, (tw, th), interpolation=cv2.INTER_NEAREST)
        if th > H or tw > W:
            continue

        res = cv2.matchTemplate(full_edge, tmpl, cv2.TM_CCOEFF_NORMED)
        res_n = res.copy()

        # Up to 3 peaks per scale (simple non-max suppression window)
        for _ in range(3):
            _, max_val, _, max_loc = cv2.minMaxLoc(res_n)
            x, y = max_loc
            if return_maps:
                candidates.append((float(max_val), (x, y), (tw, th), float(s), res, (tw, th)))
            else:
                candidates.append((float(max_val), (x, y), (tw, th), float(s)))

            # Suppress a half-window around the found peak to avoid near-duplicates
            y0 = max(0, y - th // 2); y1 = min(res.shape[0], y + th // 2)
            x0 = max(0, x - tw // 2); x1 = min(res.shape[1], x + tw // 2)
            res_n[y0:y1, x0:x1] = -1.0

    # Global Top-K across all scales
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[:topk]

def draw_matches_colored(img1, kp1, img2, kp2, matches, inlier_mask, max_matches=300):
    """
    Draw matches between two images with inliers in green, outliers in red.
    Left = img1 (crop), Right = img2 (full); img2 is drawn to the right of img1.

    Args:
        img1, kp1 : First image and its keypoints
        img2, kp2 : Second image and its keypoints
        matches   : List of cv2.DMatch
        inlier_mask : Boolean array (same length as matches) or None
        max_matches : Maximum number of matches to draw (default=300)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1]      = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for i, m in enumerate(matches[:max_matches]):  # <--- slice here
        pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
        pt2 = tuple(np.int32(kp2[m.trainIdx].pt))
        pt2_shift = (pt2[0] + w1, pt2[1])
        color = (0,255,0) if (inlier_mask is not None and inlier_mask[i]) else (0,0,255)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2_shift, 3, color, -1)
        cv2.line(canvas, pt1, pt2_shift, color, 1, cv2.LINE_AA)
    return canvas


def extract_roi(img, x, y, w, h, pad=40):
    """
    Extract a padded region of interest (ROI) around a coarse candidate box.
    Bounds are clamped to the image size.

    Returns:
        roi      : Cropped ROI (same dtype as img).
        (x0, y0) : Top-left origin of the ROI within the full image.
    """
    H, W = img.shape[:2]
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return img[y0:y1, x0:x1], (x0, y0)

# ============================================================
#                 FINE ALIGNMENT (ORB + RANSAC)
# ============================================================

def orb_ransac_align(crop_img, full_roi, roi_origin_xy,
                     nfeatures=8000, ratio=0.78, reproj=3.5):
    """
    Fine alignment inside a candidate ROI using ORB keypoints and RANSAC.

    Steps:
      - Preprocess (CLAHE + blur) both images.
      - ORB detect+describe.
      - KNN match (Lowe ratio test).
      - RANSAC (estimateAffinePartial2D) to get similarity transform.

    Returns:
        dict with: M (2x3 affine), inliers, mean_err, kp_c, kp_r, good, inl_mask, roi_origin, ...
        or None if not enough matches / model fit failed.
    """
    def prep(gray):
        eq = clahe(gray, 2.0, (8,8))
        return cv2.GaussianBlur(eq, (3,3), 0.6)

    crop_p = prep(crop_img)
    roi_p  = prep(full_roi)

    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=15, patchSize=31, fastThreshold=10)

    kp_c, des_c = orb.detectAndCompute(crop_p, None)
    kp_r, des_r = orb.detectAndCompute(roi_p,  None)
    if des_c is None or des_r is None or len(kp_c) < 4 or len(kp_r) < 4:
        return None

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_c, des_r, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 6:
        return None

    # Prepare RANSAC correspondence sets in full-image coordinates
    src     = np.float32([kp_c[m.queryIdx].pt for m in good])     # crop coords
    dst_roi = np.float32([kp_r[m.trainIdx].pt for m in good])     # ROI coords
    ox, oy  = roi_origin_xy
    dst     = dst_roi + np.array([ox, oy], np.float32)            # shift to full coords

    M, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=reproj,
        maxIters=5000, confidence=0.995
    )
    if M is None or inl is None or inl.sum() < 3:
        return None

    # Compute mean reprojection error on inliers (diagnostic)
    inlier_count = int(inl.sum())
    src_h = np.hstack([src, np.ones((len(src), 1), np.float32)])
    proj = (M @ src_h.T).T
    err = np.linalg.norm(proj - dst, axis=1)
    mean_err = float(np.mean(err[inl.flatten() == 1])) if inlier_count > 0 else 1e9

    return dict(M=M, inliers=inlier_count, mean_err=mean_err,
                kp_c=kp_c, kp_r=kp_r, good=good, inl_mask=inl,
                roi_origin=(ox, oy),
                crop_p=crop_p, roi_p=roi_p)

# ============================================================
#                 VISUALIZATION HELPERS
# ============================================================

def overlay_edges(gray, edges, color=(255,0,0)):
    """
    Overlay a colored edge mask on top of a grayscale image for inspection.
    """
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb[edges > 0] = color
    return rgb

def show_edges_and_candidates(full_eq, full_edge, crop_eq, crop_edge, cands):
    """
    Show (CLAHE) images and overlay the top coarse candidates on the satellite.
    Draws the best candidate in RED, others in YELLOW.
    """
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.imshow(full_eq, cmap='gray'); plt.title("Satellite"); plt.axis("off")
    plt.subplot(2,2,2); plt.imshow(crop_eq, cmap='gray'); plt.title("Drone"); plt.axis("off")

    vis_edges = overlay_edges(full_eq, full_edge)
    for i, (score, (x,y), (w,h), s, *_) in enumerate(cands):
        cv2.rectangle(vis_edges, (x, y), (x + w, y + h),
                      (0,0,255) if i == 0 else (0,255,255), 2)
        cv2.putText(vis_edges, f"{i+1}:{score:.2f}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255) if i == 0 else (0,255,255), 1, cv2.LINE_AA)

    plt.subplot(2,2,3); plt.imshow(cv2.cvtColor(vis_edges, cv2.COLOR_BGR2RGB))
    plt.title("Satellite edges + ROI candidates"); plt.axis("off")
    plt.subplot(2,2,4); plt.imshow(crop_edge, cmap='gray'); plt.title("Drone edges"); plt.axis("off")
    plt.tight_layout(); plt.show()



def draw_matches_colored(img1, kp1, img2, kp2, matches, inlier_mask, max_matches=500):
    """
    Draw matches between two images with inliers in green, outliers in red.
    Left = img1 (crop), Right = img2 (full); img2 is drawn to the right of img1.

    Args:
        img1, kp1 : First image and its keypoints
        img2, kp2 : Second image and its keypoints
        matches   : List of cv2.DMatch
        inlier_mask : Boolean array (same length as matches) or None
        max_matches : Maximum number of matches to draw (default=300)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1]      = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for i, m in enumerate(matches[:max_matches]):  # <--- slice here
        pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
        pt2 = tuple(np.int32(kp2[m.trainIdx].pt))
        pt2_shift = (pt2[0] + w1, pt2[1])
        color = (0,255,0) if (inlier_mask is not None and inlier_mask[i]) else (0,0,255)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2_shift, 3, color, -1)
        cv2.line(canvas, pt1, pt2_shift, color, 1, cv2.LINE_AA)
    return canvas


# ============================================================
#                 DATA LOADING
# ============================================================

data = os.path.join(os.getcwd(), 'data')

# Satellite image (grayscale)
map_path = os.path.join(data, 'RAJA2.png')
full = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

# Drone image (grayscale), then rotate slightly and blur a bit to simulate conditions
drone_path = os.path.join(data, 'RAJA2_drone_1.png')
crop = cv2.imread(drone_path, cv2.IMREAD_GRAYSCALE)

if full is None:
    raise FileNotFoundError(f"Missing satellite image: {map_path}")
if crop is None:
    raise FileNotFoundError(f"Missing drone image: {drone_path}")


crop  = cv2.GaussianBlur(crop, (0,0), 1.0)

# ============================================================
#                 STAGE 1: COARSE CANDIDATES
# ============================================================

# Produce CLAHE images and edge maps for both frames
full_eq, full_edge = preprocess_edges(full)
crop_eq, crop_edge = preprocess_edges(crop)

# Run multi-scale template matching on edges (Top-K)
cands = multiscale_match_edges(full_edge, crop_edge, topk=5, return_maps=False)

if not cands:
    raise RuntimeError("No coarse candidates found. Consider widening scale range or adjusting Canny thresholds.")

print('show_edges_and_candidates')
show_edges_and_candidates(full_eq, full_edge, crop_eq, crop_edge, cands)

# ============================================================
#                 STAGE 2: FINE (ORB + RANSAC)
# ============================================================

best = None
for score, (x, y), (w_box, h_box), s in cands:
    # Extract a padded ROI around each coarse box
    roi, origin = extract_roi(full, x, y, w_box, h_box, pad=60)

    # Align the crop to this ROI using ORB + RANSAC
    result = orb_ransac_align(crop, roi, origin,
                              nfeatures=10000, ratio=0.8, reproj=4.5)
    if result is None:
        continue

    # Prefer more inliers, then smaller reprojection error, then higher coarse score
    key = (result["inliers"], -result["mean_err"], score)
    if (best is None) or (key > best["key"]):
        result["key"] = key
        result["coarse"] = (score, (x, y), (w_box, h_box), s)
        best = result

if best is None:
    raise RuntimeError("Could not refine any candidate with ORB+RANSAC. Try larger pad, more ORB features, or relax ratio threshold.")

M = best["M"]
inliers   = best["inliers"]
mean_err  = best["mean_err"]
(score, (x, y), (w_box, h_box), s) = best["coarse"]
print(f"Chosen candidate: coarse_score={score:.3f}, inliers={inliers}, mean reproj err={mean_err:.2f}px")

# ============================================================
#                 VISUALIZATIONS
# ============================================================

# (1) Keypoints right after ORB (crop & the *same* ROI used for the winner)
kp_c, kp_r = best["kp_c"], best["kp_r"]
roi, origin = extract_roi(full, x, y, w_box, h_box, pad=60)

kp_vis_crop = cv2.drawKeypoints(crop, kp_c, None, color=(0,255,0), flags=0)
kp_vis_roi  = cv2.drawKeypoints(roi,  kp_r, None, color=(0,255,0), flags=0)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(kp_vis_crop, cv2.COLOR_BGR2RGB)); plt.title("Drone – ORB keypoints"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(kp_vis_roi,  cv2.COLOR_BGR2RGB)); plt.title("Satellite ROI – ORB keypoints"); plt.axis("off")
plt.tight_layout(); plt.show()

# Colored matches (green = inliers, red = outliers)
good     = best["good"]
inl_mask = best["inl_mask"].flatten().astype(bool)

# Shift ROI keypoints to full-image coords for drawing
ox, oy     = best["roi_origin"]
kp_r_full  = []
for k in kp_r:
    kp_r_full.append(cv2.KeyPoint(float(k.pt[0] + ox),
                                  float(k.pt[1] + oy),
                                  float(k.size),
                                  float(k.angle),
                                  float(k.response),
                                  int(k.octave),
                                  int(k.class_id)))

colored = draw_matches_colored(crop, kp_c, full, kp_r_full, good, inl_mask)
plt.figure(figsize=(14,6))
plt.imshow(colored[..., ::-1]); plt.title("Matches (green=inliers, red=outliers)"); plt.axis("off")
plt.tight_layout(); plt.show()

# Final footprint + overlay (warp the crop onto the satellite with the affine M)
h_c, w_c = crop.shape[:2]
corners   = np.array([[0,0], [w_c,0], [w_c,h_c], [0,h_c]], np.float32)
corners_h = np.hstack([corners, np.ones((4,1), np.float32)])
corners_tr = (M @ corners_h.T).T.astype(int)

sat_vis = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
cv2.polylines(sat_vis, [corners_tr], True, (0,0,255), 3)

warp = cv2.warpAffine(crop, M, (full.shape[1], full.shape[0]))
overlay = cv2.addWeighted(cv2.cvtColor(full, cv2.COLOR_GRAY2BGR), 0.7,
                          cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR), 0.3, 0)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(sat_vis, cv2.COLOR_BGR2RGB)); plt.title("Satellite + oriented footprint"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(overlay,  cv2.COLOR_BGR2RGB)); plt.title("Overlay"); plt.axis("off")
plt.tight_layout(); plt.show()

