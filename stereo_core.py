import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image


# ── helpers ────────────────────────────────────────────────────────────────

def img_to_b64(img_bgr):
    """BGR numpy array → base64 PNG string for embedding in HTML."""
    ok, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode('utf-8')


def normalize_points_2d(x):
    """
    Normalize 2-D homogeneous points (3×N) so centroid→(0,0) and
    mean distance from origin → √2.
    Returns (xn, T).
    """
    x_inh = x[:2] / x[2]          # 2×N inhomogeneous
    cx, cy = x_inh[0].mean(), x_inh[1].mean()
    xs = x_inh - np.array([[cx], [cy]])
    mean_d = np.sqrt((xs**2).sum(axis=0)).mean()
    s = np.sqrt(2) / mean_d
    T = np.array([[s, 0, -s*cx],
                  [0, s, -s*cy],
                  [0, 0,  1.0]])
    xn = T @ x
    return xn, T


def fundamental_normalized_8point(x, xp):
    """
    Compute fundamental matrix via normalised 8-point algorithm.
    x, xp : 3×N homogeneous point arrays.
    """
    N = x.shape[1]
    xn,  T  = normalize_points_2d(x)
    xpn, Tp = normalize_points_2d(xp)

    u  = xn[0]  / xn[2];  v  = xn[1]  / xn[2]
    up = xpn[0] / xpn[2]; vp = xpn[1] / xpn[2]

    A = np.column_stack([
        up*u, up*v, up,
        vp*u, vp*v, vp,
        u,    v,    np.ones(N)
    ])

    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1]
    Fn = f.reshape(3, 3)

    U, S, Vt2 = np.linalg.svd(Fn)
    S[2] = 0
    Fn_rank2 = U @ np.diag(S) @ Vt2

    F = Tp.T @ Fn_rank2 @ T
    if abs(F[2, 2]) > 1e-12:
        F /= F[2, 2]
    return F


def compute_epipoles(F):
    _, _, Vt  = np.linalg.svd(F)
    e_left = Vt[-1]; e_left /= e_left[2]

    _, _, Vt2 = np.linalg.svd(F.T)
    e_right = Vt2[-1]; e_right /= e_right[2]
    return e_left, e_right


def epipolar_error(F, x, xp):
    errs = np.array([xp[:, i] @ F @ x[:, i] for i in range(x.shape[1])])
    return float(np.mean(np.abs(errs)))


# ── stereo matching ─────────────────────────────────────────────────────────

def stereo_matching(left_gray, right_gray, window_size=15, max_disparity=64):
    """SSD block-matching disparity map (pure NumPy, vectorised over disparity)."""
    left  = left_gray.astype(np.float32)
    right = right_gray.astype(np.float32)
    rows, cols = left.shape
    hw = window_size // 2

    best_cost = np.full((rows, cols), np.inf)
    disp_map  = np.zeros((rows, cols), dtype=np.float32)

    for d in range(max_disparity + 1):
        shifted = np.roll(right, d, axis=1)
        shifted[:, :d] = 0
        diff = left - shifted

        # integral-image trick for fast box sum
        ii = np.cumsum(np.cumsum(diff**2, axis=0), axis=1)

        def box_sum(ii, r, c, hw):
            r1, r2 = r - hw - 1, r + hw
            c1, c2 = c - hw - 1, c + hw
            s = ii[r2, c2].copy()
            if r1 >= 0: s -= ii[r1, c2]
            if c1 >= 0: s -= ii[r2, c1]
            if r1 >= 0 and c1 >= 0: s += ii[r1, c1]
            return s

        r2 = np.clip(np.arange(rows) + hw, 0, rows - 1)
        r1 = np.clip(np.arange(rows) - hw - 1, -1, rows - 1)
        c2 = np.clip(np.arange(cols) + hw, 0, cols - 1)
        c1 = np.clip(np.arange(cols) - hw - 1, -1, cols - 1)

        S  = ii[np.ix_(r2, c2)]
        S -= np.where(r1[:, None] >= 0, ii[np.ix_(np.maximum(r1,0), c2)], 0)
        S -= np.where(c1[None, :] >= 0, ii[np.ix_(r2, np.maximum(c1,0))], 0)
        r1c1_valid = (r1[:, None] >= 0) & (c1[None, :] >= 0)
        S += np.where(r1c1_valid, ii[np.ix_(np.maximum(r1,0), np.maximum(c1,0))], 0)

        mask = S < best_cost
        best_cost[mask] = S[mask]
        disp_map[mask] = d

    # zero out borders
    disp_map[:hw, :]  = 0; disp_map[-hw:, :] = 0
    disp_map[:, :hw]  = 0; disp_map[:, -hw:] = 0
    return disp_map


def disparity_to_color(disp_map):
    """Convert float disparity map to a coloured BGR image."""
    norm = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    return colored


# ── feature matching ────────────────────────────────────────────────────────

def detect_and_match_features(left_gray, right_gray, max_pts=200):
    """SIFT feature detection and BF matching with Lowe ratio test."""
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(left_gray,  None)
    kp2, des2 = sift.detectAndCompute(right_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    good = sorted(good, key=lambda m: m.distance)[:max_pts]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).T  # 2×N
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).T

    x  = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
    xp = np.vstack([pts2, np.ones((1, pts2.shape[1]))])
    return x, xp, kp1, kp2, good


def draw_matches_image(left_bgr, right_bgr, kp1, kp2, good, max_draw=40):
    h1, w1 = left_bgr.shape[:2]
    h2, w2 = right_bgr.shape[:2]
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = left_bgr
    out[:h2, w1:] = right_bgr

    for m in good[:max_draw]:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = (int(kp2[m.trainIdx].pt[0]) + w1, int(kp2[m.trainIdx].pt[1]))
        color = tuple(np.random.randint(100, 255, 3).tolist())
        cv2.circle(out, pt1, 4, color, -1)
        cv2.circle(out, pt2, 4, color, -1)
        cv2.line(out, pt1, pt2, color, 1, cv2.LINE_AA)
    return out


# ── epipolar lines ──────────────────────────────────────────────────────────

def draw_epipolar_lines(left_bgr, right_bgr, x, xp, F, n_lines=12):
    """Draw epipolar lines on both images; return two BGR images."""
    im1 = left_bgr.copy(); im2 = right_bgr.copy()
    h, w = im1.shape[:2]

    idx = np.round(np.linspace(0, x.shape[1]-1, n_lines)).astype(int)

    colors = [tuple(c) for c in (np.random.randint(50,255,(n_lines,3))).tolist()]

    for k, i in enumerate(idx):
        col = colors[k]
        # point in image 1
        px1 = int(x[0,i]/x[2,i]); py1 = int(x[1,i]/x[2,i])
        cv2.circle(im1, (px1, py1), 6, col, -1)

        # epipolar line in image 1: l = F^T x'
        l = F.T @ xp[:, i]
        _draw_hline(im1, l, col)

        # point in image 2
        px2 = int(xp[0,i]/xp[2,i]); py2 = int(xp[1,i]/xp[2,i])
        cv2.circle(im2, (px2, py2), 6, col, -1)

        # epipolar line in image 2: l' = F x
        lp = F @ x[:, i]
        _draw_hline(im2, lp, col)

    return im1, im2


def _draw_hline(img, l, color):
    h, w = img.shape[:2]
    a, b, c = float(l[0]), float(l[1]), float(l[2])
    pts = []
    if abs(b) > 1e-10:
        y = int((-a*0 - c) / b)
        if 0 <= y < h: pts.append((0, y))
        y = int((-a*(w-1) - c) / b)
        if 0 <= y < h: pts.append((w-1, y))
    if abs(a) > 1e-10:
        x = int((-b*0 - c) / a)
        if 0 <= x < w: pts.append((x, 0))
        x = int((-b*(h-1) - c) / a)
        if 0 <= x < w: pts.append((x, h-1))
    if len(pts) >= 2:
        cv2.line(img, pts[0], pts[1], color, 1, cv2.LINE_AA)


# ── rectification ───────────────────────────────────────────────────────────

def rectify_images(left_bgr, right_bgr, x, xp, F):
    """Compute homographies, warp images, return rectified pair."""
    h, w = left_bgr.shape[:2]
    pts1 = (x[:2] / x[2]).T.astype(np.float32)   # N×2
    pts2 = (xp[:2] / xp[2]).T.astype(np.float32)

    # use RANSAC F for robust rectification
    F_cv, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(inliers1, inliers2, F_cv, (w, h))
    if not ok:
        return None, None, None, None

    left_rect  = cv2.warpPerspective(left_bgr,  H1, (w, h))
    right_rect = cv2.warpPerspective(right_bgr, H2, (w, h))

    # draw horizontal scanlines on side-by-side view
    side = np.concatenate([left_rect, right_rect], axis=1)
    for y in np.linspace(30, h-30, 12).astype(int):
        cv2.line(side, (0, y), (w*2, y), (0, 0, 255), 1)

    return left_rect, right_rect, side, F_cv
