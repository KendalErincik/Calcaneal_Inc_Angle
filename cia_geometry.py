"""Deterministic landmark extraction and Calcaneal Inclination Angle (CIA) computation.

This is the reproducible, model-independent core referenced in the manuscript: given
three binary segmentation masks (calcaneus inferior, 5th metatarsal head,
calcaneonavigular / inferior calcaneocuboid), it derives the four landmarks P1-P4,
performs the lower-tangent search and returns the CIA in degrees. The segmentation
itself is produced by three independently trained single-class U-Nets (see the paper
and the RadiologyMate application); this file contains only the deterministic geometry,
which is identical regardless of how the masks were obtained.

Dependencies: numpy, opencv-python.

    masks = {"calcaneus_inferior": m0, "metatarsal_5": m1, "calcaneonavicular": m2}
    cia, P1, P2, P3, P4 = compute_from_masks(masks)
"""
import numpy as np
import cv2


def _largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).squeeze(axis=1)
    if c.ndim == 1:
        c = c.reshape(-1, 2)
    return c.astype(np.float64)


def derive_P1(mask):
    """5th metatarsal head: inferior-most contour point (max Y)."""
    c = _largest_contour(mask)
    return c[np.argmax(c[:, 1])] if c is not None else None


def derive_P2(mask):
    """Calcaneus inferior tuberosity: inferior-most contour point (max Y)."""
    c = _largest_contour(mask)
    return c[np.argmax(c[:, 1])] if c is not None else None


def derive_P4(mask, flipped=False):
    """Calcaneonavicular: posterior-inferior corner (max X+Y, or max Y-X if flipped)."""
    c = _largest_contour(mask)
    if c is None:
        return None
    score = c[:, 1] - c[:, 0] if flipped else c[:, 0] + c[:, 1]
    return c[np.argmax(score)]


def _mask_centroid_x(mask):
    ys_xs = np.argwhere(mask > 0)
    if ys_xs.size == 0:
        return None
    return float(ys_xs[:, 1].mean())


def detect_flipped(masks):
    """Orientation from mask geometry: True when the forefoot is left of the heel."""
    fore = _mask_centroid_x(masks.get('metatarsal_5'))
    heel = _mask_centroid_x(masks.get('calcaneus_inferior'))
    if fore is None or heel is None:
        return None
    return fore < heel


def extract_inferior_contour(mask, smooth_window=8, bin_size=3):
    """Lower envelope of the calcaneus mask contour: per X-bin inferior-most point."""
    c = _largest_contour((mask > 0).astype(np.uint8) * 255)
    if c is None or len(c) < 5:
        return None
    xs, ys = c[:, 0], c[:, 1]
    bins = np.arange(xs.min(), xs.max() + bin_size, bin_size)
    inf = []
    for i in range(len(bins) - 1):
        m = (xs >= bins[i]) & (xs < bins[i + 1])
        if m.any():
            idx = np.where(m)[0]
            inf.append(c[idx[np.argmax(ys[idx])]])
    if len(inf) < 3:
        return None
    inf = np.array(inf)[np.argsort(np.array(inf)[:, 0])]
    if smooth_window > 1 and len(inf) > smooth_window:
        k = np.ones(smooth_window) / smooth_window
        inf = np.stack([
            np.convolve(inf[:, 0], k, 'valid'),
            np.convolve(inf[:, 1], k, 'valid'),
        ], axis=1)
    return inf


def lower_tangent(P4, contour, min_dist=10.0):
    """Tangent point P3 on the inferior contour from pivot P4 (convex-hull method)."""
    P4 = np.asarray(P4, np.float64)
    far = contour[np.linalg.norm(contour - P4, axis=1) >= min_dist]
    if len(far) < 3:
        return None
    hull_idx = cv2.convexHull(far.astype(np.float32), returnPoints=False).reshape(-1)
    hull = far[hull_idx]
    n = len(hull)
    if n < 3:
        return None
    candidates = []
    for i in range(n):
        V, Vp, Vn = hull[i], hull[(i - 1) % n], hull[(i + 1) % n]
        d = V - P4
        if np.linalg.norm(d) < 1e-6:
            continue
        sp = d[0] * (Vp[1] - V[1]) - d[1] * (Vp[0] - V[0])
        sn = d[0] * (Vn[1] - V[1]) - d[1] * (Vn[0] - V[0])
        if sp * sn >= -1e-6:
            candidates.append(V)
    if not candidates:
        return None
    cands = np.array(candidates)
    return cands[np.argmax(cands[:, 1])]


def compute_CIA(P1, P2, P3, P4):
    """CIA = angle between ground line (P1->P2) and tangent line (P4->P3), in [0, 90]."""
    P1, P2, P3, P4 = (np.asarray(p, np.float64) for p in [P1, P2, P3, P4])
    vg = P2 - P1
    vt = P3 - P4
    cos_t = np.dot(vg, vt) / (np.linalg.norm(vg) * np.linalg.norm(vt) + 1e-10)
    angle = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
    return min(angle, 180 - angle)


def compute_from_masks(masks, flipped=None):
    """End-to-end geometry from three binary masks. Returns (cia, P1, P2, P3, P4).

    masks: dict with keys 'calcaneus_inferior', 'metatarsal_5', 'calcaneonavicular'
    (each a uint8/bool array, non-zero = foreground). Any missing landmark -> cia None.
    """
    if flipped is None:
        flipped = detect_flipped(masks)
        if flipped is None:
            flipped = False
    P1 = derive_P1(masks["metatarsal_5"])
    P2 = derive_P2(masks["calcaneus_inferior"])
    P4 = derive_P4(masks["calcaneonavicular"], flipped=flipped)
    P3 = cia = None
    if all(p is not None for p in (P1, P2, P4)):
        contour = extract_inferior_contour(masks["calcaneus_inferior"])
        if contour is not None:
            P3 = lower_tangent(P4, contour)
            if P3 is not None:
                cia = compute_CIA(P1, P2, P3, P4)
    return cia, P1, P2, P3, P4
