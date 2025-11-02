# Chain Code Representation (Freeman codes)
# Summary:
# - Represent a contour as a sequence of unit steps in 8 directions (codes 0..7).
# - Each code encodes direction; differences between successive codes (mod 8) encode turns.
# - From the sequence we derive: length (number of steps), direction histogram, turn histogram.
#   Turn step -> signed angle step of 45° increments for curvature profiling.

import cv2
import numpy as np
from typing import Dict, List

def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    # Ensure single-channel uint8 input to thresholding and contour extraction.
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _binarize(img: np.ndarray, use_otsu: bool, manual_thresh: int) -> np.ndarray:
    # Produce a binary foreground mask (fingerprint ink/ridges as foreground).
    # - Otsu threshold by default (inverse so ridges become 1).
    # - Light morphological opening to remove specks.
    if use_otsu:
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(img, manual_thresh, 255, cv2.THRESH_BINARY_INV)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return th

def _largest_contour(bin_img: np.ndarray):
    # Utility to obtain the largest external contour (if needed by callers).
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def _chain_code(contour, eight_conn: bool = True):
    # Convert a dense contour (Nx1x2) into a Freeman chain code:
    # - Map unit steps between consecutive points to direction codes 0..7:
    #   0:(+1,0), 1:(+1,-1), 2:(0,-1), 3:(-1,-1), 4:(-1,0), 5:(-1,+1), 6:(0,+1), 7:(+1,+1)
    # - If 4-connectivity requested, diagonals are snapped to nearest axis directions {0,2,4,6}.
    # contour: Nx1x2
    pts = contour.reshape(-1, 2).astype(np.int32)
    if pts.shape[0] < 2:
        return []
    # Freeman 8-direction mapping in image coords (y down)
    dir_map = {
        (1, 0): 0,  (1, -1): 1,  (0, -1): 2,  (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5,  (0, 1): 6,   (1, 1): 7
    }
    # For 4-conn, map diagonals to nearest 4-direction
    dir_map4 = {(1, 0): 0, (0, -1): 2, (-1, 0): 4, (0, 1): 6}

    codes = []
    for i in range(1, pts.shape[0]):
        dx = int(pts[i, 0] - pts[i - 1, 0])
        dy = int(pts[i, 1] - pts[i - 1, 1])
        # normalize to unit step if needed
        dx = np.clip(dx, -1, 1); dy = np.clip(dy, -1, 1)
        if dx == 0 and dy == 0:
            continue
        if eight_conn:
            code = dir_map.get((dx, dy), None)
            if code is None:
                continue
        else:
            # snap diagonal to nearest axis
            if abs(dx) >= abs(dy):
                code = dir_map4[(np.sign(dx), 0)]
            else:
                code = dir_map4[(0, np.sign(dy))]
        codes.append(int(code))
    return codes

def extract_features(img: np.ndarray,
                     use_otsu: bool = True,
                     manual_thresh: int = 128,
                     eight_conn: bool = True) -> Dict[str, float]:
    # Outputs:
    # - CC_len: total number of steps in the main chain (proxy for contour length).
    # - CC_hist_0..7: normalized histogram of direction codes.
    # - CC_turn_hist_0..7: normalized histogram of turn codes Δdir mod 8.
    g = ensure_gray_u8(img)
    bin_img = _binarize(g, use_otsu=use_otsu, manual_thresh=manual_thresh)
    cnt = _largest_contour(bin_img)
    feats: Dict[str, float] = {}
    if cnt is None:
        # no contour found
        feats["CC_len"] = 0.0
        for i in range(8):
            feats[f"CC_hist_{i}"] = 0.0
            feats[f"CC_turn_hist_{i}"] = 0.0
        return feats

    codes = _chain_code(cnt, eight_conn=eight_conn)
    n = len(codes)
    feats["CC_len"] = float(n)

    if n == 0:
        for i in range(8):
            feats[f"CC_hist_{i}"] = 0.0
            feats[f"CC_turn_hist_{i}"] = 0.0
        return feats

    # direction histogram
    hist = np.bincount(np.array(codes, dtype=np.int32), minlength=8).astype(np.float32)
    hist /= (hist.sum() + 1e-9)
    for i in range(8):
        feats[f"CC_hist_{i}"] = float(hist[i])

    # turning histogram (delta code modulo 8)
    turns = (np.diff(np.array(codes, dtype=np.int32), prepend=codes[0]) + 8) % 8
    th = np.bincount(turns, minlength=8).astype(np.float32)
    th /= (th.sum() + 1e-9)
    for i in range(8):
        feats[f"CC_turn_hist_{i}"] = float(th[i])

    return feats

def get_chain_code_details(img: np.ndarray,
                           use_otsu: bool = True,
                           manual_thresh: int = 128,
                           eight_conn: bool = True) -> Dict:
    # Return all data needed to visualize Sharat Chikkerur-style chain code contours:
    # - gray: original grayscale
    # - binary: thresholded mask used for edge detection
    # - contours: list of contours (each Nx2) used for arrow overlays
    # - codes_list: aligned chain-code sequences for each contour
    # Curvature profile is derived from turn steps t via signed 45° increments:
    #   signed_turn = wrap_to([-4..+3], t), curvature_deg = cumsum( signed_turn * 45° )
    g = ensure_gray_u8(img)
    bin_img = _binarize(g, use_otsu=use_otsu, manual_thresh=manual_thresh)

    # Edge map to capture many ridge contours (not only the outer boundary)
    edges = cv2.Canny(bin_img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Filter tiny contours and limit count for clarity
    filtered = []
    codes_list: List[List[int]] = []
    for c in contours:
        if c.shape[0] < 15:
            continue
        cc = _chain_code(c, eight_conn=eight_conn)
        if len(cc) < 10:
            continue
        filtered.append(c.reshape(-1, 2))
        codes_list.append(cc)

    return {
        "gray": g,
        "binary": bin_img,
        "edges": edges,
        "contours": filtered,     # list of (N,2) arrays
        "codes_list": codes_list  # list of chain-code sequences aligned with contours
    }

def visualize_chain_code(details: Dict):
    # Minimal visualization: overlay sparse red arrows along contours to depict direction flow.
    # No extra plots to keep the figure focused (as requested).
    import matplotlib.pyplot as plt

    gray = details.get("gray")
    contours = details.get("contours", [])
    codes_list = details.get("codes_list", [])

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    ax.imshow(gray, cmap='gray')
    ax.set_title("Chain Code Contours")
    ax.axis('off')

    # Freeman 8-direction unit vectors in image coords (x right, y down)
    vec = {
        0: (1, 0), 1: (1, -1), 2: (0, -1), 3: (-1, -1),
        4: (-1, 0), 5: (-1, 1), 6: (0, 1), 7: (1, 1)
    }

    # Draw for each contour: small red arrows with sparse sampling
    for cnt, codes in zip(contours, codes_list):
        if cnt.shape[0] <= 1:
            continue
        # sample every s points to avoid clutter
        s = 4
        bases_x, bases_y, U, V = [], [], [], []
        pts = cnt
        # Codes are steps between successive points; align lengths
        steps = min(len(codes), pts.shape[0] - 1)
        for i in range(0, steps, s):
            x0, y0 = pts[i, 0], pts[i, 1]
            dx, dy = vec.get(codes[i], (0, 0))
            bases_x.append(x0)
            bases_y.append(y0)
            # small arrow length
            U.append(dx * 2.0)
            V.append(dy * 2.0)

        # scatter small red circles
        ax.scatter(bases_x, bases_y, s=10, facecolors='none', edgecolors='r', linewidths=0.8)
        # arrows
        if len(bases_x):
            ax.quiver(bases_x, bases_y, U, V, angles='xy', scale_units='xy', scale=1.0,
                      color='r', width=0.003, headwidth=3, headlength=4)

    fig.tight_layout()
    try:
        plt.show(block=False)
    except TypeError:
        plt.show()
