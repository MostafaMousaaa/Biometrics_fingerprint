# Contextual Filtering for Fingerprints
# Summary of main equations and steps used:
# 1) Orientation field (per-block):
#    theta = 0.5 * atan2( 2*Gx*Gy, Gx^2 - Gy^2 )
#    where Gx, Gy are image gradients (Sobel).
# 2) Ridge frequency (per-block):
#    - Rotate the block by -theta to align ridges vertically.
#    - Average rows => 1D profile p[n], window and FFT => dominant f in [1/max_wl, 1/min_wl].
# 3) Gabor contextual filter (per-block):
#    g(x,y) = exp( -(x'^2 + (gamma^2) y'^2) / (2 sigma^2) ) * cos( 2*pi*x'/lambda + psi )
#    with (x',y') rotated by local theta, lambda = 1/f, sigma ~ 0.5*lambda.
# 4) Energy map:
#    E = |response| (magnitude of the Gabor response), normalized to [0,255].

import cv2
import numpy as np

def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    # Ensure single-channel 8-bit image for downstream processing.
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _orientation_field(img: np.ndarray, block: int = 16, smooth_ks: int = 9) -> np.ndarray:
    # Estimate local ridge orientation using image gradients and the double-angle formula:
    # theta = 0.5 * atan2( 2*Gx*Gy, Gx^2 - Gy^2 )
    # Block-wise Gaussian smoothing stabilizes the field.
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    h, w = img.shape
    Vx = cv2.GaussianBlur(2 * gx * gy, (smooth_ks, smooth_ks), 0)
    Vy = cv2.GaussianBlur(gx * gx - gy * gy, (smooth_ks, smooth_ks), 0)
    theta = 0.5 * np.arctan2(Vx, Vy)
    # block-wise smoothing
    if block > 1:
        k = (block | 1)
        theta = cv2.GaussianBlur(theta, (k, k), 0)
    return theta.astype(np.float32)

def _ridge_frequency(img: np.ndarray, theta: np.ndarray, block: int = 16, min_wl=4.0, max_wl=16.0) -> np.ndarray:
    # For each block:
    # - Rotate by -theta to align ridges vertically,
    # - Average along rows to form a 1D profile p[n],
    # - Window (Hann) and FFT to locate the dominant frequency within [1/max_wl, 1/min_wl].
    # Returns a cycles/pixel frequency map used to set Gabor wavelength lambda = 1/f.
    h, w = img.shape
    freq = np.zeros((h, w), dtype=np.float32)
    fimg = img.astype(np.float32)
    fmin = 1.0 / float(max_wl)
    fmax = 1.0 / float(min_wl)

    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = fimg[y:y+block, x:x+block]
            ang = float(theta[y, x])  # radians
            deg = -ang * 180.0 / np.pi  # rotate so ridges vertical

            # rotate patch
            M = cv2.getRotationMatrix2D((block * 0.5 - 0.5, block * 0.5 - 0.5), deg, 1.0)
            rot = cv2.warpAffine(patch, M, (block, block), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

            # 1D profile across ridges (x-axis), zero-mean + window
            prof = rot.mean(axis=0)
            prof = prof - prof.mean()
            if np.allclose(prof, 0.0):
                f = 0.0
            else:
                win = np.hanning(len(prof))
                sig = prof * win
                spec = np.fft.rfft(sig)
                amp = np.abs(spec)
                freqs = np.fft.rfftfreq(len(prof), d=1.0)  # cycles/pixel

                # mask allowed band
                band = (freqs >= fmin) & (freqs <= fmax)
                band[0] = False  # exclude DC
                if not np.any(band):
                    f = 0.0
                else:
                    k = np.argmax(amp[band])
                    f = float(freqs[band][k])

            freq[y:y+block, x:x+block] = f

    return freq

def _contextual_gabor(img: np.ndarray, theta: np.ndarray, freq: np.ndarray, block: int, gamma: float = 0.5, gain: float = 1.0):
    # Apply an oriented, frequency-adaptive Gabor filter per block:
    # - Kernel parameters set from local theta and lambda=1/f.
    # - Blend response with original via 'gain' for stability.
    # Also compute energy E = |response| as a quality cue.
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    energy = np.zeros_like(img, dtype=np.float32)
    fimg = img.astype(np.float32) / 255.0
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            ang = float(theta[y, x])
            f = float(freq[y, x])
            if f <= 0:
                blk = fimg[y:y+block, x:x+block]
                out[y:y+block, x:x+block] = blk
                energy[y:y+block, x:x+block] = 0.0
                continue
            wl = 1.0 / f
            sigma = 0.5 * wl
            ksize = int(max(7, int(np.ceil(3 * sigma)) | 1))
            kern = cv2.getGaborKernel((ksize, ksize), sigma=sigma, theta=ang, lambd=wl, gamma=gamma, psi=0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(fimg[y:y+block, x:x+block], cv2.CV_32F, kern)
            out[y:y+block, x:x+block] = (1.0 - gain) * fimg[y:y+block, x:x+block] + gain * resp
            energy[y:y+block, x:x+block] = np.abs(resp)
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
    out_u8 = (out * 255.0).astype(np.uint8)
    energy_u8 = (energy * 255.0).astype(np.uint8)
    return out_u8, energy_u8

def preprocess_image(
    img: np.ndarray,
    apply: bool = True,
    block: int = 16,
    min_wl: float = 4.0,
    max_wl: float = 16.0,
    gamma: float = 0.5,
    gain: float = 1.0,
    return_maps: bool = False,
):
    # Pipeline:
    # 1) Ensure grayscale 8-bit.
    # 2) Estimate orientation field and ridge frequency.
    # 3) Apply contextual Gabor filter and compute energy map.
    # 4) Optionally return diagnostic maps for visualization.
    x = ensure_gray_u8(img)
    if not apply:
        if return_maps:
            return x, None, None, None
        return x
    theta = _orientation_field(x, block=block)
    freq = _ridge_frequency(x, theta, block=block, min_wl=min_wl, max_wl=max_wl)
    y, energy = _contextual_gabor(x, theta, freq, block=block, gamma=gamma, gain=gain)
    if return_maps:
        return y, energy, freq, theta
    return y

def visualize_contextual_maps(energy: np.ndarray, freq: np.ndarray, theta: np.ndarray):
    # Diagnostic visualization (separate figure):
    # - (b) Energy map with colorbar.
    # - (d) Orientation map in degrees [0..180] with colorbar.
    # Note: Frequency map is intentionally omitted per user request.
    import matplotlib.pyplot as plt
    if energy is None or theta is None:
        return

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # (b) Energy map with colorbar
    im0 = axs[0].imshow(energy, cmap='Reds', interpolation='nearest')
    axs[0].set_title("(b) Energy map")
    axs[0].axis('off')
    cb0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    cb0.ax.set_ylabel("Energy", rotation=270, labelpad=12)

    # (d) Orientation map as colored image with colorbar (0..180 deg)
    deg = (theta + np.pi/2.0) * (180.0/np.pi)  # [-90,90] -> [0,180]
    im1 = axs[1].imshow(deg, cmap='hsv', vmin=0.0, vmax=180.0, interpolation='nearest')
    axs[1].set_title("(d) Orientation map")
    axs[1].axis('off')
    cb1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cb1.ax.set_ylabel("degrees", rotation=270, labelpad=12)

    fig.tight_layout()
    try:
        plt.show(block=False)
    except TypeError:
        plt.show()
