"""
Step 2 of the deblurring pipeline: the inverse filter.

Loads the blurred images written by degrade.py, rebuilds the *same* kernel that
produced them, and restores by dividing in the Fourier domain:

    Fhat(u,v) = G(u,v) / H(u,v)

The kernel is not stored on disk.  It is fully determined by sigma, so this
module imports gaussian_kernel() from degrade.py and calls it with the same
sigma -- the restoration therefore uses the identical h(x,y) by construction.

Because H decays exponentially for a Gaussian, |H| becomes numerically tiny at
high frequencies.  Dividing by it amplifies whatever small perturbations are
present, so |H| is floored at epsilon:

    Heps = H          where |H| >= epsilon
         = epsilon    where |H| <  epsilon

Usage:
    python3 deblurring.py --sigma 2.0
    python3 deblurring.py --sigma 2.0 --eps-sweep
    python3 deblurring.py --sigma 2.0 --source memory   # skip 8-bit round trip
"""

import argparse
import time
from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# the kernel and I/O come straight from the degradation stage
from degrade import (DATA_DIR, OUT_DIR, blur_image, gaussian_kernel,
                     list_images, load_image, psf_to_otf, save_image)

HERE = Path(__file__).resolve().parent

# one folder per restoration method; each run overwrites its own folder
DEBLURRED_DIR = HERE / "deblurred-imgs"

# side-by-side comparisons across every method
COMBINED_DIR = HERE / "Combined"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def scores(restored, sharp):
    r = np.clip(restored, 0.0, 1.0)
    return (psnr_fn(sharp, r, data_range=1.0),
            ssim_fn(sharp, r, data_range=1.0, channel_axis=2))


# ---------------------------------------------------------------------------
# the inverse filter
# ---------------------------------------------------------------------------

def inverse_filter(blurry, psf, eps=1e-2):
    """
    Fhat = G / H, with |H| floored at eps so the division cannot blow up.

    Applied independently to each colour channel.
    """
    H = psf_to_otf(psf, blurry.shape[:2])
    H_eps = np.where(np.abs(H) < eps, eps, H)

    out = np.empty_like(blurry)
    for c in range(blurry.shape[2]):
        out[:, :, c] = np.real(ifft2(fft2(blurry[:, :, c]) / H_eps))
    return out


# ---------------------------------------------------------------------------
# pair loading
# ---------------------------------------------------------------------------

def load_pairs(sigma, source="disk", limit=None):
    """
    Yield (name, blurry, sharp) triples.

    source="disk"   : read the PNGs degrade.py wrote (8-bit quantised)
    source="memory" : re-blur the sharp images now, keeping full float precision
    """
    sharp_paths = list_images(DATA_DIR)
    if limit:
        sharp_paths = sharp_paths[:limit]

    if source == "memory":
        psf = gaussian_kernel(sigma)
        for p in sharp_paths:
            sharp = load_image(p)
            yield p.name, blur_image(sharp, psf), sharp
        return

    blur_dir = Path(OUT_DIR) / f"sigma_{sigma:g}"
    if not blur_dir.is_dir():
        raise SystemExit(f"{blur_dir} not found -- run:  "
                         f"python3 degrade.py --sigma {sigma:g}")
    for p in sharp_paths:
        bp = blur_dir / p.name
        if bp.exists():
            yield p.name, load_image(bp), load_image(p)


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# combined side-by-side output
# ---------------------------------------------------------------------------

BAR = 26        # height of the caption strip above each panel


def _to_pil(img):
    return Image.fromarray(
        np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))


def save_combined(panels, path):
    """
    Write one horizontal strip: panels is a list of (label, image, psnr, ssim)
    where psnr/ssim may be None for the ground-truth column.
    """
    pils = [_to_pil(im) for _, im, _, _ in panels]
    h = max(p.height for p in pils)
    w = sum(p.width for p in pils)

    canvas = Image.new("RGB", (w, h + BAR), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for (label, _, p, s), pil in zip(panels, pils):
        canvas.paste(pil, (x, BAR))
        cap = label if p is None else f"{label}  {p:.2f}dB / {s:.3f}"
        draw.text((x + 4, 7), cap, fill="black")
        x += pil.width
    canvas.save(path)


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def run(sigma, source="disk", eps=1e-2, limit=None, save=True, combined=True):
    psf = gaussian_kernel(sigma)
    print(f"sigma = {sigma:g}   kernel {psf.shape[0]}x{psf.shape[1]}   "
          f"eps = {eps:g}   source = {source}\n")

    dest = None
    if save:
        dest = DEBLURRED_DIR / "inverse"
        dest.mkdir(parents=True, exist_ok=True)
    comb = None
    if combined:
        comb = COMBINED_DIR
        comb.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, blurry, sharp in load_pairs(sigma, source, limit):
        p_in, s_in = scores(blurry, sharp)             # blurry vs sharp
        t0 = time.perf_counter()
        rec = inverse_filter(blurry, psf, eps)
        dt = time.perf_counter() - t0
        p_out, s_out = scores(rec, sharp)              # restored vs sharp
        rows.append((p_in, s_in, p_out, s_out, dt))

        if dest:
            save_image(rec, dest / name)
        if comb:
            save_combined([
                ("blurry",  blurry, p_in,  s_in),
                ("inverse", rec,    p_out, s_out),
                ("sharp",   sharp,  None,  None),
            ], comb / name)

    a = np.array(rows)
    n = len(rows)
    print(f"{'':22} {'PSNR':>8} {'SSIM':>8}")
    print(f"{'blurry input':22} {a[:,0].mean():8.2f} {a[:,1].mean():8.4f}")
    print(f"{'inverse filter':22} {a[:,2].mean():8.2f} {a[:,3].mean():8.4f}")
    d_psnr = a[:, 2].mean() - a[:, 0].mean()
    print(f"{'change':22} {d_psnr:+8.2f} {a[:,3].mean()-a[:,1].mean():+8.4f}")
    print(f"\n{n} images, {a[:,4].mean()*1000:.0f} ms per image")
    if dest:
        print(f"deblurred -> {dest}")
    if comb:
        print(f"combined  -> {comb}")
    return a


def eps_sweep(sigma, source="disk", limit=None):
    psf = gaussian_kernel(sigma)
    pairs = list(load_pairs(sigma, source, limit))
    base = np.array([scores(b, s) for _, b, s in pairs])
    print(f"sigma = {sigma:g}   source = {source}   {len(pairs)} images\n")
    print(f"{'eps':>10} {'PSNR':>8} {'SSIM':>8}")
    print(f"{'(blurry)':>10} {base[:,0].mean():8.2f} {base[:,1].mean():8.4f}")
    for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-10]:
        r = np.array([scores(inverse_filter(b, psf, eps), s)
                      for _, b, s in pairs])
        print(f"{eps:10.0e} {r[:,0].mean():8.2f} {r[:,1].mean():8.4f}")


def main():
    ap = argparse.ArgumentParser(description="Inverse-filter deblurring.")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--source", choices=["disk", "memory"], default="disk")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eps-sweep", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    if args.eps_sweep:
        eps_sweep(args.sigma, args.source, args.limit)
    else:
        run(args.sigma, args.source, args.eps, args.limit,
            save=not args.no_save, combined=not args.no_combined)


if __name__ == "__main__":
    main()
