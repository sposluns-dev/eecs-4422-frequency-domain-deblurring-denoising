"""
Step 1 of the deblurring pipeline: load a CBSD68 image and blur it with a known
Gaussian kernel.

No noise, no restoration filters -- this module only produces the (blurry, sharp,
kernel) triple that later stages consume.

The blur is applied as a *circular* convolution in the Fourier domain, i.e.

    G = H . F        ->        g = ifft2( H * fft2(f) )

so the degradation matches the restoration model exactly.  A spatial convolution
with zero or reflected padding would introduce boundary effects that no
frequency-domain filter can undo, which would contaminate the comparison.

Usage:
    python3 degrade.py --sigma 2.0             # blur all 68 images, write to blurred/
    python3 degrade.py --sigma 2.0 --limit 5   # just the first 5
    python3 degrade.py --test                  # run correctness checks instead
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image

HERE = Path(__file__).resolve().parent

# master copy of the dataset, one level up from this folder
DATA_DIR = HERE.parent / "CBSD68"

# blurred output lands beside this script
OUT_DIR = HERE / "blurred"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_image(path):
    """Read an image as float64 RGB in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def list_images(data_dir=DATA_DIR):
    """All CBSD68 image paths, sorted."""
    return sorted(Path(data_dir).glob("*.png"))


# ---------------------------------------------------------------------------
# blur kernel
# ---------------------------------------------------------------------------

def gaussian_kernel(sigma, size=None):
    """
    Normalised 2-D Gaussian PSF.

        h(x, y) = 1 / (2 pi sigma^2) * exp( -(x^2 + y^2) / (2 sigma^2) )

    `size` defaults to an odd window wide enough to hold the kernel
    (+/- 3 sigma), so truncation is negligible.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    if size % 2 == 0:
        size += 1

    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()          # unit sum -> blur preserves mean brightness


def psf_to_otf(psf, shape):
    """
    Zero-pad a PSF to `shape`, wrap it so its centre sits at index (0, 0), and
    transform.  The roll is what makes the convolution zero-phase; without it
    the result comes out shifted by half the kernel width.
    """
    pad = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, -(kh // 2), axis=0)
    pad = np.roll(pad, -(kw // 2), axis=1)
    return fft2(pad)


# ---------------------------------------------------------------------------
# degradation
# ---------------------------------------------------------------------------

def blur_image(sharp, psf):
    """
    Circular convolution of `sharp` with `psf`, done per colour channel in the
    Fourier domain.  Returns a float image in [0, 1].
    """
    H = psf_to_otf(psf, sharp.shape[:2])
    out = np.empty_like(sharp)
    for c in range(sharp.shape[2]):
        out[:, :, c] = np.real(ifft2(H * fft2(sharp[:, :, c])))
    return np.clip(out, 0.0, 1.0)


def degrade(path, sigma):
    """
    Load the image at `path` and blur it with a Gaussian of width `sigma`.

    Returns (blurry, sharp, psf).
    """
    sharp = load_image(path)
    psf = gaussian_kernel(sigma)
    return blur_image(sharp, psf), sharp, psf


# ---------------------------------------------------------------------------
# writing blurred images to disk
# ---------------------------------------------------------------------------

def save_image(img, path):
    """Write a float image in [0, 1] as an 8-bit PNG."""
    arr = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def write_blurred(sigma, limit=None, data_dir=DATA_DIR, out_dir=OUT_DIR):
    """
    Blur every image in `data_dir` with a Gaussian of width `sigma` and write the
    results to  out_dir/sigma_<sigma>/ .  Returns the output directory.
    """
    paths = list_images(data_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images found in {data_dir}")

    dest = Path(out_dir) / f"sigma_{sigma:g}"
    dest.mkdir(parents=True, exist_ok=True)

    psf = gaussian_kernel(sigma)
    print(f"sigma = {sigma:g}   kernel {psf.shape[0]}x{psf.shape[1]}   "
          f"sum = {psf.sum():.6f}")
    print(f"source: {data_dir}")
    print(f"dest:   {dest}\n")

    for n, p in enumerate(paths, 1):
        sharp = load_image(p)
        save_image(blur_image(sharp, psf), dest / p.name)
        if n % 17 == 0 or n == len(paths):
            print(f"  {n}/{len(paths)}")

    total = sum(f.stat().st_size for f in dest.glob("*.png"))
    print(f"\nwrote {len(paths)} images  ({total/1e6:.1f} MB)")
    return dest


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _self_test():
    paths = list_images()
    print(f"dataset: {DATA_DIR}")
    print(f"images found: {len(paths)}")
    if not paths:
        raise SystemExit("no images found -- check DATA_DIR")

    sharp = load_image(paths[0])
    print(f"sample: {paths[0].name}  shape={sharp.shape}  "
          f"range=[{sharp.min():.3f}, {sharp.max():.3f}]")

    print(f"\n{'sigma':>6} {'ksize':>6} {'ksum':>8} {'mean drift':>11} "
          f"{'grad ratio':>11} {'invertible':>11}")
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        psf = gaussian_kernel(sigma)
        blurry = blur_image(sharp, psf)

        # 1. kernel is normalised
        ksum = psf.sum()

        # 2. blurring must not shift overall brightness
        drift = abs(blurry.mean() - sharp.mean())

        # 3. blurring must reduce high-frequency content
        def grad_energy(im):
            gy, gx = np.gradient(im.mean(axis=2))
            return float(np.mean(gx ** 2 + gy ** 2))
        ratio = grad_energy(blurry) / grad_energy(sharp)

        # 4. with no noise, dividing by H must recover the original almost
        #    exactly -- this confirms degradation and transfer function agree
        H = psf_to_otf(psf, sharp.shape[:2])
        rec = np.empty_like(sharp)
        unclipped = np.empty_like(sharp)
        for c in range(3):
            unclipped[:, :, c] = np.real(ifft2(H * fft2(sharp[:, :, c])))
        for c in range(3):
            rec[:, :, c] = np.real(ifft2(fft2(unclipped[:, :, c]) / H))
        err = float(np.abs(rec - sharp).max())

        print(f"{sigma:6.1f} {psf.shape[0]:6d} {ksum:8.5f} {drift:11.2e} "
              f"{ratio:11.4f} {err:11.2e}")

    print("\nchecks: ksum == 1, mean drift ~ 0, grad ratio < 1 and falling,")
    print("invertible (max abs error of noiseless H-division) ~ 0")


def main():
    ap = argparse.ArgumentParser(
        description="Blur CBSD68 images with a known Gaussian kernel.")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Gaussian blur width in pixels (default 2.0)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N images")
    ap.add_argument("--test", action="store_true",
                    help="run correctness checks instead of writing images")
    args = ap.parse_args()

    if args.test:
        _self_test()
    else:
        write_blurred(args.sigma, args.limit)


if __name__ == "__main__":
    main()
