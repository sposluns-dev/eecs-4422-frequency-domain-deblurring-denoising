"""
Frequency-domain deblurring: degradation and restoration in one file.

Self-contained -- no imports from sibling modules.

Pipeline
--------
    1. load a clean CBSD68 image
    2. degrade it: blur with a known Gaussian, then add white Gaussian noise
    3. restore in the Fourier domain            (write to deblurred-imgs/)
    4. write side-by-side comparisons           (write to Combined/)

The degradation has two parameters: sigma, the width of the blur, and sigma_n,
the standard deviation of the noise.  The Gaussian's transfer function H(u,v) is
built directly from sigma, and the same H is used both to blur and to restore,
so the degradation matches the restoration model exactly:

    F = H . G + N    ->    f = ifft2( H * fft2(g) ) + n,   n ~ N(0, sigma_n^2)

The blur is therefore a *circular* convolution.  Zero- or reflection-padded
convolution would introduce boundary behaviour that no frequency-domain filter
can undo.

Degradation happens in memory, in float64, and the restoration filters are given
that float array directly.  Nothing is quantised on the way in, so the only
perturbation present is the noise the model says is there -- sigma_n is the
whole of n(x,y).  The PNGs written to blurred/ are for the report's figures.

Noise levels are given on the 0-255 scale, as the denoising literature quotes
them (sigma_n = 15, 25, 50 are the usual CBSD68 operating points); internally
they are divided by 255 to match the [0, 1] float images.

Usage
-----
    python3 degrade-deblur.py blur                      # write degraded images
    python3 degrade-deblur.py deblur                    # degrade -> restore
    python3 degrade-deblur.py deblur --sigma-n 25
    python3 degrade-deblur.py deblur --eps 0.1
    python3 degrade-deblur.py sweep                     # epsilon sweep
    python3 degrade-deblur.py test                      # correctness checks

Common options:  --sigma 2.0   --sigma-n 15   --limit N   --no-save
                 --no-combined
"""

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

HERE = Path(__file__).resolve().parent

DATA_DIR      = HERE.parent / "CBSD68"      # clean reference images
BLURRED_DIR   = HERE / "blurred"            # synthetically blurred inputs
DEBLURRED_DIR = HERE / "deblurred-imgs"     # one subfolder per method
COMBINED_DIR  = HERE / "Combined"           # side-by-side strips

# Noise is quoted on the 0-255 scale in the report and divided by 255 here.
SIGMA_N_DEFAULT = 15.0                      # the usual CBSD68 operating point
NOISE_SEED      = 4422                      # fixed, so every run is reproducible


# ===========================================================================
# I/O
# ===========================================================================

def load_image(path):
    """Read an image as float64 RGB in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def save_image(img, path):
    """Write a float image in [0, 1] as an 8-bit PNG."""
    arr = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def list_images(data_dir=DATA_DIR):
    """All clean image paths, sorted."""
    return sorted(Path(data_dir).glob("*.png"))


def blurred_dir(sigma, sigma_n):
    """Where the degraded PNGs for one (blur, noise) setting live."""
    return BLURRED_DIR / f"sigma_{sigma:g}_n{sigma_n * 255:g}"


# ===========================================================================
# the Gaussian blur, as a transfer function
# ===========================================================================

def kernel_size(sigma):
    """Odd window width holding the Gaussian out to +/- 3 sigma."""
    return int(2 * np.ceil(3 * sigma) + 1)


def gaussian_kernel(sigma):
    """The isotropic Gaussian kernel on its own small odd window, unit sum."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    r = kernel_size(sigma) // 2
    off = np.arange(-r, r + 1)              # -r ... 0 ... +r

    g = np.exp(-off ** 2 / (2.0 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def gaussian_h(sigma, shape):
    """
    The Gaussian blur kernel h, laid out on an image-sized grid.

    The kernel is written straight onto the full-size grid, centred on index
    (0,0).  Offsets run -r..+r and are placed with wraparound (`% shape`), which
    is what a periodic image means -- so the kernel is already positioned
    correctly and nothing needs shifting afterwards.

        h[0, 0]   = centre of the Gaussian
        h[-1, -1] = the pixel diagonally "before" it, i.e. bottom-right corner

    Normalising to unit sum means the blur preserves brightness (and, once
    transformed, gives H(0,0) = 1).

    Returns the spatial kernel; callers apply fft2() to obtain H.
    """
    k = gaussian_kernel(sigma)
    r = k.shape[0] // 2
    off = np.arange(-r, r + 1)              # -r ... 0 ... +r

    h = np.zeros(shape, dtype=np.float64)
    h[np.ix_(off % shape[0], off % shape[1])] = k
    return h


"""
Naming convention used throughout, following the course notes:

    lower case = spatial domain (a grid of pixels)
    UPPER case = frequency domain (its 2-D Fourier transform)

    g, G          original (sharp) image
    f, F          degraded observation:  f = h * g + n
    h, H          blur kernel / transfer function
    n, N          noise
    w, W          restoration filter
    g_hat, G_hat  restored estimate of the original image
"""


# ===========================================================================
# degradation
# ===========================================================================

def image_rng(name):
    """
    The noise generator for one image, seeded from its file name.

    Every stage that needs the degraded version of an image draws its noise from
    here, so the array the filters see, the PNG written to blurred/, and the
    panel in the report are all the same realisation of n.
    """
    return np.random.default_rng(NOISE_SEED + zlib.crc32(str(name).encode()))


def blur_image(sharp, sigma, sigma_n=0.0, rng=None):
    """
    Degrade `sharp`: blur with a Gaussian of width `sigma`, then add white
    Gaussian noise of standard deviation `sigma_n`.

        F = H . G + N     ->     f = ifft2(H . G) + n

    `sigma_n` is in [0, 1] image units; divide the 0-255 figure by 255 first.

    The blurred image is clipped to [0, 1] -- a non-negative unit-sum kernel
    cannot leave that range, so this only removes float round-off -- and the
    noise is added afterwards and left unclipped, so n stays exactly Gaussian
    and the degradation is exactly the model the filters are derived from.
    """
    h = gaussian_h(sigma, sharp.shape[:2])      # kernel, pixels
    H = fft2(h)                                 # kernel, frequencies

    out = np.empty_like(sharp)
    for c in range(sharp.shape[2]):
        g = sharp[:, :, c]                      # original channel, pixels
        G = fft2(g)                             # apply the Fourier transform
        F = H * G                               # apply the blur
        f = np.real(ifft2(F))                   # -> pixels
        out[:, :, c] = f

    out = np.clip(out, 0.0, 1.0)

    if sigma_n > 0.0:
        if rng is None:
            rng = np.random.default_rng(NOISE_SEED)
        out = out + rng.normal(0.0, sigma_n, size=out.shape)   # adding n(x,y)

    return out


def degrade(path, sigma, sigma_n=0.0):
    """Load the image at `path` and degrade it.  Returns (degraded, sharp)."""
    sharp = load_image(path)
    return blur_image(sharp, sigma, sigma_n, image_rng(Path(path).name)), sharp


def write_blurred(sigma, sigma_n, limit=None):
    """
    Degrade every clean image and write to blurred/sigma_<sigma>_n<sigma_n>/.

    These PNGs are what the report's figures display; the restoration itself
    never reads them back, so the 8-bit rounding here costs nothing.
    """
    paths = list_images()
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images found in {DATA_DIR}")

    dest = blurred_dir(sigma, sigma_n)
    dest.mkdir(parents=True, exist_ok=True)

    n_k = kernel_size(sigma)
    print(f"sigma = {sigma:g}   kernel {n_k}x{n_k}   "
          f"sigma_n = {sigma_n * 255:g}/255")
    print(f"source: {DATA_DIR}")
    print(f"dest:   {dest}\n")

    for n, p in enumerate(paths, 1):
        save_image(degrade(p, sigma, sigma_n)[0], dest / p.name)
        if n % 17 == 0 or n == len(paths):
            print(f"  {n}/{len(paths)}")

    total = sum(f.stat().st_size for f in dest.glob("*.png"))
    print(f"\nwrote {len(paths)} images  ({total/1e6:.1f} MB)")
    return dest


# ===========================================================================
# metrics
# ===========================================================================

def scores(restored, sharp):
    r = np.clip(restored, 0.0, 1.0)
    return (psnr_fn(sharp, r, data_range=1.0),
            ssim_fn(sharp, r, data_range=1.0, channel_axis=2))


# ===========================================================================
# restoration -- one function per method
# ===========================================================================

def inverse_filter(blurry, sigma, eps=1e-2):
    """
    Ghat = F / H, with |H| floored at eps so the division cannot blow up.

    H decays exponentially for a Gaussian, so at high frequencies it is
    numerically tiny; dividing by it amplifies whatever perturbation is present
    rather than recovering detail.
    """
    h = gaussian_h(sigma, blurry.shape[:2])     # kernel, pixels
    H = fft2(h)                                 # kernel, frequencies
    H_eps = np.where(np.abs(H) < eps, eps, H)   # floored, avoid divide by zero

    out = np.empty_like(blurry)
    for c in range(blurry.shape[2]):
        f = blurry[:, :, c]                     # degraded channel, pixels
        F = fft2(f)                             # frequencies
        G_hat = F / H_eps                       # divide out the blur
        g_hat = np.real(ifft2(G_hat))           # pixels
        out[:, :, c] = g_hat

    return out #deblurred image


def wiener_filter(blurry, sigma, K=None, sigma_n=0.0):
    """
    Ghat = conj(H) F / (|H|^2 + K).

    Same division as the inverse filter, but the denominator can never reach
    zero: where |H| is large the factor tends to 1/H, and where |H| is small it
    tends to conj(H)/K, which rolls smoothly to zero instead of exploding.
    K = P_n / P_g, the noise-to-signal power ratio, taken here to be constant.

    Left to itself, K is estimated per image as sigma_n^2 / var(f): the noise
    level is a property of the sensor and the observed variance is measurable,
    so unlike the oracle below this stays inside what a real system knows.
    """
    if K is None:
        # K = P_n / P_g, but neither PSD is available, so both are replaced by
        # the variance they integrate to -- a variance in pixels and a power
        # spectrum in frequencies are the same energy, by Parseval.
        #
        #   P_n -> sigma_n^2    exact: white noise is flat, so its power at any
        #                       one frequency IS its variance, and collapsing
        #                       the spectrum to a single number loses nothing.
        #
        #   P_g -> var(f)       approximate, in three separate ways.  Natural
        #                       images are far from flat (most of their energy
        #                       sits at low frequencies), var() measures signal
        #                       AND noise so it overstates P_g slightly, and it
        #                       is read off the blurred image rather than the
        #                       sharp one.  This is why the filter under-damps
        #                       low frequencies and over-damps high ones, and
        #                       most of why it trails wiener_filter_oracle.

        K = sigma_n ** 2 / max(float(np.var(blurry)), 1e-12)

    h = gaussian_h(sigma, blurry.shape[:2])     # kernel, pixels
    H = fft2(h)                                 # kernel, frequencies
    W = np.conj(H) / (np.abs(H) ** 2 + K)       # the restoration filter

    out = np.empty_like(blurry)
    for c in range(blurry.shape[2]):
        f = blurry[:, :, c]                     # degraded channel, pixels
        F = fft2(f)                             # frequencies
        G_hat = W * F                           # apply the filter
        g_hat = np.real(ifft2(G_hat))           # pixels
        out[:, :, c] = g_hat

    return out


def wiener_filter_oracle(blurry, sigma, sharp, sigma_n=0.0):
    """
    Ghat = conj(H) P_g F / (|H|^2 P_g + P_n), using the true PSDs.

    Instead of collapsing P_n / P_g into a single constant K, this measures both
    per frequency:

        P_g = |G|^2           from the ground-truth image
        P_n = M N sigma_n^2   flat, because the added noise is white

    P_g is the PSD of the image being recovered, so it is not available to a real
    restoration system -- this is an ORACLE, and its score is an upper bound on
    what the Wiener filter can achieve, not an achievable result.
    """
    h = gaussian_h(sigma, blurry.shape[:2])     # kernel, pixels
    H = fft2(h)                                 # kernel, frequencies

    M, N = blurry.shape[:2]
    # white noise -> flat PSD; floored so the sigma_n = 0 case stays finite
    P_n = max(M * N * sigma_n ** 2, 1e-12)

    out = np.empty_like(blurry)
    for c in range(blurry.shape[2]):
        f = blurry[:, :, c]                     # degraded channel, pixels
        F = fft2(f)                             # frequencies
        G = fft2(sharp[:, :, c])                # ground truth, frequencies
        P_g = np.abs(G) ** 2                    # true PSD, per frequency

        W = np.conj(H) * P_g / (np.abs(H) ** 2 * P_g + P_n)
        G_hat = W * F                           # apply the filter
        g_hat = np.real(ifft2(G_hat))           # pixels
        out[:, :, c] = g_hat

    return out


def laplacian_p(shape):
    """
    The 3x3 Laplacian operator p, laid out on an image-sized grid.

        p = [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]]

    Placed centred on index (0,0) with wraparound, exactly as gaussian_h() does,
    so that fft2(p) is the transfer function of the roughness measure.
    """
    off = np.arange(-1, 2)                      # -1, 0, +1
    k = np.array([[0.0,  1.0, 0.0],
                  [1.0, -4.0, 1.0],
                  [0.0,  1.0, 0.0]])

    p = np.zeros(shape, dtype=np.float64)
    p[np.ix_(off % shape[0], off % shape[1])] = k
    return p


# gamma tracks the noise-to-signal ratio, but the Laplacian's |P|^2 is much
# larger than 1 over most of the spectrum, so the constant that balances them is
# not 1.  This multiplier is the one that maximised PSNR across sigma_n = 5..50.
GAMMA_OVER_NSR = 8.0


def cls_laplacian_filter(blurry, sigma, gamma=None, sigma_n=0.0):
    """
    Ghat = conj(H) F / (|H|^2 + gamma |P|^2).

    Same shape as the Wiener filter, but the constant K is replaced by
    gamma |P|^2, which grows with frequency because P is a high-pass operator.
    The penalty is therefore weak on smooth content and strong on the high
    frequencies where the inverse filter would otherwise amplify error --- no
    *spectral* noise statistics are required, only a smoothness preference and a
    single scalar setting how hard to press it.

    Left to itself gamma is set to GAMMA_OVER_NSR * sigma_n^2 / var(f), a fixed
    multiple of the same measurable ratio the Wiener filter uses: a standing-in
    for the textbook rule of tuning gamma until the residual matches the known
    noise power, without the iteration.
    """
    if gamma is None:
        gamma = GAMMA_OVER_NSR * sigma_n ** 2 / max(float(np.var(blurry)), 1e-12)

    h = gaussian_h(sigma, blurry.shape[:2])     # blur kernel, pixels
    H = fft2(h)                                 # blur, frequencies
    p = laplacian_p(blurry.shape[:2])           # Laplacian, pixels
    P = fft2(p)                                 # roughness, frequencies

    W = np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)

    out = np.empty_like(blurry)
    for c in range(blurry.shape[2]):
        f = blurry[:, :, c]                     # degraded channel, pixels
        F = fft2(f)                             # frequencies
        G_hat = W * F                           # apply the filter
        g_hat = np.real(ifft2(G_hat))           # pixels
        out[:, :, c] = g_hat

    return out


# ===========================================================================
# pair loading
# ===========================================================================

def load_pairs(sigma, sigma_n, limit=None):
    """
    Yield (name, degraded, sharp) triples.

    The degradation is generated here, in float64, and handed to the filters
    directly: there is no 8-bit round trip anywhere on this path, so the only
    perturbation the filters have to contend with is n itself.  Seeding from the
    file name keeps the realisation identical across stages and across runs.
    """
    sharp_paths = list_images()
    if limit:
        sharp_paths = sharp_paths[:limit]

    for p in sharp_paths:
        sharp = load_image(p)
        yield p.name, blur_image(sharp, sigma, sigma_n, image_rng(p.name)), sharp


# ===========================================================================
# combined side-by-side output
# ===========================================================================

BAR = 26        # height of the caption strip above each panel


def _to_pil(img):
    return Image.fromarray(
        np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))


def save_combined(panels, path):
    """
    Write one horizontal strip.  `panels` is a list of (label, image, psnr, ssim)
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


# ===========================================================================
# runs
# ===========================================================================

def run_deblur(sigma, sigma_n, eps=1e-2, K=None, gamma=None,
               limit=None, save=True, combined=True):
    n_k = kernel_size(sigma)
    print(f"sigma = {sigma:g}   kernel {n_k}x{n_k}   "
          f"sigma_n = {sigma_n * 255:g}/255   "
          f"eps = {eps:g}   K = {'auto' if K is None else f'{K:g}'}   "
          f"gamma = {'auto' if gamma is None else f'{gamma:g}'}\n")

    # each entry takes (degraded, ground truth); only the oracle uses the latter
    methods = [
        ("inverse",       lambda f, g: inverse_filter(f, sigma, eps)),
        ("wiener",        lambda f, g: wiener_filter(f, sigma, K, sigma_n)),
        ("wiener-oracle", lambda f, g: wiener_filter_oracle(f, sigma, g, sigma_n)),
        ("cls-laplacian", lambda f, g: cls_laplacian_filter(f, sigma, gamma, sigma_n)),
    ]

    dests = {}
    if save:
        for label, _ in methods:
            d = DEBLURRED_DIR / label
            d.mkdir(parents=True, exist_ok=True)
            dests[label] = d
    comb = None
    if combined:
        comb = COMBINED_DIR
        comb.mkdir(parents=True, exist_ok=True)

    # the degraded inputs the figures show are the ones the filters actually saw
    deg_dir = blurred_dir(sigma, sigma_n)
    if save:
        deg_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, blurry, sharp in load_pairs(sigma, sigma_n, limit):
        if save:
            save_image(blurry, deg_dir / name)
        p_in, s_in = scores(blurry, sharp)
        panels = [("blurry", blurry, p_in, s_in)]
        row = [p_in, s_in]

        for label, fn in methods:
            t0 = time.perf_counter()
            rec = fn(blurry, sharp)
            dt = time.perf_counter() - t0
            p_out, s_out = scores(rec, sharp)
            row += [p_out, s_out, dt]
            panels.append((label, rec, p_out, s_out))
            if label in dests:
                save_image(rec, dests[label] / name)

        panels.append(("sharp", sharp, None, None))
        if comb:
            save_combined(panels, comb / name)
        rows.append(row)

    a = np.array(rows)
    print(f"{'':22} {'PSNR':>8} {'SSIM':>8} {'ms':>7}")
    print(f"{'degraded input':22} {a[:,0].mean():8.2f} {a[:,1].mean():8.4f}")
    for i, (label, _) in enumerate(methods):
        p, s, t = a[:, 2 + 3*i], a[:, 3 + 3*i], a[:, 4 + 3*i]
        print(f"{label:22} {p.mean():8.2f} {s.mean():8.4f} {t.mean()*1000:7.0f}"
              f"   ({p.mean()-a[:,0].mean():+.2f} dB)")
    print(f"\n{len(rows)} images")
    if save:
        print(f"{'degraded':10} -> {deg_dir}")
    for label, d in dests.items():
        print(f"{label:10} -> {d}")
    if comb:
        print(f"{'combined':10} -> {comb}")
    save_results(a, methods, sigma, sigma_n, len(rows))
    save_panels(sigma, sigma_n, eps=eps, K=K, gamma=gamma)
    return a


# ===========================================================================
# results -- chart and table for the report
# ===========================================================================

REPORT_DIR = HERE / "frequency-domain-deblurring-report"

# how each method is named in the report
PRETTY = {
    "inverse":       "Inverse",
    "wiener":        "Wiener",
    "wiener-oracle": "Wiener\n(oracle)",
    "cls-laplacian": "CLS\n(Laplacian)",
}


def save_results(a, methods, sigma, sigma_n, n):
    """
    Write the PSNR/SSIM chart and the matching LaTeX table into the report
    folder, both straight from the array `run_deblur` accumulated.

    The table is written as a fragment the report \\inputs, so the figures in
    the text cannot drift from the figures in the chart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Degraded\ninput"] + [PRETTY[l] for l, _ in methods]
    psnr = [a[:, 0].mean()] + [a[:, 2 + 3*i].mean() for i in range(len(methods))]
    ssim = [a[:, 1].mean()] + [a[:, 3 + 3*i].mean() for i in range(len(methods))]

    # the input is the reference, and the oracle is not achievable in practice
    face = ["0.75"] + ["#4878a8"] * len(methods)
    hatch = [""] + ["" if l != "wiener-oracle" else "//" for l, _ in methods]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, vals, name in ((axes[0], psnr, "PSNR (dB)"),
                           (axes[1], ssim, "SSIM")):
        bars = ax.bar(labels, vals, color=face, edgecolor="white", zorder=3)
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", color="0.9", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
        # bar length encodes magnitude, so the axis starts at zero: the near-tie
        # between the three damped filters should look like a near-tie
        ax.set_ylim(0, max(vals) * 1.18 if name.startswith("PSNR") else 1.0)
        fmt = "{:.2f}" if name.startswith("PSNR") else "{:.4f}"
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Fourier-domain restoration, CBSD68 ($\\sigma$ = {sigma:g}, "
                 f"$\\sigma_n$ = {sigma_n * 255:g}/255, {n} images); "
                 f"hatched = oracle", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "results.pdf")
    plt.close(fig)

    # the same numbers, as the report's table
    base_p, base_s = a[:, 0].mean(), a[:, 1].mean()
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Method & PSNR (dB) & $\Delta$ PSNR & SSIM & ms/image \\",
            r"\midrule",
            rf"Degraded input & {base_p:.2f} & --- & {base_s:.4f} & --- \\",
            r"\midrule"]
    names = {"inverse": "Inverse filter", "wiener": "Wiener filter",
             "wiener-oracle": "Wiener filter (oracle)",
             "cls-laplacian": "Constrained least squares"}
    for i, (label, _) in enumerate(methods):
        p, s, t = (a[:, 2 + 3*i].mean(), a[:, 3 + 3*i].mean(),
                   a[:, 4 + 3*i].mean() * 1000)
        rows.append(rf"{names[label]} & {p:.2f} & {p - base_p:+.2f} & "
                    rf"{s:.4f} & {t:.0f} \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (REPORT_DIR / "results-table.tex").write_text("\n".join(rows) + "\n")

    print(f"{'chart':10} -> {REPORT_DIR / 'results.pdf'}")
    print(f"{'table':10} -> {REPORT_DIR / 'results-table.tex'}")


SAMPLES = ["3096", "12084", "16077", "19021", "24077"]


def save_panels(sigma, sigma_n, names=SAMPLES, eps=1e-2, K=None, gamma=None):
    """
    Write sample-panels.tex: one 3x2 portrait grid per sample image, each cell
    captioned with that panel's own PSNR/SSIM.

    The degradation is regenerated here from the same name-seeded noise the
    deblur run used, so the scores printed in the captions belong to the images
    the paths point at.  Only the LaTeX arranging them is new.
    """
    deg_dir = blurred_dir(sigma, sigma_n)
    rel_deg = f"../blurred/{deg_dir.name}"
    # (caption, path relative to the report)
    cells = [
        ("Degraded input",            rel_deg),
        ("Inverse filter",            "../deblurred-imgs/inverse"),
        ("Wiener filter",             "../deblurred-imgs/wiener"),
        ("Wiener filter (oracle)",    "../deblurred-imgs/wiener-oracle"),
        ("Constrained least squares", "../deblurred-imgs/cls-laplacian"),
        ("Ground truth",              "../../CBSD68"),
    ]

    out = []
    for n, name in enumerate(names):
        fname = f"{name}.png"
        sharp = load_image(DATA_DIR / fname)
        blurry = blur_image(sharp, sigma, sigma_n, image_rng(fname))
        imgs = [blurry,
                inverse_filter(blurry, sigma, eps),
                wiener_filter(blurry, sigma, K, sigma_n),
                wiener_filter_oracle(blurry, sigma, sharp, sigma_n),
                cls_laplacian_filter(blurry, sigma, gamma, sigma_n),
                sharp]

        if n:                                   # break between grids, not before
            out.append(r"\clearpage")
        out.append(r"\begin{center}")
        for i, ((label, folder), im) in enumerate(zip(cells, imgs)):
            if label == "Ground truth":
                cap = label
            else:
                p, s = scores(im, sharp)
                cap = rf"{label} --- {p:.2f}\,dB / {s:.4f}"
            out.append(r"  \begin{minipage}{0.48\linewidth}\centering")
            out.append(rf"    \includegraphics[width=\linewidth]"
                       rf"{{{folder}/{name}.png}}\\[2pt]")
            out.append(rf"    {{\footnotesize {cap}}}")
            out.append(r"  \end{minipage}" + ("%" if i % 2 == 0 else ""))
            out.append(r"  \hfill" if i % 2 == 0 else r"  \\[8pt]")
        out.append(r"  \captionof{figure}{\texttt{" + name + r".png} restored from "
                   rf"$\sigma = {sigma:g}$ blur with $\sigma_n = {sigma_n * 255:g}$ "
                   rf"noise. Read left to right, top to bottom.}}")
        out.append(r"\end{center}")
        out.append("")

    (REPORT_DIR / "sample-panels.tex").write_text("\n".join(out) + "\n")
    print(f"{'panels':10} -> {REPORT_DIR / 'sample-panels.tex'}  "
          f"({len(names)} grids)")


def run_sweep(sigma, sigma_n, limit=None):
    pairs = list(load_pairs(sigma, sigma_n, limit))
    base = np.array([scores(b, s) for _, b, s in pairs])
    print(f"sigma = {sigma:g}   sigma_n = {sigma_n * 255:g}/255   "
          f"{len(pairs)} images\n")
    print(f"{'eps':>10} {'PSNR':>8} {'SSIM':>8}")
    print(f"{'(degraded)':>10} {base[:,0].mean():8.2f} {base[:,1].mean():8.4f}")
    # spread either side of |H|'s useful range: even the best eps stays below
    # the degraded input, which is the point the sweep is there to make
    for eps in [8e-1, 5e-1, 3e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-10]:
        r = np.array([scores(inverse_filter(b, sigma, eps), s)
                      for _, b, s in pairs])
        print(f"{eps:10.0e} {r[:,0].mean():8.2f} {r[:,1].mean():8.4f}")


def run_test():
    paths = list_images()
    print(f"dataset: {DATA_DIR}")
    print(f"images found: {len(paths)}")
    if not paths:
        raise SystemExit("no images found -- check DATA_DIR")

    sharp = load_image(paths[0])
    print(f"sample: {paths[0].name}  shape={sharp.shape}  "
          f"range=[{sharp.min():.3f}, {sharp.max():.3f}]")

    print(f"\n{'sigma':>6} {'ksize':>6} {'H(0,0)':>9} {'mean drift':>11} "
          f"{'grad ratio':>11} {'invertible':>11}")
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        H = fft2(gaussian_h(sigma, sharp.shape[:2]))
        blurry = blur_image(sharp, sigma)

        # unit-sum kernel  <=>  DC gain of exactly 1
        dc = float(np.real(H[0, 0]))
        drift = abs(blurry.mean() - sharp.mean())

        def grad_energy(im):
            gy, gx = np.gradient(im.mean(axis=2))
            return float(np.mean(gx ** 2 + gy ** 2))
        ratio = grad_energy(blurry) / grad_energy(sharp)

        # noiseless round trip: blur then divide by H must recover the original
        rec = np.empty_like(sharp)
        for c in range(sharp.shape[2]):
            G = fft2(sharp[:, :, c])            # original -> frequencies
            F = H * G                           # blur
            G_hat = F / H                       # divide it straight back out
            rec[:, :, c] = np.real(ifft2(G_hat))
        err = float(np.abs(rec - sharp).max())

        print(f"{sigma:6.1f} {kernel_size(sigma):6d} {dc:9.5f} {drift:11.2e} "
              f"{ratio:11.4f} {err:11.2e}")

    print("\nchecks: H(0,0) == 1, mean drift ~ 0, grad ratio < 1 and falling,")
    print("invertible (max abs error of noiseless H-division) ~ 0")

    # the noise the degradation adds must have the standard deviation asked for
    print(f"\n{'sigma_n':>9} {'measured':>10} {'ratio':>8}")
    clean = blur_image(sharp, 2.0)
    for s255 in [5.0, 15.0, 25.0, 50.0]:
        sn = s255 / 255.0
        noisy = blur_image(sharp, 2.0, sn, image_rng(paths[0].name))
        meas = float(np.std(noisy - clean)) * 255.0
        print(f"{s255:9.1f} {meas:10.2f} {meas / s255:8.4f}")
    print("\nchecks: measured / requested ~ 1")


# ===========================================================================
# cli
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Frequency-domain deblurring: degrade and restore.")
    ap.add_argument("stage",
                    choices=["blur", "deblur", "sweep", "test"],
                    help="blur: write degraded images; deblur: restore them; "
                         "sweep: vary epsilon; test: correctness checks")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="blur width, in pixels")
    ap.add_argument("--sigma-n", type=float, default=SIGMA_N_DEFAULT,
                    help="noise standard deviation on the 0-255 scale "
                         f"(default {SIGMA_N_DEFAULT:g})")
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("-K", type=float, default=None,
                    help="Wiener NSR constant; default estimates it per image")
    ap.add_argument("--gamma", type=float, default=None,
                    help="CLS smoothness weight; default scales it with sigma_n")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    sigma_n = args.sigma_n / 255.0              # report in 0-255, compute in [0,1]

    if args.stage == "blur":
        write_blurred(args.sigma, sigma_n, args.limit)
    elif args.stage == "deblur":
        run_deblur(args.sigma, sigma_n, args.eps, args.K, args.gamma,
                   args.limit,
                   save=not args.no_save, combined=not args.no_combined)
    elif args.stage == "sweep":
        run_sweep(args.sigma, sigma_n, args.limit)
    elif args.stage == "test":
        run_test()


if __name__ == "__main__":
    main()
