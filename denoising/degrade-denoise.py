"""
Frequency-domain denoising: degradation and restoration in one file.

Mirrors deblurring/degrade-deblur.py so both halves of the project share the
same pipeline shape, metrics, and report hooks.

Pipeline
--------
    1. load a clean CBSD68 image
    2. add white Gaussian noise              (write to noisy/)
    3. restore with Fourier and spatial methods (write to denoised-imgs/)
    4. write side-by-side comparisons         (write to Combined/)

Degradation model (no blur):

    f = g + n,   n ~ N(0, sigma^2)

Fourier methods operate on F = fft2(f).  Spatial baselines run in the pixel
domain for the comparison the assignment asks for.

Usage
-----
    python3 degrade-denoise.py noise
    python3 degrade-denoise.py denoise
    python3 degrade-denoise.py denoise --sigma 25
    python3 degrade-denoise.py denoise --source memory
    python3 degrade-denoise.py sweep
    python3 degrade-denoise.py test

Common options:  --sigma 25   --limit N   --no-save   --no-combined
"""

import argparse
import time
from pathlib import Path

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from PIL import Image, ImageDraw
from skimage.filters import gaussian as sk_gaussian
from skimage.filters import median as sk_median
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral

HERE = Path(__file__).resolve().parent

DATA_DIR     = HERE.parent / "CBSD68"     # clean reference images
NOISY_DIR    = HERE / "noisy"            # synthetically noisy inputs
DENOISED_DIR = HERE / "denoised-imgs"    # one subfolder per method
COMBINED_DIR = HERE / "Combined"         # side-by-side strips
REPORT_DIR   = HERE                      # report lives alongside this script


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


# ===========================================================================
# naming (course notes)
# ===========================================================================

"""
    lower case = spatial domain
    UPPER case = frequency domain

    g, G          original (clean) image
    f, F          degraded observation:  f = g + n
    n, N          additive noise
    w, W          restoration filter
    g_hat, G_hat  restored estimate of the original image
"""


# ===========================================================================
# degradation
# ===========================================================================

def add_noise(clean, sigma, rng=None):
    """
    Add white Gaussian noise of std-dev `sigma` (in [0, 1] intensity units).

        f = g + n,   n ~ N(0, sigma^2)

    `sigma` is usually given in 8-bit grey levels (e.g. 25); pass the float
    fraction yourself, or use sigma_from_levels().
    """
    if rng is None:
        rng = np.random.default_rng(0)
    noise = rng.normal(0.0, sigma, size=clean.shape)
    return np.clip(clean + noise, 0.0, 1.0)


def sigma_from_levels(levels):
    """Convert an 8-bit noise level (e.g. 25) to a float-in-[0,1] std-dev."""
    return float(levels) / 255.0


def write_noisy(sigma_levels, limit=None, seed=0):
    """Add noise to every clean image and write to noisy/sigma_<levels>/."""
    paths = list_images()
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images found in {DATA_DIR}")

    sigma = sigma_from_levels(sigma_levels)
    dest = NOISY_DIR / f"sigma_{sigma_levels:g}"
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"sigma = {sigma_levels:g}/255  ({sigma:.5f} float)")
    print(f"source: {DATA_DIR}")
    print(f"dest:   {dest}\n")

    for n, p in enumerate(paths, 1):
        # per-image sub-RNG so --limit N is a prefix of the full run
        img_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        save_image(add_noise(load_image(p), sigma, img_rng), dest / p.name)
        if n % 17 == 0 or n == len(paths):
            print(f"  {n}/{len(paths)}")

    total = sum(f.stat().st_size for f in dest.glob("*.png"))
    print(f"\nwrote {len(paths)} images  ({total/1e6:.1f} MB)")
    return dest


# ===========================================================================
# metrics
# ===========================================================================

def scores(restored, clean):
    r = np.clip(restored, 0.0, 1.0)
    return (psnr_fn(clean, r, data_range=1.0),
            ssim_fn(clean, r, data_range=1.0, channel_axis=2))


# ===========================================================================
# frequency helpers
# ===========================================================================

def radial_grid(shape):
    """
    Distance from the spectrum centre, in cycles per sample, after fftshift.

    DC sits at the array centre.  The Nyquist corner is about sqrt(0.5^2+0.5^2).
    """
    M, N = shape
    v = np.fft.fftfreq(M)[:, None]
    u = np.fft.fftfreq(N)[None, :]
    # shift so (0,0) frequency is at the centre, matching H construction below
    D = np.sqrt(fftshift(v) ** 2 + fftshift(u) ** 2)
    return D


def apply_transfer(noisy, H_shifted):
    """
    Apply a zero-centred transfer function H to each colour channel.

    H_shifted is laid out with DC in the middle (as radial_grid produces).
    We ifftshift it before multiplying the unshifted FFT.
    """
    H = ifftshift(H_shifted)
    out = np.empty_like(noisy)
    for c in range(noisy.shape[2]):
        F = fft2(noisy[:, :, c])
        G_hat = H * F
        out[:, :, c] = np.real(ifft2(G_hat))
    return out


# ===========================================================================
# restoration -- frequency domain
# ===========================================================================

def ideal_lpf(noisy, cutoff=0.08):
    """
    Ideal low-pass: keep frequencies with D <= cutoff, zero the rest.

    Sharp cutoffs ring (Gibbs), which is the point of including this method.
    """
    D = radial_grid(noisy.shape[:2])
    H = (D <= cutoff).astype(np.float64)
    return apply_transfer(noisy, H)


def butterworth_lpf(noisy, cutoff=0.08, order=2):
    """
    Butterworth low-pass of order `order`:

        H(u,v) = 1 / (1 + (D / D0)^{2n})

    Softer than the ideal filter, so less ringing for a similar cutoff.
    """
    D = radial_grid(noisy.shape[:2])
    H = 1.0 / (1.0 + (D / max(cutoff, 1e-12)) ** (2 * order))
    return apply_transfer(noisy, H)


def gaussian_lpf(noisy, cutoff=0.08):
    """
    Gaussian low-pass in the Fourier domain:

        H(u,v) = exp( -D^2 / (2 D0^2) )

    No ringing; comparable role to a spatial Gaussian blur, but applied as a
    multiplication on F.
    """
    D = radial_grid(noisy.shape[:2])
    H = np.exp(-(D ** 2) / (2.0 * max(cutoff, 1e-12) ** 2))
    return apply_transfer(noisy, H)


def wiener_denoise(noisy, sigma, K=None):
    """
    Frequency-domain Wiener filter for additive noise with H = 1:

        Ghat = P_g / (P_g + P_n) * F

    P_g is unknown, so we estimate it from the noisy periodogram:

        P_f = |F|^2
        P_g ≈ max(P_f - P_n, 0)
        P_n = M N sigma^2          (white, flat)

    If K is given, fall back to the constant-ratio form W = 1/(1+K) instead
    (a uniform shrink -- mainly useful as a didactic baseline).
    """
    M, N = noisy.shape[:2]
    P_n = M * N * (sigma ** 2)

    out = np.empty_like(noisy)
    for c in range(noisy.shape[2]):
        F = fft2(noisy[:, :, c])
        if K is not None:
            W = 1.0 / (1.0 + K)
        else:
            P_f = np.abs(F) ** 2
            P_g = np.maximum(P_f - P_n, 0.0)
            W = P_g / (P_g + P_n)
        out[:, :, c] = np.real(ifft2(W * F))
    return out


def wiener_denoise_oracle(noisy, sigma, clean):
    """
    Oracle Wiener with H = 1, using the true per-frequency signal PSD:

        W = P_g / (P_g + P_n)

    P_n is flat white noise: M N sigma^2.  Uses the ground-truth image, so this
    is an upper bound, not an achievable method -- same role as wiener-oracle
    in the deblurring half.
    """
    M, N = noisy.shape[:2]
    P_n = M * N * (sigma ** 2)

    out = np.empty_like(noisy)
    for c in range(noisy.shape[2]):
        F = fft2(noisy[:, :, c])
        G = fft2(clean[:, :, c])
        P_g = np.abs(G) ** 2
        W = P_g / (P_g + P_n)
        out[:, :, c] = np.real(ifft2(W * F))
    return out


# ===========================================================================
# restoration -- spatial-domain baselines
# ===========================================================================

def gaussian_spatial(noisy, sigma_s=1.0):
    """Isotropic Gaussian smoothing in the pixel domain (skimage)."""
    # channel_axis keeps RGB from being blurred across colour
    return sk_gaussian(noisy, sigma=sigma_s, channel_axis=2,
                       preserve_range=True)


def median_spatial(noisy, radius=1):
    """Per-channel median filter with a disk footprint."""
    out = np.empty_like(noisy)
    footprint = disk(radius)
    for c in range(noisy.shape[2]):
        out[:, :, c] = sk_median(noisy[:, :, c], footprint=footprint)
    return out


def bilateral_spatial(noisy, sigma_c=0.08, sigma_s=2.0):
    """
    Bilateral filter: spatial Gaussian weighted by tonal similarity.

    Edges survive better than a plain Gaussian of similar spatial width.
    """
    # denoise_bilateral expects float in [0,1] when channel_axis is set
    return denoise_bilateral(noisy, sigma_color=sigma_c, sigma_spatial=sigma_s,
                             channel_axis=2)


# ===========================================================================
# pair loading
# ===========================================================================

def load_pairs(sigma_levels, source="disk", limit=None, seed=0):
    """
    Yield (name, noisy, clean) triples.

    source="disk"   : read the PNGs written by the noise stage (8-bit quantised)
    source="memory" : re-noise now, keeping full float precision
    """
    clean_paths = list_images()
    if limit:
        clean_paths = clean_paths[:limit]

    sigma = sigma_from_levels(sigma_levels)

    if source == "memory":
        rng = np.random.default_rng(seed)
        for p in clean_paths:
            clean = load_image(p)
            img_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            yield p.name, add_noise(clean, sigma, img_rng), clean
        return

    noise_dir = NOISY_DIR / f"sigma_{sigma_levels:g}"
    if not noise_dir.is_dir():
        raise SystemExit(
            f"{noise_dir} not found -- run:  "
            f"python3 degrade-denoise.py noise --sigma {sigma_levels:g}")
    for p in clean_paths:
        np_ = noise_dir / p.name
        if np_.exists():
            yield p.name, load_image(np_), load_image(p)


# ===========================================================================
# combined side-by-side output
# ===========================================================================

BAR = 26


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

def method_list(sigma, cutoff, butter_order, gauss_s, median_r,
                bilat_c, bilat_s, K):
    """Build the (label, fn(noisy, clean)) list used by denoise / panels."""
    return [
        ("ideal",
         lambda f, g: ideal_lpf(f, cutoff)),
        ("butterworth",
         lambda f, g: butterworth_lpf(f, cutoff, butter_order)),
        ("gaussian-lpf",
         lambda f, g: gaussian_lpf(f, cutoff)),
        ("wiener",
         lambda f, g: wiener_denoise(f, sigma, K)),
        ("wiener-oracle",
         lambda f, g: wiener_denoise_oracle(f, sigma, g)),
        ("gaussian-spatial",
         lambda f, g: gaussian_spatial(f, gauss_s)),
        ("median",
         lambda f, g: median_spatial(f, median_r)),
        ("bilateral",
         lambda f, g: bilateral_spatial(f, bilat_c, bilat_s)),
    ]


def run_denoise(sigma_levels, source="disk", cutoff=0.08, butter_order=2,
                gauss_s=1.0, median_r=1, bilat_c=0.08, bilat_s=2.0, K=None,
                limit=None, save=True, combined=True, seed=0):
    sigma = sigma_from_levels(sigma_levels)
    print(f"sigma = {sigma_levels:g}/255  ({sigma:.5f} float)   "
          f"cutoff = {cutoff:g}   source = {source}\n")

    methods = method_list(sigma, cutoff, butter_order, gauss_s, median_r,
                          bilat_c, bilat_s, K)

    dests = {}
    if save:
        for label, _ in methods:
            d = DENOISED_DIR / label
            d.mkdir(parents=True, exist_ok=True)
            dests[label] = d
    comb = None
    if combined:
        comb = COMBINED_DIR
        comb.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, noisy, clean in load_pairs(sigma_levels, source, limit, seed):
        p_in, s_in = scores(noisy, clean)
        panels = [("noisy", noisy, p_in, s_in)]
        row = [p_in, s_in]

        for label, fn in methods:
            t0 = time.perf_counter()
            rec = fn(noisy, clean)
            dt = time.perf_counter() - t0
            p_out, s_out = scores(rec, clean)
            row += [p_out, s_out, dt]
            panels.append((label, rec, p_out, s_out))
            if label in dests:
                save_image(rec, dests[label] / name)

        panels.append(("clean", clean, None, None))
        if comb:
            save_combined(panels, comb / name)
        rows.append(row)

    a = np.array(rows)
    print(f"{'':22} {'PSNR':>8} {'SSIM':>8} {'ms':>7}")
    print(f"{'noisy input':22} {a[:,0].mean():8.2f} {a[:,1].mean():8.4f}")
    for i, (label, _) in enumerate(methods):
        p, s, t = a[:, 2 + 3*i], a[:, 3 + 3*i], a[:, 4 + 3*i]
        print(f"{label:22} {p.mean():8.2f} {s.mean():8.4f} {t.mean()*1000:7.0f}"
              f"   ({p.mean()-a[:,0].mean():+.2f} dB)")
    print(f"\n{len(rows)} images")
    for label, d in dests.items():
        print(f"{label:18} -> {d}")
    if comb:
        print(f"{'combined':18} -> {comb}")
    save_results(a, methods, sigma_levels, len(rows))
    save_panels(sigma_levels, source=source, cutoff=cutoff,
                butter_order=butter_order, gauss_s=gauss_s, median_r=median_r,
                bilat_c=bilat_c, bilat_s=bilat_s, K=K, seed=seed)
    return a


# ===========================================================================
# results -- chart and table for the report
# ===========================================================================

PRETTY = {
    "ideal":             "Ideal\nLPF",
    "butterworth":       "Butterworth\nLPF",
    "gaussian-lpf":      "Gaussian\nLPF",
    "wiener":            "Wiener",
    "wiener-oracle":     "Wiener\n(oracle)",
    "gaussian-spatial":  "Gaussian\n(spatial)",
    "median":            "Median",
    "bilateral":         "Bilateral",
}

NAMES = {
    "ideal":             "Ideal LPF",
    "butterworth":       "Butterworth LPF",
    "gaussian-lpf":      "Gaussian LPF",
    "wiener":            "Wiener filter",
    "wiener-oracle":     "Wiener filter (oracle)",
    "gaussian-spatial":  "Gaussian (spatial)",
    "median":            "Median filter",
    "bilateral":         "Bilateral filter",
}


def save_results(a, methods, sigma_levels, n):
    """Write results.pdf and results-table.tex from the denoise run array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Noisy\ninput"] + [PRETTY[l] for l, _ in methods]
    psnr = [a[:, 0].mean()] + [a[:, 2 + 3*i].mean() for i in range(len(methods))]
    ssim = [a[:, 1].mean()] + [a[:, 3 + 3*i].mean() for i in range(len(methods))]

    face = ["0.75"]
    hatch = [""]
    for l, _ in methods:
        # frequency methods in one colour, spatial baselines in another
        if l in ("gaussian-spatial", "median", "bilateral"):
            face.append("#6a9a5b")
        else:
            face.append("#4878a8")
        hatch.append("//" if l == "wiener-oracle" else "")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
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
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, max(vals) * 1.18 if name.startswith("PSNR") else 1.0)
        fmt = "{:.2f}" if name.startswith("PSNR") else "{:.4f}"
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=6)

    fig.suptitle(
        f"Denoising on CBSD68 ($\\sigma$ = {sigma_levels:g}/255, {n} images); "
        f"blue = Fourier, green = spatial, hatched = oracle",
        fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "results.pdf")
    plt.close(fig)

    base_p, base_s = a[:, 0].mean(), a[:, 1].mean()
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Method & PSNR (dB) & $\Delta$ PSNR & SSIM & ms/image \\",
            r"\midrule",
            rf"Noisy input & {base_p:.2f} & --- & {base_s:.4f} & --- \\",
            r"\midrule"]
    for i, (label, _) in enumerate(methods):
        p, s, t = (a[:, 2 + 3*i].mean(), a[:, 3 + 3*i].mean(),
                   a[:, 4 + 3*i].mean() * 1000)
        domain = "Fourier" if label not in (
            "gaussian-spatial", "median", "bilateral") else "spatial"
        # mark domain lightly in the name for the report table
        tag = NAMES[label]
        if label == "wiener-oracle":
            tag += " *"
        rows.append(rf"{tag} & {p:.2f} & {p - base_p:+.2f} & "
                    rf"{s:.4f} & {t:.0f} \\")
        _ = domain  # kept for readability / future grouping
    rows += [r"\bottomrule", r"\end{tabular}"]
    (REPORT_DIR / "results-table.tex").write_text("\n".join(rows) + "\n")

    print(f"{'chart':18} -> {REPORT_DIR / 'results.pdf'}")
    print(f"{'table':18} -> {REPORT_DIR / 'results-table.tex'}")


SAMPLES = ["3096", "12084", "16077", "19021", "24077"]


def save_panels(sigma_levels, names=SAMPLES, source="disk", cutoff=0.08,
                butter_order=2, gauss_s=1.0, median_r=1, bilat_c=0.08,
                bilat_s=2.0, K=None, seed=0):
    """
    Write sample-panels.tex: one grid per sample image.  Paths point at the
    PNGs already written by the noise / denoise stages.
    """
    sigma = sigma_from_levels(sigma_levels)
    noise_dir = NOISY_DIR / f"sigma_{sigma_levels:g}"

    # a 3x3-ish selection: noisy, three Fourier, three spatial, clean
    cells = [
        ("Noisy input",               f"noisy/sigma_{sigma_levels:g}"),
        ("Ideal LPF",                 "denoised-imgs/ideal"),
        ("Butterworth LPF",           "denoised-imgs/butterworth"),
        ("Gaussian LPF",              "denoised-imgs/gaussian-lpf"),
        ("Wiener (oracle)",           "denoised-imgs/wiener-oracle"),
        ("Gaussian (spatial)",        "denoised-imgs/gaussian-spatial"),
        ("Median",                    "denoised-imgs/median"),
        ("Bilateral",                 "denoised-imgs/bilateral"),
        ("Ground truth",              "../CBSD68"),
    ]

    methods = method_list(sigma, cutoff, butter_order, gauss_s, median_r,
                          bilat_c, bilat_s, K)
    by_label = {lab: fn for lab, fn in methods}

    out = []
    for n, name in enumerate(names):
        clean = load_image(DATA_DIR / f"{name}.png")
        noisy = load_image(noise_dir / f"{name}.png")

        restored = {
            "ideal": by_label["ideal"](noisy, clean),
            "butterworth": by_label["butterworth"](noisy, clean),
            "gaussian-lpf": by_label["gaussian-lpf"](noisy, clean),
            "wiener-oracle": by_label["wiener-oracle"](noisy, clean),
            "gaussian-spatial": by_label["gaussian-spatial"](noisy, clean),
            "median": by_label["median"](noisy, clean),
            "bilateral": by_label["bilateral"](noisy, clean),
        }
        imgs = [noisy,
                restored["ideal"], restored["butterworth"], restored["gaussian-lpf"],
                restored["wiener-oracle"],
                restored["gaussian-spatial"], restored["median"], restored["bilateral"],
                clean]

        if n:
            out.append(r"\clearpage")
        out.append(r"\begin{center}")
        for i, ((label, folder), im) in enumerate(zip(cells, imgs)):
            if label == "Ground truth":
                cap = label
            else:
                p, s = scores(im, clean)
                cap = rf"{label} --- {p:.2f}\,dB / {s:.4f}"
            out.append(r"  \begin{minipage}{0.32\linewidth}\centering")
            out.append(rf"    \includegraphics[width=\linewidth]"
                       rf"{{{folder}/{name}.png}}\\[2pt]")
            out.append(rf"    {{\footnotesize {cap}}}")
            out.append(r"  \end{minipage}" + ("%" if i % 3 != 2 else ""))
            if i % 3 != 2:
                out.append(r"  \hfill")
            else:
                out.append(r"  \\[8pt]")
        out.append(r"  \captionof{figure}{\texttt{" + name + r".png} restored from "
                   rf"$\sigma = {sigma_levels:g}/255$ noise. "
                   r"Read left to right, top to bottom.}")
        out.append(r"\end{center}")
        out.append("")

    (REPORT_DIR / "sample-panels.tex").write_text("\n".join(out) + "\n")
    print(f"{'panels':18} -> {REPORT_DIR / 'sample-panels.tex'}  "
          f"({len(names)} grids)")


def run_sweep(sigma_levels, source="disk", limit=None, seed=0):
    """Sweep the Fourier LPF cutoff; print mean PSNR/SSIM for Butterworth."""
    pairs = list(load_pairs(sigma_levels, source, limit, seed))
    base = np.array([scores(f, g) for _, f, g in pairs])
    print(f"sigma = {sigma_levels:g}/255   source = {source}   "
          f"{len(pairs)} images\n")
    print(f"{'cutoff':>10} {'PSNR':>8} {'SSIM':>8}")
    print(f"{'(noisy)':>10} {base[:,0].mean():8.2f} {base[:,1].mean():8.4f}")
    for cutoff in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        r = np.array([scores(butterworth_lpf(f, cutoff), g)
                      for _, f, g in pairs])
        print(f"{cutoff:10.2f} {r[:,0].mean():8.2f} {r[:,1].mean():8.4f}")


def run_test():
    paths = list_images()
    print(f"dataset: {DATA_DIR}")
    print(f"images found: {len(paths)}")
    if not paths:
        raise SystemExit("no images found -- check DATA_DIR")

    clean = load_image(paths[0])
    print(f"sample: {paths[0].name}  shape={clean.shape}  "
          f"range=[{clean.min():.3f}, {clean.max():.3f}]")

    print(f"\n{'sigma':>8} {'meas std':>10} {'PSNR':>8} {'identity':>10}")
    rng = np.random.default_rng(0)
    for levels in [10, 15, 25, 50]:
        sigma = sigma_from_levels(levels)
        noisy = add_noise(clean, sigma, rng)
        # measured residual std on the unclipped interior-ish statistic
        # (clipping shrinks it slightly at high sigma)
        meas = float(np.std(noisy - clean))
        p, _ = scores(noisy, clean)
        # zero noise must leave the image untouched
        ident = float(np.abs(add_noise(clean, 0.0, rng) - clean).max())
        print(f"{levels:8.0f} {meas:10.5f} {p:8.2f} {ident:10.2e}")

    print("\nchecks: meas std ~ sigma/255, PSNR falls with sigma, "
          "identity error ~ 0")


# ===========================================================================
# cli
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Frequency-domain denoising: degrade and restore.")
    ap.add_argument("stage",
                    choices=["noise", "denoise", "sweep", "test"],
                    help="noise: write degraded images; denoise: restore them; "
                         "sweep: vary LPF cutoff; test: correctness checks")
    ap.add_argument("--sigma", type=float, default=25.0,
                    help="noise std-dev in 8-bit levels (default 25)")
    ap.add_argument("--cutoff", type=float, default=0.08,
                    help="Fourier LPF cutoff in cycles/sample (default 0.08)")
    ap.add_argument("--order", type=int, default=2,
                    help="Butterworth order (default 2)")
    ap.add_argument("--gauss-s", type=float, default=1.0,
                    help="spatial Gaussian sigma in pixels (default 1)")
    ap.add_argument("--median-r", type=int, default=1,
                    help="median filter disk radius (default 1)")
    ap.add_argument("--bilat-c", type=float, default=0.08,
                    help="bilateral sigma_color in [0,1] (default 0.08)")
    ap.add_argument("--bilat-s", type=float, default=2.0,
                    help="bilateral sigma_spatial in pixels (default 2)")
    ap.add_argument("-K", type=float, default=None,
                    help="Wiener K = Pn/Pg; default estimates from sigma")
    ap.add_argument("--source", choices=["disk", "memory"], default="disk")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    if args.stage == "noise":
        write_noisy(args.sigma, args.limit, args.seed)
    elif args.stage == "denoise":
        run_denoise(args.sigma, args.source, args.cutoff, args.order,
                    args.gauss_s, args.median_r, args.bilat_c, args.bilat_s,
                    args.K, args.limit,
                    save=not args.no_save, combined=not args.no_combined,
                    seed=args.seed)
    elif args.stage == "sweep":
        run_sweep(args.sigma, args.source, args.limit, args.seed)
    elif args.stage == "test":
        run_test()


if __name__ == "__main__":
    main()
