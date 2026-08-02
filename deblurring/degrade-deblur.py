"""
Frequency-domain deblurring: degradation and restoration in one file.

Self-contained -- no imports from sibling modules.

Pipeline
--------
    1. load a clean CBSD68 image
    2. blur it with a known Gaussian          (write to blurred/)
    3. restore by dividing in the Fourier domain (write to deblurred-imgs/)
    4. write side-by-side comparisons         (write to Combined/)

Everything is parameterised by sigma alone.  The Gaussian's transfer function
H(u,v) is built directly from sigma, and the same H is used both to blur and to
restore, so the degradation matches the restoration model exactly:

    F = H . G        ->        f = ifft2( H * fft2(g) )

The blur is therefore a *circular* convolution.  Zero- or reflection-padded
convolution would introduce boundary behaviour that no frequency-domain filter
can undo.

Usage
-----
    python3 degrade-deblur.py blur                      # write blurred images
    python3 degrade-deblur.py deblur                    # blur -> inverse filter
    python3 degrade-deblur.py deblur --eps 0.1
    python3 degrade-deblur.py deblur --source memory    # skip the 8-bit round trip
    python3 degrade-deblur.py sweep                     # epsilon sweep
    python3 degrade-deblur.py test                      # correctness checks

Common options:  --sigma 2.0   --limit N   --no-save   --no-combined
"""

import argparse
import time
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

def blur_image(sharp, sigma):
    """
    Blur `sharp` with a Gaussian of width `sigma`.

        F = H . G
    """
    h = gaussian_h(sigma, sharp.shape[:2])      # kernel, pixels
    H = fft2(h)                                 # kernel, frequencies

    out = np.empty_like(sharp)
    for c in range(sharp.shape[2]):
        g = sharp[:, :, c]                      # original channel, pixels
        G = fft2(g)                             # -> frequencies
        F = H * G                               # apply the blur
        f = np.real(ifft2(F))                   # -> pixels
        out[:, :, c] = f

    return np.clip(out, 0.0, 1.0)


def degrade(path, sigma):
    """Load the image at `path` and blur it.  Returns (blurry, sharp)."""
    sharp = load_image(path)
    return blur_image(sharp, sigma), sharp


def write_blurred(sigma, limit=None):
    """Blur every clean image and write to blurred/sigma_<sigma>/."""
    paths = list_images()
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images found in {DATA_DIR}")

    dest = BLURRED_DIR / f"sigma_{sigma:g}"
    dest.mkdir(parents=True, exist_ok=True)

    n_k = kernel_size(sigma)
    print(f"sigma = {sigma:g}   kernel {n_k}x{n_k}")
    print(f"source: {DATA_DIR}")
    print(f"dest:   {dest}\n")

    for n, p in enumerate(paths, 1):
        save_image(blur_image(load_image(p), sigma), dest / p.name)
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


def wiener_filter(blurry, sigma, K=3e-4):
    """
    Ghat = conj(H) F / (|H|^2 + K).

    Same division as the inverse filter, but the denominator can never reach
    zero: where |H| is large the factor tends to 1/H, and where |H| is small it
    tends to conj(H)/K, which rolls smoothly to zero instead of exploding.
    K = P_n / P_g, the noise-to-signal power ratio, taken here to be constant.
    """
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


# quantisation noise from the 8-bit round trip: uniform on +/- 1/510,
# so its variance is (1/255)^2 / 12
QUANT_VAR = (1.0 / 255.0) ** 2 / 12.0


def wiener_filter_oracle(blurry, sigma, sharp):
    """
    Ghat = conj(H) P_g F / (|H|^2 P_g + P_n), using the true PSDs.

    Instead of collapsing P_n / P_g into a single constant K, this measures both
    per frequency:

        P_g = |G|^2         from the ground-truth image
        P_n = M N var_n     flat, from the 8-bit quantisation variance

    P_g is the PSD of the image being recovered, so it is not available to a real
    restoration system -- this is an ORACLE, and its score is an upper bound on
    what the Wiener filter can achieve, not an achievable result.
    """
    h = gaussian_h(sigma, blurry.shape[:2])     # kernel, pixels
    H = fft2(h)                                 # kernel, frequencies

    M, N = blurry.shape[:2]
    P_n = M * N * QUANT_VAR                     # white noise -> flat PSD

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


def cls_laplacian_filter(blurry, sigma, gamma=1e-4):
    """
    Ghat = conj(H) F / (|H|^2 + gamma |P|^2).

    Same shape as the Wiener filter, but the constant K is replaced by
    gamma |P|^2, which grows with frequency because P is a high-pass operator.
    The penalty is therefore weak on smooth content and strong on the high
    frequencies where the inverse filter would otherwise amplify error --- no
    noise statistics are required, only a smoothness preference.
    """
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

def load_pairs(sigma, source="disk", limit=None):
    """
    Yield (name, blurry, sharp) triples.

    source="disk"   : read the PNGs written by the blur stage (8-bit quantised)
    source="memory" : re-blur now, keeping full float precision
    """
    sharp_paths = list_images()
    if limit:
        sharp_paths = sharp_paths[:limit]

    if source == "memory":
        for p in sharp_paths:
            sharp = load_image(p)
            yield p.name, blur_image(sharp, sigma), sharp
        return

    blur_dir = BLURRED_DIR / f"sigma_{sigma:g}"
    if not blur_dir.is_dir():
        raise SystemExit(f"{blur_dir} not found -- run:  "
                         f"python3 degrade-deblur.py blur --sigma {sigma:g}")
    for p in sharp_paths:
        bp = blur_dir / p.name
        if bp.exists():
            yield p.name, load_image(bp), load_image(p)


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

def run_deblur(sigma, source="disk", eps=1e-2, K=3e-4, gamma=1e-4,
               limit=None, save=True, combined=True):
    n_k = kernel_size(sigma)
    print(f"sigma = {sigma:g}   kernel {n_k}x{n_k}   "
          f"eps = {eps:g}   K = {K:g}   gamma = {gamma:g}   "
          f"source = {source}\n")

    # each entry takes (degraded, ground truth); only the oracle uses the latter
    methods = [
        ("inverse",       lambda f, g: inverse_filter(f, sigma, eps)),
        ("wiener",        lambda f, g: wiener_filter(f, sigma, K)),
        ("wiener-oracle", lambda f, g: wiener_filter_oracle(f, sigma, g)),
        ("cls-laplacian", lambda f, g: cls_laplacian_filter(f, sigma, gamma)),
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

    rows = []
    for name, blurry, sharp in load_pairs(sigma, source, limit):
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
    print(f"{'blurry input':22} {a[:,0].mean():8.2f} {a[:,1].mean():8.4f}")
    for i, (label, _) in enumerate(methods):
        p, s, t = a[:, 2 + 3*i], a[:, 3 + 3*i], a[:, 4 + 3*i]
        print(f"{label:22} {p.mean():8.2f} {s.mean():8.4f} {t.mean()*1000:7.0f}"
              f"   ({p.mean()-a[:,0].mean():+.2f} dB)")
    print(f"\n{len(rows)} images")
    for label, d in dests.items():
        print(f"{label:10} -> {d}")
    if comb:
        print(f"{'combined':10} -> {comb}")
    save_results(a, methods, sigma, len(rows))
    save_panels(sigma, source=source)
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


def save_results(a, methods, sigma, n):
    """
    Write the PSNR/SSIM chart and the matching LaTeX table into the report
    folder, both straight from the array `run_deblur` accumulated.

    The table is written as a fragment the report \\inputs, so the figures in
    the text cannot drift from the figures in the chart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Blurry\ninput"] + [PRETTY[l] for l, _ in methods]
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
                 f"{n} images); hatched = oracle", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "results.pdf")
    plt.close(fig)

    # the same numbers, as the report's table
    base_p, base_s = a[:, 0].mean(), a[:, 1].mean()
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Method & PSNR (dB) & $\Delta$ PSNR & SSIM & ms/image \\",
            r"\midrule",
            rf"Blurry input & {base_p:.2f} & --- & {base_s:.4f} & --- \\",
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


def save_panels(sigma, names=SAMPLES, source="disk"):
    """
    Write sample-panels.tex: one 3x2 portrait grid per sample image, each cell
    captioned with that panel's own PSNR/SSIM.

    Paths point at the PNGs the blur and deblur stages already wrote, so this
    adds no images -- only the LaTeX that arranges them and the scores that
    label them.
    """
    blur_dir = BLURRED_DIR / f"sigma_{sigma:g}"
    # (subfolder or None for the two ends, caption, path relative to the report)
    cells = [
        ("Blurry input",              f"../blurred/sigma_{sigma:g}"),
        ("Inverse filter",            "../deblurred-imgs/inverse"),
        ("Wiener filter",             "../deblurred-imgs/wiener"),
        ("Wiener filter (oracle)",    "../deblurred-imgs/wiener-oracle"),
        ("Constrained least squares", "../deblurred-imgs/cls-laplacian"),
        ("Ground truth",              "../../CBSD68"),
    ]

    out = []
    for n, name in enumerate(names):
        sharp = load_image(DATA_DIR / f"{name}.png")
        blurry = load_image(blur_dir / f"{name}.png")
        imgs = [blurry,
                inverse_filter(blurry, sigma),
                wiener_filter(blurry, sigma),
                wiener_filter_oracle(blurry, sigma, sharp),
                cls_laplacian_filter(blurry, sigma),
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
                   rf"$\sigma = {sigma:g}$ blur. Read left to right, top to bottom.}}")
        out.append(r"\end{center}")
        out.append("")

    (REPORT_DIR / "sample-panels.tex").write_text("\n".join(out) + "\n")
    print(f"{'panels':10} -> {REPORT_DIR / 'sample-panels.tex'}  "
          f"({len(names)} grids)")


def run_sweep(sigma, source="disk", limit=None):
    pairs = list(load_pairs(sigma, source, limit))
    base = np.array([scores(b, s) for _, b, s in pairs])
    print(f"sigma = {sigma:g}   source = {source}   {len(pairs)} images\n")
    print(f"{'eps':>10} {'PSNR':>8} {'SSIM':>8}")
    print(f"{'(blurry)':>10} {base[:,0].mean():8.2f} {base[:,1].mean():8.4f}")
    for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-10]:
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
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("-K", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1e-4)
    ap.add_argument("--source", choices=["disk", "memory"], default="disk")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    if args.stage == "blur":
        write_blurred(args.sigma, args.limit)
    elif args.stage == "deblur":
        run_deblur(args.sigma, args.source, args.eps, args.K, args.gamma,
                   args.limit,
                   save=not args.no_save, combined=not args.no_combined)
    elif args.stage == "sweep":
        run_sweep(args.sigma, args.source, args.limit)
    elif args.stage == "test":
        run_test()


if __name__ == "__main__":
    main()
