"""
Helpers for the NX-421 mini-projec notebooks.
"""
import glob
import os
import os.path as op
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np


# --------------------------------------------------------------------------
# Files and commands
# --------------------------------------------------------------------------
def find_one(root, pattern):
    """Return the single file matching `pattern` anywhere under `root`."""
    hits = sorted(glob.glob(op.join(root, "**", pattern), recursive=True))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{pattern}' under {root}, found: {hits}")
    return hits[0]


def check_tools(*tools):
    """Fail early if a command-line tool (e.g. bet, fast) is not on the PATH."""
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"Not found on PATH: {missing}. Did the FSL module load?")
    print("FSLDIR =", os.environ.get("FSLDIR"))
    print("All tools found:", ", ".join(tools))


def run(cmd):
    """
    Run a command given as a list of strings, print it, show its output in the
    notebook and raise an error (with FSL's message) if it fails.
    """
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout.strip():
        print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"{cmd[0]} failed (exit code {res.returncode})")
    if res.stderr.strip():
        print(res.stderr)
    print(f"done in {time.time() - t0:.0f} s")


def needs(*outputs, force=False):
    """True if at least one output file is missing (or force=True): the step must run."""
    return force or not all(op.exists(o) for o in outputs)


# --------------------------------------------------------------------------
# Quick quality control
# --------------------------------------------------------------------------
def describe(path):
    """Print shape, voxel size, orientation and data type of a NIfTI file."""
    img = nib.load(path)
    zooms = tuple(round(float(z), 3) for z in img.header.get_zooms())
    print(f"{op.basename(path)}\n"
          f"  shape:       {img.shape}\n"
          f"  voxel size:  {zooms}  (4th value = TR in s for 4D data)\n"
          f"  orientation: {''.join(nib.aff2axcodes(img.affine))}\n"
          f"  dtype:       {img.get_data_dtype()}")


def center_of_mass_world(mask_path):
    """Center of mass of a binary mask, in world (mm) coordinates. Handy as cut_coords."""
    img = nib.load(mask_path)
    ijk = np.argwhere(np.asarray(img.dataobj) > 0).mean(axis=0)
    return tuple(float(c) for c in nib.affines.apply_affine(img.affine, ijk))


def tissue_volumes(seg_path, labels):
    """
    Volume (mL) and fraction of each label in a hard segmentation.
    `labels` maps integer label -> tissue name, e.g. {1: "CSF", 2: "GM", 3: "WM"}.
    """
    img = nib.load(seg_path)
    data = np.asarray(img.dataobj)
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    vols = {name: float((data == lab).sum()) * voxel_ml for lab, name in labels.items()}
    total = sum(vols.values())
    return {name: (v, v / total) for name, v in vols.items()}


def spatial_corr(ref_path, img_path, mask_path):
    """
    Pearson correlation between a reference image and another image inside a mask.
    `img_path` is resampled onto the reference grid using the affines only (no
    registration), so this measures how well the two images already overlap.
    """
    from nilearn import image
    ref = nib.load(ref_path)
    img = image.resample_to_img(img_path, ref, interpolation="continuous")
    mask = np.asarray(nib.load(mask_path).dataobj) > 0
    a = ref.get_fdata()[mask]
    b = img.get_fdata()[mask]
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def save_fig(fig, fig_dir, name, dpi=200):
    """Save a matplotlib figure as PNG in the figures folder and return its path."""
    os.makedirs(fig_dir, exist_ok=True)
    path = op.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("saved", path)
    return path


# --------------------------------------------------------------------------
# Functional preprocessing (step 1.2) -- added
# --------------------------------------------------------------------------
def framewise_displacement(par_path, radius=50.0):
    """
    Framewise displacement (Power et al. 2012) from an MCFLIRT `.par` file.
    MCFLIRT columns: 3 rotations (radians) then 3 translations (mm). Rotations are
    converted to mm of arc on a sphere of `radius` mm (50 mm ~ cortex), then
    FD[t] = sum |param[t] - param[t-1]|. FD[0] = 0 by convention.
    """
    par = np.loadtxt(par_path)
    d = np.abs(np.diff(par, axis=0))
    d[:, :3] *= radius
    return np.concatenate([[0.0], d.sum(axis=1)])


def scrub_regressors(fd, thresh):
    """
    One-hot "spike" regressors, one column per volume with FD > thresh.
    Returns (indices of flagged volumes, array of shape (n_vols, n_flagged)).
    Including them in a GLM removes those volumes' influence without deleting
    them (so the timing of the experimental design is preserved).
    """
    idx = np.where(np.asarray(fd) > thresh)[0]
    spikes = np.zeros((len(fd), len(idx)))
    spikes[idx, np.arange(len(idx))] = 1.0
    return idx, spikes


def dice(mask_a, mask_b):
    """Dice overlap between two boolean arrays of the same shape."""
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    return 2.0 * (a & b).sum() / (a.sum() + b.sum())


def edge_alignment_curve(fmri_mean, anat, mask, max_shift=3, sigma=1.0):
    """
    Alignment check that works across contrasts (T1w vs EPI): correlate the gradient
    magnitude (edge strength) of the two images inside `mask` while translating the fMRI
    edge map voxel by voxel along each axis. If the images are in register the correlation
    peaks at shift 0. Both images must be on the same grid. Independent of how complete the
    brain mask is (dropout regions simply do not contribute).
    Returns (shifts in voxels, {axis: correlations}) with axis 0/1/2 = i/j/k.
    """
    from scipy.ndimage import gaussian_gradient_magnitude
    g_f = gaussian_gradient_magnitude(np.asarray(fmri_mean, float), sigma)
    g_a = gaussian_gradient_magnitude(np.asarray(anat, float), sigma)
    mask = np.asarray(mask, bool)
    shifts = np.arange(-max_shift, max_shift + 1)
    out = {}
    for axis in range(3):
        vals = []
        for s in shifts:
            m = mask & np.roll(mask, s, axis=axis)
            vals.append(np.corrcoef(np.roll(g_f, s, axis=axis)[m], g_a[m])[0, 1])
        out[axis] = np.array(vals)
    return shifts, out
