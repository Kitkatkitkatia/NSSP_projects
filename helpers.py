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
