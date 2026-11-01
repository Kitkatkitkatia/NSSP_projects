import os
import os.path as op
import numpy as np
import pandas as pd 
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.plotting import plot_stat_map, plot_roi
from sklearn.decomposition import PCA

from helpers import save_fig

def first_figure_background_masking(data):
    """
    Creates a mask from the first volume to filter all voxels that aren't background and applies it to data.
    Same method as in Lab.
    """
    first_volume = data[:, :, :, 0] # First volume
    
    # Creating a mask to filter all voxels that aren't background, since the volume's background 
    # voxels are the same as all following volumes we don't have to reproduce it for each volume
    slice_non_background = first_volume > 500
    
    # Vectorize : Taking only non-zero voxels into a vector 
    samples = data[slice_non_background, :].T
    return slice_non_background, samples 

def anatomic_masking(data, mask_path='./derivatives/func/func_brain_mask.nii.gz'):
    """
    Uses mask produced in part 1 to filter voxels that aren't background and applies it to data.
    Method recommanded by assistant.
    """
    # Load the mask file
    mask_img = nib.load(mask_path)
    mask_data = np.asanyarray(mask_img.dataobj)
    
    # Create the boolean mask 
    slice_non_background = mask_data > 0
    
    samples = data[slice_non_background, :].T    
    return slice_non_background, samples

def plot_variance(pca, nb_clusters=0, fig_dir= './figures/part2'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ratios = pca.explained_variance_ratio_
    cumulative_ratios = np.cumsum(ratios)

    ax[0].plot(np.arange(1, len(ratios)+1), ratios, label='explained variance ratios', c='k')
    ax[1].plot(np.arange(1, len(cumulative_ratios)+1), cumulative_ratios, 
               label='cumulative explained variance', c='k')
    if nb_clusters > 0:
        ax[0].scatter([nb_clusters], [ratios[nb_clusters-1]], [150], 
                      marker='x', color='r', label='cutoff')
        ax[1].hlines(y=cumulative_ratios[nb_clusters-1], xmin=1, 
                     xmax=len(ratios)-1, linestyle='--', color='r', label='cutoff')

    for k in range(2):
        ax[k].set_xlabel("Components #", size=15)
        ax[k].legend(prop={"size": 15})  
        ax[k].tick_params(axis="both", which="major", labelsize=15)

        ax[k].grid(True, alpha=0.3)

    fig.suptitle(f"PCA Variance (n_components={len(ratios)})", fontsize=16)
    plt.tight_layout()
    plt.show()  
    if nb_clusters > 0:
        save_fig(fig, fig_dir, f"PCA_Variance_{len(ratios)}_components")

def reshape_components(pca, vol_shape, slice_non_background):
    pca_clusters = [] # List of spatial components
    
    for comp in pca.components_:
        # Create voxel of sample size 
        vol = np.zeros(vol_shape) 
        
        # Insert the 1D non-background voxel weights back into their 3D coordinates
        vol[slice_non_background] = comp
        
        # Append the reconstructed 3D volume to the list
        pca_clusters.append(vol) 
    return pca_clusters

def plot_axial_clusters(pca_clusters, affine, nb_components, nb_slices=5, fig_dir= './figures/part2'):
    for idx, cluster_3d in enumerate(pca_clusters):
        
        # Convert the 3D numpy array to a NIfTI image
        comp_img = nib.Nifti1Image(cluster_3d, affine)
    
        # Compute the maximum absolute loading across components to unify the scale, 
        # because otherwise hard to compare cuts
        global_vmax = np.max([np.abs(cluster_3d).max() for cluster_3d in pca_clusters])
        
        # Plot separately in axial view only with multiple slices
        plot_stat_map(
            comp_img, 
            bg_img=None, 
            threshold=1e-3,
            vmax=global_vmax,
            display_mode='z',        # Restricts visualization to axial (Z) slices only
            cut_coords=nb_slices,            # Sets the number of axial slices to display
            black_bg=True, 
            title=f'PCA Cluster {idx}'
        )
        
        fig = plt.gcf()
        save_fig(fig, fig_dir, f"figure_axial_{nb_components}_components_cluster_{idx}")
        
        plt.show()