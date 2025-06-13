import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import rotate


def viz_report(p_corrs, s_corrs, f_l_corrs,loop, file_shape=None):

    hist_kws = {'edgecolor': 'black', 'linewidth': 0.5 }
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(14, 12), sharex=False, gridspec_kw={'height_ratios': [3, 3, 3, 3, 3]})
    num_bins = 200
    sns.histplot(p_corrs.flatten(), bins=num_bins, kde=True, color='skyblue', ax=ax1, **hist_kws)
    ax1.set_title(f"File Shape: {file_shape}\nDistribution of Pearson Correlations")
    ax1.set_xlabel("Correlation Coefficient")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle='--', alpha=0.5)
	
    #num_bins = min(50, max(1, int(np.sqrt(len(np.unique(np.nanmax(p_corrs, axis=1)))))))
    #if num_bins < 2: 
    #    num_bins = 100
    sns.histplot(list(np.nanmax(p_corrs, axis=1)), bins = num_bins, kde=True, color='skyblue', ax=ax2, **hist_kws) 
    #sns.histplot(p_corrs.diagonal(), bins = num_bins, kde=True, color='skyblue', ax=ax2, **hist_kws) 
    ax2.set_title("Distribution of Pearson MAX")
    ax2.set_xlabel("Correlation Coefficient")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, linestyle='--', alpha=0.5)

    sns.histplot(s_corrs.flatten(), bins=num_bins, kde=True, color='skyblue', ax=ax3, **hist_kws)
    ax3.set_title("Distribution of SSIM Correlations")
    ax3.set_xlabel("Correlation Coefficient")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, linestyle='--', alpha=0.5)
	
    #num_bins = min(50, max(1, int(np.sqrt(len(np.unique(np.nanmax(s_corrs, axis=1)))))))
    #if num_bins == 1 or num_bins < 2: 
    #    num_bins = 100
    sns.histplot(np.nanmax(s_corrs, axis=1), bins = num_bins, kde=True, color='skyblue', ax=ax4, **hist_kws)
    #sns.histplot(s_corrs.diagonal(), bins = num_bins, kde=True, color='skyblue', ax=ax4, **hist_kws) 
    ax4.set_title("Distribution of SSIM MAX")
    ax4.set_xlabel("Correlation Coefficient")
    ax4.set_ylabel("Frequency")    
    ax4.grid(True, linestyle='--', alpha=0.5)

    sns.histplot(np.nanmax(f_l_corrs, axis=1), bins=num_bins, kde=True, color='skyblue', ax=ax5, **hist_kws)
    ax5.set_title("AllClose Distribution")
    ax5.set_xlabel("Values")
    ax5.set_ylabel("Frequency")
    ax5.grid(True, linestyle='--', alpha=0.5)
	
    plt.tight_layout()
    plt.savefig("img/"+"correlations dist along dimension " + str(loop) + ".png")
    plt.close()
    
def viz_spatiotemporal(p_corrs, s_corrs,f_l_corrs, file_shape=None, ext=None):
    
    hist_kws = {'edgecolor': 'black', 'linewidth': 0.5}
    if ext == "nii":
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=False, gridspec_kw={'height_ratios': [3, 3, 3]})
        num_bins = 200
        #num_bins = min(50, max(1, int(np.sqrt(len(np.unique(np.nanmax(p_corrs)))))))
        #if num_bins < 2: 
        #    num_bins = 100
        sns.histplot(p_corrs.flatten(), bins=num_bins, kde=True, color='skyblue', ax=ax1, **hist_kws)
        ax1.set_title(f"File Shape: {file_shape}\nSpatioTemporal Pearson Correlations")
        ax1.set_xlabel("Correlation Coefficient")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, linestyle='--', alpha=0.5)
    
        sns.histplot(s_corrs.flatten(), bins = num_bins, kde=True, color='skyblue', ax=ax2, **hist_kws)
        ax2.set_title("SpatioTemporal SSIM score")
        ax2.set_xlabel("Correlation Coefficient")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, linestyle='--', alpha=0.5)

        sns.histplot(f_l_corrs.flatten(), bins=num_bins, kde=True, color='skyblue', ax=ax3, **hist_kws)
        ax3.set_title("SpatioTemporal unique vectors")
        ax3.set_xlabel("Correlation Coefficient")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, linestyle='--', alpha=0.5)
    
    elif ext == "fif" or ext == "vhdr":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False, gridspec_kw={'height_ratios': [3, 3]})
        num_bins = 200
        sns.histplot(p_corrs.flatten(), bins=num_bins, kde=True, color='skyblue', ax=ax1, **hist_kws)
        ax1.set_title(f"File Shape: {file_shape}\nSpatioTemporal Pearson Correlations")
        ax1.set_xlabel("Correlation Coefficient")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, linestyle='--', alpha=0.5)

        sns.histplot(f_l_corrs.flatten(), bins=num_bins, kde=True, color='skyblue', ax=ax2, **hist_kws)
        ax2.set_title("SpatioTemporal unique vectors")
        ax2.set_xlabel("Correlation Coefficient")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("img/correlations dist along Time.png")
    plt.close()
    #plt.show()
    
def viz_(data, slice_=int, png_title= None):
    if len(list(data.shape)) == 4:
        data = data[...,0]

    cmap="gray"
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    x_slice = data[slice_, :, :]  
    y_slice = data[:, slice_, :]  
    z_slice = data[:, :, slice_] 
    vmin = np.percentile(x_slice, 2)
    vmax = np.percentile(x_slice, 98)

    x_slice = rotate(x_slice, angle=90, reshape=True) 
    axes[0].imshow(x_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('x dimension[axial]')
    axes[0].axis("off")
    y_slice = rotate(y_slice, angle=90, reshape=True) 
    axes[1].imshow(y_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('y dimension[sagittal]')
    axes[1].axis("off")
    z_slice = rotate(z_slice, angle=90*3, reshape=True) 
    axes[2].imshow(z_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title('z dimension[coronal]')
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("img/"+png_title)
    plt.close()


def viz_psd(data_path, type_=None, which=None):
    
    import mne
    if type_ == "fif":
        d = mne.io.read_raw_fif(data_path, preload=True)
        psd = d.compute_psd(method='welch', fmin=1, fmax=100)
    if type_ == "vhdr":
        d = mne.io.read_raw_brainvision(data_path, preload=True)
        eeg_channels = list(d.ch_names)
        eog_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
        d.set_channel_types({ch: 'eeg' for ch in eeg_channels})
        d.set_channel_types({ch: 'eog' for ch in eog_channels})
        psd = d.compute_psd(picks='eeg', fmin=1, fmax=100)
    
    psd.plot(average=True, picks='data',color="blue")  
    plt.title(f"Power Spectral Density (PSD)[{which}]")
    plt.tight_layout()
    plt.savefig(f"img/{which}.png")
    plt.close()

