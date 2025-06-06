from numba import njit, prange
import numpy as np

@njit(parallel=True)
def leakage_(data_o: np.ndarray, data_s: np.ndarray, axis: int) -> float:
    shape = data_o.shape[axis]
    p_corrs = np.full((shape, shape), np.nan)
    s_corrs = np.full((shape, shape), np.nan)
    f_l_corrs = np.full((shape, shape), np.nan) # f_l refers to full leakage 

    C1 = 1e-4
    C2 = 9e-4
    
    for i in prange(shape):
        for j in range(shape):
            if axis == 0:
                slice_o = data_o[i, :, :]
                slice_s = data_s[j, :, :]

            elif axis == 1:
                slice_o = data_o[:, i, :]
                slice_s = data_s[:, j, :]

            else:
                slice_o = data_o[:, :, i]
                slice_s = data_s[:, :, j]


            flat_o = slice_o.ravel()
            flat_s = slice_s.ravel()
            valid_mask = ~np.isnan(flat_o) & ~np.isnan(flat_s)
            flat_o = flat_o[valid_mask]
            flat_s = flat_s[valid_mask]
			########################## PEARSON ##########################
            mean_o = np.mean(flat_o)
            std_o = np.sqrt(np.sum((flat_o - mean_o) ** 2) / (len(flat_o) - 1))#np.std(flat_o, ddof=1)
            mean_s = np.mean(flat_s)
            std_s = np.sqrt(np.sum((flat_s - mean_s) ** 2) / (len(flat_s) - 1))#np.std(flat_s, ddof=1)
            #p_corr = np.sum((flat_o - mean_o) * (flat_s - mean_s)) / (len(flat_o) * std_o * std_s)
            p_corr = np.sum((flat_o - mean_o) * (flat_s - mean_s)) / ((len(flat_o) - 1) * std_o * std_s)   

            if p_corr >= 0.99999:
                p_corrs[i,j] = 1.0
            else:
                 p_corrs[i,j] = p_corr
            ########################## SSIM ##########################
            # RESCALING TO [0,1]
            flat_o = (flat_o - flat_o.min()) / (flat_o.max() - flat_o.min())
            flat_s = (flat_s - flat_s.min()) / (flat_s.max() - flat_s.min())
            mean_o = np.mean(flat_o)
            mean_s = np.mean(flat_s)
            
            sigma_o = flat_o.var()
            sigma_s = flat_s.var()
            sigma_xy = ((flat_o - mean_o) * (flat_s - mean_s)).mean()
            
            numerator = (2 * mean_o * mean_s + C1) * (2 * sigma_xy + C2)
            denominator = (mean_o**2 + mean_s**2 + C1) * (sigma_o + sigma_s + C2)
            ssim_val = numerator / denominator

            s_corrs[i, j] = ssim_val    
            ########################## ALLCLOSE ##########################
            if not np.any(slice_o) and not np.any(slice_o):
                #print("found zero vectors. ",i,j)
                pass
            else:
                f_l_corrs[i, j] = np.allclose(slice_o, slice_s)    
    return p_corrs, s_corrs, f_l_corrs

    

@njit(parallel=True)
def leakage_2D(data_o: np.ndarray, data_s: np.ndarray) -> float:
    shape = data_o.shape[0]
    p_corrs = np.full(shape, np.nan)
    s_corrs = np.full(shape, np.nan)
    f_l_corrs = np.full(shape, np.nan) 
    C1 = 1e-4
    C2 = 9e-4
    for i in prange(shape):

        cube_o = data_o[i]
        cube_s = data_s[i]


        flat_o = cube_o.ravel()
        flat_s = cube_s.ravel()
        valid_mask = ~np.isnan(flat_o) & ~np.isnan(flat_s)
        flat_o = flat_o[valid_mask]
        flat_s = flat_s[valid_mask]
        ########################## PEARSON ##########################
        mean_o = np.mean(flat_o)
        std_o = np.sqrt(np.sum((flat_o - mean_o) ** 2) / (len(flat_o) - 1))#np.std(flat_o, ddof=1)
        mean_s = np.mean(flat_s)
        std_s = np.sqrt(np.sum((flat_s - mean_s) ** 2) / (len(flat_s) - 1))#np.std(flat_s, ddof=1)
        p_corr = np.sum((flat_o - mean_o) * (flat_s - mean_s)) / ((len(flat_o) - 1) * std_o * std_s)   

        if p_corr >= 0.99999:
            p_corrs[i] = 1.0
        else:
             p_corrs[i] = p_corr


        ########################## SSIM ##########################
        # RESCALING TO [0,1]
        flat_o = (flat_o - flat_o.min()) / (flat_o.max() - flat_o.min())
        flat_s = (flat_s - flat_s.min()) / (flat_s.max() - flat_s.min())
        mean_o = np.mean(flat_o)
        mean_s = np.mean(flat_s)
        
        sigma_o = flat_o.var()
        sigma_s = flat_s.var()
        sigma_xy = ((flat_o - mean_o) * (flat_s - mean_s)).mean()
        
        numerator = (2 * mean_o * mean_s + C1) * (2 * sigma_xy + C2)
        denominator = (mean_o**2 + mean_s**2 + C1) * (sigma_o + sigma_s + C2)
        ssim_val = numerator / denominator

        s_corrs[i] = ssim_val    
        ########################## ALLCLOSE ##########################
        if not np.any(cube_o) and not np.any(cube_o):
            #print("found zero vectors. ",i)
            pass
        else:
            f_l_corrs[i] = np.allclose(cube_o, cube_s, atol=1e-11) # by default atol=1e-08  
    return p_corrs, s_corrs,f_l_corrs

