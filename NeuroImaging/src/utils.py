import numpy as np
import os

def detect_file(folder):
    has_fif = False
    has_nii = False
    has_vhdr = False
    
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".fif") or f.endswith(".fif.gz"):
                has_fif = True
                break 
            elif f.endswith(".vhdr") or f.endswith(".vhdr.gz"):
                has_vhdr = True
                break  
            elif f.endswith(".nii.gz") or "_T1w.nii.gz" in f:
                has_nii = True
        if has_fif:
            break  
    if has_fif:
        return "fif"
    if has_vhdr:
        return "vhdr"
    elif has_nii:
        return "nii"
    else:
        return None 


def leak_detect(results, type_):
    if type_ == "anat" or type_ == "func": 
        fl_p = np.nanmean([results["x"]['p_corr_fl'], results["z"]['p_corr_fl'], results["y"]['p_corr_fl']])
        fl_f = np.nanmean([results["x"]['f_corr_fl'], results["z"]['f_corr_fl'], results["y"]['f_corr_fl']])
        if fl_p == 0.0 and fl_f == 0.0:
            fl = False
        else: fl = True
        pl = np.nanmean([results["x"]['p_corr_pl_max'], results["y"]['p_corr_pl_max'],results["z"]['p_corr_pl_max']])
        return fl, pl

    if type_ == "fif":
        fl, pl = results["fl"], results["pl"]
        return fl, pl
    if type_ == "vhdr":
        fl_p = np.nanmean(results["time"]['p_corr_fl'])
        fl_f = np.nanmean(results["time"]['f_corr_fl'])
        if fl_p == 0.0 and fl_f == 0.0:
            fl = False
        else: fl = True
        pl = np.nanmean(results["time"]['p_corr_pl_max'])
        return fl, pl


def parse_info(file_, type_):
    parts = file_.split('/')[-1].split('_')
    if type_ == 'fif':
        return parts[0], parts[2].split('-')[1], parts[3]
    elif type_ == 'vhdr':
        return parts[0], parts[2].split('-')[1], parts[1].split('-')[1]
    elif type_ == 'func':
        return parts[0], parts[1].split('-')[1], parts[2] # o.split('/')[-1].split('_')[1].split('-')[1] 
    elif type_ == 'anat':
        return parts[0], None, None
    else:
        raise ValueError("Unknown data type")

def summary(matrix,type_=None):
    if type_ == "nii":
        f_l = (np.concatenate([arr[arr >= 0.99999] for arr in np.nanmax(matrix,axis=1)]).shape[0] / 
                    (matrix.shape[0] - np.isnan(np.nanmax(matrix,axis=1)).sum())) * 100
    elif type_ == "fif":
        f_l = (np.argwhere(matrix >= .99999).shape[0] / matrix.shape[0] - np.isnan(np.nanmax(matrix)).sum()) * 100
    try:
        avg_p_l = np.nanmean(matrix)
    except ValueError:
        avg_p_l= np.inf
    try:
        max_p_l = np.nanmax(matrix)
    except ValueError:
        max_p_l= np.inf
    try:
        min_p_l = np.nanmin(matrix)
    except ValueError:
        min_p_l= np.inf
        
    return f_l, avg_p_l, min_p_l,max_p_l

def cubeT(data, cube_size=16, stride=4):
    x_max, y_max, z_max, t_max = data.shape
    patches = []

    for x in range(0, x_max - cube_size + 1, stride):
        for y in range(0, y_max - cube_size + 1, stride):
            for z in range(0, z_max - cube_size + 1, stride):
                cube = data[
                    x : x + cube_size,
                    y : y + cube_size,
                    z : z + cube_size,
                    :
                ]  
                patches.append(cube)
    return patches
