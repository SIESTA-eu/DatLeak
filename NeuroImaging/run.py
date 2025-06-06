import numpy as np
import warnings, sys, os, tqdm, time, copy, traceback
warnings.filterwarnings("ignore")
from numba import njit, prange
import matplotlib.pyplot as plt
import seaborn as sns
from src.report import report
from src.report_meg import report_meg

from src.loader import *
from src.viz import *
from src.logging_config import setup_logging
from src.utils import *
from src.leakage_ import *

logger = setup_logging()

def main(original_path, scrambled_path, subject_name,task=None, run_=None, Dim4 = False,r=False):
    
    r = False if r == "False" else True
    
    ext = original_path.split("/")[-1]
    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        original = nii_reader(original_path).get_data()
        scrambled = nii_reader(scrambled_path).get_data()
        if len(list(original.shape)) == 4:
            Dim4 = True
    elif ext.endswith(".fif") \
        or ext.endswith(".fif.gz") \
        or ext.endswith(".vhdr") \
        or ext.endswith(".vhdr.gz"):
        original = neuro_reader(original_path).get_data()
        scrambled = neuro_reader(scrambled_path).get_data()

    else: 
        print(f" - Unsupported file type.")
    
    def result_dict():
        keys = [
            "p_corr_fl", "s_corr_fl", "f_corr_fl",
            "p_corr_pl_avg", "p_corr_pl_min", "p_corr_pl_max",
            "s_corr_pl_avg", "s_corr_pl_min", "s_corr_pl_max",
            "f_corr_pl_avg", "f_corr_pl_min", "f_corr_pl_max"
            ]
        return {key: [] for key in keys}

        
    results = {"x": result_dict(), "y": result_dict(), "z": result_dict()}
    axes = ["x", "y", "z"]
    
    #########################################
    #             VIZ                       #
    #########################################
    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        if len(list(original.shape)) == 4:
            viz_(original[...,0], slice_=original.shape[0]//2, png_title= "original.png")
            viz_(scrambled[...,0], slice_=original.shape[0]//2, png_title= "scrambled.png")    

        else:
            viz_(original, slice_=original.shape[0]//2, png_title= "original.png")
            viz_(scrambled, slice_=original.shape[0]//2, png_title= "scrambled.png")


    elif ext.endswith(".fif") or ext.endswith(".fif.gz"):
        viz_psd(original_path, type_="fif", which="original")
        viz_psd(scrambled_path, type_="fif",which="scrambled")
    elif ext.endswith(".vhdr") or ext.endswith(".vhdr.gz"):
        viz_psd(original_path, type_="vhdr", which="original")
        viz_psd(scrambled_path, type_="vhdr",which="scrambled")
    else: print(f" - Unsupported file type.")
    #########################################
    #             SPATIAL ANALYSIS          #
    #########################################
    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        print(f" - Spatial Analysis")
        print(f" - Averaged over time dimension")
        for i, axis in enumerate(axes):
            if Dim4:
                logger.info(f"Spatial Leakage Analysis on 'func'")
                p_corrs, s_corrs, f_l_corrs = list(), list(), list()
                for time in range(original.shape[-1]):
                    p_corrs_aux, s_corrs_aux, f_l_corrs_aux = leakage_(original[:, :, :, time], scrambled[:, :, :, time], axis=i)
                    p_corrs.append(p_corrs_aux)
                    s_corrs.append(s_corrs_aux)
                    f_l_corrs.append(f_l_corrs_aux)
                    
                p_corrs = np.mean(np.stack(p_corrs, axis=0), axis=0)
                s_corrs = np.mean(np.stack(s_corrs, axis=0), axis=0)
                f_l_corrs = np.mean(np.stack(f_l_corrs, axis=0), axis=0)
                
            if not Dim4:
                logger.info(f"Spatial Leakage Analysis on 'anat'")
                p_corrs, s_corrs, f_l_corrs = leakage_(original, scrambled, axis=i)
            pfl, avg_p_l, min_p_l, max_p_l = summary(p_corrs, type_="nii")
            sfl, avg_s_l, min_s_l, max_s_l = summary(s_corrs, type_="nii")
            ffl, avg_f_l, min_f_l, max_f_l = summary(f_l_corrs, type_="nii")
    
            result = results[axis]
            result["f_corr_fl"].append(ffl)
            result["f_corr_pl_avg"].append(avg_f_l)
            result["f_corr_pl_min"].append(min_f_l)
            result["f_corr_pl_max"].append(max_f_l)
       
            result["p_corr_fl"].append(pfl)
            result["p_corr_pl_avg"].append(avg_p_l)
            result["p_corr_pl_min"].append(min_p_l)
            result["p_corr_pl_max"].append(max_p_l)    

            result["s_corr_fl"].append(sfl)
            result["s_corr_pl_avg"].append(avg_s_l)
            result["s_corr_pl_min"].append(min_s_l)
            result["s_corr_pl_max"].append(max_s_l)
            left = np.concatenate([arr[arr >= 0.99999] for arr in np.nanmax(p_corrs,axis=1)]).shape[0]
            right = p_corrs.shape[0]# - np.isnan(np.nanmax(p_corrs,axis=1)).sum()
            print(f"\t - Dimension[{axis.upper()}]: \tFull Leakage: {left}/{right} slices\tPartial Leakage: {round(max_p_l, 2)}") # \tIdentical: {round(ffl, 2)}%
            #print("HERE",p_corrs.shape, np.isnan(np.nanmax(p_corrs)).sum(), s_corrs.shape, np.isnan(np.nanmax(s_corrs)).sum(),f_l_corrs.shape, np.isnan(np.nanmax(f_l_corrs)).sum())
            viz_report(p_corrs, s_corrs, f_l_corrs,loop=i, file_shape=original.shape)
       
    #########################################
    #             SPATIOTEMPORAL            #
    #########################################
        if Dim4:
            print(f" - Spatiotemporal voxel-wise Analysis")
            o_p = cubeT(original, cube_size=1, stride=1)
            o_s = cubeT(scrambled, cube_size=1, stride=1)
            print(f"\t - Total voxels: {len(o_p)} \tShape {o_p[0].shape}")
            p_corrs, s_corrs,f_l_corrs = leakage_2D(np.array(o_p),np.array(o_s))
            full_leakage = True if np.nanmax(p_corrs) >= 0.99999 else False
            partial_leakage = np.round(np.nanmax(p_corrs),2) 
            #identical = (np.argwhere(f_l_corrs >= .99999).shape[0] / f_l_corrs.shape[0]) * 100
            print(f"\t - SpatioTemporal: \tFull Leakage: {np.argwhere(p_corrs >= .99999).shape[0]}/{p_corrs.shape[0]} voxels \tPartial Leakage {partial_leakage}") # \tIdentical: {identical}%
            
            viz_spatiotemporal(p_corrs, s_corrs,f_l_corrs, file_shape=original.shape)
    #########################################
    #             FIF                       #
    #########################################
    elif ext.endswith(".fif") or ext.endswith(".fif.gz") or ext.endswith(".vhdr") or ext.endswith(".vhdr.gz"):
        p_corrs, s_corrs, f_l_corrs = leakage_2D(original,scrambled)
        pfl, avg_p_l, min_p_l, max_p_l = summary(p_corrs, type_="fif")
        sfl, avg_s_l, min_s_l, max_s_l = summary(s_corrs, type_="fif")
        ffl, avg_f_l, min_f_l, max_f_l = summary(f_l_corrs, type_="fif")
        results = {"time": result_dict()}
        result = results["time"]
        result["f_corr_fl"].append(ffl)
        result["f_corr_pl_avg"].append(avg_f_l)
        result["f_corr_pl_min"].append(min_f_l)
        result["f_corr_pl_max"].append(max_f_l)
       
        result["p_corr_fl"].append(pfl)
        result["p_corr_pl_avg"].append(avg_p_l)
        result["p_corr_pl_min"].append(min_p_l)
        result["p_corr_pl_max"].append(max_p_l)    
        result["s_corr_fl"].append(sfl)
        result["s_corr_pl_avg"].append(avg_s_l)
        result["s_corr_pl_min"].append(min_s_l)
        result["s_corr_pl_max"].append(max_s_l)
        
        full_leakage = True if np.nanmax(p_corrs) >= 0.99999 else False
        partial_leakage = np.round(np.nanmax(p_corrs),4) if np.round(np.nanmax(p_corrs),2) == 0.0 else np.round(np.nanmax(p_corrs),2)  
        #identical = (np.argwhere(f_l_corrs >= .99999).shape[0] / f_l_corrs.shape[0]) * 100
        print(f"\t - Temporal: \tFull Leakage: {np.argwhere(p_corrs >= .99999).shape[0]}/{p_corrs.shape[0]} channels \tPartial Leakage {partial_leakage}") # \tIdentical: {identical}%
        #results = {"fl": full_leakage, "pl": partial_leakage}
        
        viz_spatiotemporal(p_corrs, s_corrs, f_l_corrs)
        
    else: print(f" - Unsupported file type.")

    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        if Dim4: 
            fl, pl = leak_detect(results, "func")
        if not Dim4:
            fl, pl = leak_detect(results, "anat")
    elif ext.endswith(".fif") or ext.endswith(".fif.gz") or ext.endswith(".vhdr") or ext.endswith(".vhdr.gz"):
        fl, pl = leak_detect(results, "vhdr")
        
    #########################################
    #             REPORT                    #
    #########################################
    
    if r:
        report_output = subject_name +"_"+ task +"_"+ run_ if task else subject_name
        
        if ext.endswith(".nii") or ext.endswith(".nii.gz"):
            report(
            top_image_paths=[
                "img/original.png",
                "img/scrambled.png"
            ],
            bottom_image_paths=[
                "img/correlations dist along dimension 0.png",
                "img/correlations dist along dimension 1.png",
                "img/correlations dist along dimension 2.png"
            ],
            top_titles=[
                "Original image",
                "Scrambled image"
            ],
            bottom_titles=[
                "x dim",
                "y dim",
                "z dim"
            ],
            output_html="report/"+report_output+".html", 
            partial_leakage=round(pl,2) * 100,
            
           p_leakage_x_min=np.mean(results["x"]["p_corr_pl_min"]),
           p_leakage_x_max=np.mean(results["x"]["p_corr_pl_max"]),
           p_leakage_x_avg=np.mean(results["x"]["p_corr_pl_avg"]), 
           p_leakage_y_min=np.mean(results["y"]["p_corr_pl_min"]),
           p_leakage_y_max=np.mean(results["y"]["p_corr_pl_max"]),
           p_leakage_y_avg=np.mean(results["y"]["p_corr_pl_avg"]), 
           p_leakage_z_min=np.mean(results["z"]["p_corr_pl_min"]),
           p_leakage_z_max=np.mean(results["z"]["p_corr_pl_max"]),
           p_leakage_z_avg=np.mean(results["z"]["p_corr_pl_avg"]),
            
           s_leakage_x_min=np.mean(results["x"]["s_corr_pl_min"]),
           s_leakage_x_max=np.mean(results["x"]["s_corr_pl_max"]),
           s_leakage_x_avg=np.mean(results["x"]["s_corr_pl_avg"]), 
           s_leakage_y_min=np.mean(results["y"]["s_corr_pl_min"]),
           s_leakage_y_max=np.mean(results["y"]["s_corr_pl_max"]),
           s_leakage_y_avg=np.mean(results["y"]["s_corr_pl_avg"]), 
           s_leakage_z_min=np.mean(results["z"]["s_corr_pl_min"]),
           s_leakage_z_max=np.mean(results["z"]["s_corr_pl_max"]),
           s_leakage_z_avg=np.mean(results["z"]["s_corr_pl_avg"]),           
           full_leakage=fl)
        elif ext.endswith(".fif") or ext.endswith(".fif.gz") or ext.endswith(".vhdr") or ext.endswith(".vhdr.gz"):
            report_meg(
            top_image_paths=[
                "img/original.png",
                "img/scrambled.png"
            ],
            bottom_image_paths=[
                "img/correlations dist along Time.png"],
            top_titles=[
                "Power Spectral Density [Original]",
                "Power Spectral Density [Scrambled]"
            ],
            bottom_titles=["Time Dimension"],
            output_html="report/"+report_output+".html", 
            partial_leakage=round(pl,2) * 100,
            
           p_leakage_time_min=np.mean(results["time"]["p_corr_pl_min"]),
           p_leakage_time_max=np.mean(results["time"]["p_corr_pl_max"]),
           p_leakage_time_avg=np.mean(results["time"]["p_corr_pl_avg"]), 
            
           s_leakage_time_min=np.mean(results["time"]["s_corr_pl_min"]),
           s_leakage_time_max=np.mean(results["time"]["s_corr_pl_max"]),
           s_leakage_time_avg=np.mean(results["time"]["s_corr_pl_avg"]), 
             
           full_leakage=fl)    
      
    return np.round(pl,2),fl

def pair(o, s, data_type):
    if o.split("/")[-1] != s.split("/")[-1]:
        logger.info(f"Original and Scrambled filename did not match: \n\tOriginal: {original.split('/')[-1]}\tScrambled: {scrambled.split('/')[-1]}")
        print(f" - Warning: Original and Scrambled filename does not match.\n\t - Original: {original.split('/')[-1]}\t - Scrambled: {scrambled.split('/')[-1]}")
        print(f" - Mentioned subject will be skipped.")
        return

    logger.info(f"Comparing files:\nOriginal: {o}\nScrambled: {s}")
    
    file_ = neuro_reader(o).get_data() if "fif" in o or "vhdr" in o else nii_reader(o).get_data()
    if len(list(file_.shape)) == 3:
        data_type = "anat"
    subject_name, task, run_ = parse_info(o, data_type)
    logger.info(f"Subject ID: {subject_name}")
    if run_: logger.info(f"{'-' * 30}{subject_name}-{task}-{run_}{'-' * 30}")
    print("#" * 40)
    print(f" - Subject ID: {subject_name}")
    if task: print(f" - Task: {task}")
    if run_: 
        if data_type == "vhdr":
            print(f" - Session: {run_}")
        else:
            print(f" - Run: {run_}")
    print(f" - Shape: {file_.shape}")
    print("#" * 40)

    args = {
        "original_path": o,
        "scrambled_path": s,
        "subject_name": subject_name,
        "r": sys.argv[3],
    }
    if task: args["task"] = task
    if run_: args["run_"] = run_

    pl, fl = main(**args)

    print(f" - Partial Leakage: {pl}")
    print(f" - Full Leakage: {fl}")
    if fl > 0.0:
        print(" - Please consider applying scramble on your dataset again.")
        logger.info(f"Leakage Detected.")
    
def process_(original_list, scrambled_list, data_type):
    for o, s in tqdm.tqdm(zip(original_list, scrambled_list), total=len(original_list)):
        try:
            pair(o, s, data_type)  
        except Exception as e:
            traceback.print_exc()

    #########################################
    #             MAIN CALL                 #
    #########################################  
if __name__ == "__main__":
    start_time = time.time()
    try:
        if len(sys.argv) < 3:
            raise ValueError("Insufficient arguments")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

 
    usecase = 2.5
    if usecase == None:
        orig = sys.argv[1]
        scra = sys.argv[2]
    elif usecase == 2.2:
        orig = "/staff/vincentajoubi/Trash/wp15-usecase-2.1_aux/usecase-2.2/input/"
        scra = "/staff/vincentajoubi/Trash/wp15-usecase-2.1_aux/usecase-2.2/custom_output/"
    elif usecase == 2.3:
        orig = "/staff/vincentajoubi/wp15-chrono-T/usecase-2.3/input/"
        scra = "/staff/vincentajoubi/wp15-chrono-T/usecase-2.3/scrambled/"
    elif usecase == 2.4:
        orig = "/staff/vincentajoubi/wp15-chrono-T/usecase-2.4/input/"# sub-001/
        scra = "/staff/vincentajoubi/wp15-chrono-T/usecase-2.4/scrambled/"
    elif usecase == 2.5:
        orig = "/staff/vincentajoubi/wp15-chrono-T/usecase-2.5/input/"
        scra = "/staff/vincentajoubi/wp15-chrono-T/usecase-2.5/custom_output/" #                 
             
    if detect_file(orig) == "nii":
        process_(fetch_files(orig).nii_(), fetch_files(scra).nii_(), data_type="func")
    if detect_file(orig) == "fif":
        process_(fetch_files(orig).fif_(), fetch_files(scra).fif_(), data_type="fif")
    if detect_file(orig) == "vhdr":
        process_(fetch_files(orig).vhdr_(), fetch_files(scra).vhdr_(), data_type="vhdr")

        
    total_time = time.time() - start_time
    print(f"\nTotal time taken: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    logger.info(f"Total time taken: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")	
