import numpy as np
import warnings, sys, os, tqdm, time, copy, traceback, random
warnings.filterwarnings("ignore")
from numba import njit, prange
import matplotlib.pyplot as plt
import seaborn as sns
from src.report import report
from src.report_meg import report_meg
from collections import Counter
from src.loader import *
from src.viz import *
from src.logging_config import setup_logging
from src.utils import *
from src.leakage_ import *

logger = setup_logging()

def main(original_path, scrambled_path, subject_name,task=None, run_=None, Dim4 = False,r=False):
    
    if r == "False":
        r = False
    elif r == "True":
        r = True
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
        if Dim4:
            print(f" - Averaged over time dimension")
        for i, axis in enumerate(axes):
            if Dim4:
                p_corrs, s_corrs, f_l_corrs = list(), list(), list()
                for time in range(original.shape[-1]):
                    p_corrs_aux, s_corrs_aux, f_l_corrs_aux = leakage_(original[:, :, :, time], scrambled[:, :, :, time], axis=i)
                    p_corrs.append(p_corrs_aux)
                    s_corrs.append(s_corrs_aux)
                    f_l_corrs.append(f_l_corrs_aux)
                    
                p_corrs = np.mean(np.stack(p_corrs, axis=0), axis=0)
                s_corrs = np.mean(np.stack(s_corrs, axis=0), axis=0)
                f_l_corrs = np.mean(np.stack(f_l_corrs, axis=0), axis=0)
                logger.info(f" >>>\ \ \ Spatial Leakage Analysis on 'func', Dimension: {axis} / / /<<< ")
            if not Dim4:
                replaced = list()
                random_replace = False
                if random_replace:
                    range_ = 2
                    if axis == "x":
                        for _ in range(range_):
                            idx = random.randint(10, list(original.shape)[i] - 10)
                            scrambled[idx, :, :] = original[idx, :, :]
                            replaced.append(idx)
                    elif axis == "y":
                        for _ in range(range_):
                            idx = random.randint(10, list(original.shape)[i] - 10)
                            scrambled[:, idx, :] = original[:, idx, :]
                            replaced.append(idx)
                    elif axis == "z":
                        for _ in range(range_):
                            idx = random.randint(10, list(original.shape)[i] - 10)
                            scrambled[:, :, idx] = original[:, :, idx]
                            replaced.append(idx)
                if random_replace:
                    print(f"\t - Replaced indices in {axis}: {replaced}")

                p_corrs, s_corrs, f_l_corrs = leakage_(original, scrambled, axis=i)
                logger.info(f"Spatial Leakage Analysis on 'anat', Dimension: {axis} checked")
            pfl, avg_p_l, min_p_l, max_p_l = summary(p_corrs, type_="nii")
            sfl, avg_s_l, min_s_l, max_s_l = summary(s_corrs, type_="nii")
            ffl, avg_f_l, min_f_l, max_f_l = summary(f_l_corrs, type_="nii")
            unique, counts = np.unique(f_l_corrs, return_counts=True)
            unique = [x.item() if not np.isnan(x) else float('nan') for x in unique]
            counts = counts.tolist()
            log_msg = ("\n"+
                "\t" * 9 + "#" * 9 + "\n" +
                "\t" * 9 +"#"+"\t" + axis.upper() + "\t"+"#"+"\n" +
                "\t" * 9 +"#" * 9)
            result = results[axis] #; logger.info(log_msg)  
            result["f_corr_fl"].append(ffl) ; logger.info(f"Dimension[{axis.upper()}] np.allclose Value Count: {dict(zip(unique, counts))}")
            result["f_corr_pl_avg"].append(avg_f_l) 
            result["f_corr_pl_min"].append(min_f_l) 
            result["f_corr_pl_max"].append(max_f_l)
       
            result["p_corr_fl"].append(pfl) ; logger.info(f"Full leakage, Pearson Correlation: {pfl}")
            result["p_corr_pl_avg"].append(avg_p_l) 
            result["p_corr_pl_min"].append(min_p_l) 
            result["p_corr_pl_max"].append(max_p_l) 

            result["s_corr_fl"].append(sfl) ; logger.info(f"Full leakage, SSIM score: {sfl}")
            result["s_corr_pl_avg"].append(avg_s_l) 
            result["s_corr_pl_min"].append(min_s_l) 
            result["s_corr_pl_max"].append(max_s_l) 
            left = np.argwhere(f_l_corrs == 1.0).shape[0]#np.concatenate([arr[arr >= 0.99999] for arr in np.nanmax(p_corrs,axis=1)]).shape[0]
            right = f_l_corrs.shape[0]# - np.isnan(np.nanmax(p_corrs,axis=1)).sum()
            print(f"\t - Dimension[{axis.upper()}]: \tFull Leakage: {left}/{right} slices\tAverage Partial Leakage: {'Null' if np.nanmean(np.nanmax(p_corrs, axis=1)) == 1.0 else round(np.nanmean(np.nanmax(p_corrs, axis=1)),4)}") # \tIdentical: {round(ffl, 2)}%
            logger.info(f"Dimension[{axis.upper()}]: \tFull Leakage: {left}/{right} slices\tAverage Partial Leakage: {'Null' if np.nanmean(np.nanmax(p_corrs, axis=1)) == 1.0 else round(np.nanmean(np.nanmax(p_corrs, axis=1)),4)}")
            viz_report(p_corrs, s_corrs, f_l_corrs,loop=i, file_shape=original.shape)
            
    #########################################
    #             TEMPORAL                  #
    #########################################
        if Dim4:
            print(f" - Temporal voxel-wise Analysis")
            logger.info(f" >>>\ \ \ Temporal voxel-wise Analysis / / /<<<")
            o_p = cubeT(original, cube_size=1, stride=1)
            o_s = cubeT(scrambled, cube_size=1, stride=1)
            print(f"\t - Total voxels: {len(o_p)} of shape {o_p[0].shape}")
            logger.info(f"Total voxels: {len(o_p)} of shape {o_p[0].shape}")
            
            p_corrs, s_corrs,f_l_corrs = leakage_2D(np.array(o_p),np.array(o_s))
            #print(np.unique(f_l_corrs, return_counts=True))
            #full_leakage = True if (np.argwhere(p_corrs >= .99999).shape[0])/(p_corrs.shape[0]) > 0 else False
            partial_leakage = 'Null' if np.round(np.nanmean(p_corrs),4) == 1.0 else np.round(np.nanmean(p_corrs),4)
            full_leakage = True if np.argwhere(f_l_corrs == 1.0).shape[0] != 0 else False 
            print(f"\t - Temporal: \tFull Leakage: {np.argwhere(f_l_corrs == 1.0).shape[0]}/{f_l_corrs.shape[0]} voxels \tAverage Partial Leakage {partial_leakage}")
            dim4result = {"fl":full_leakage, "pl":partial_leakage}
            logger.info(f"Temporal: \tFull Leakage: {np.argwhere(p_corrs >= .99999).shape[0]}/{p_corrs.shape[0]} voxels \tAverage Partial Leakage {partial_leakage}")
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
        
        full_leakage = True if np.argwhere(f_l_corrs == 1.0).shape[0] != 0 else False #True if np.nanmax(p_corrs) >= 0.99999 else False
        partial_leakage = 'Null' if np.round(np.nanmean(p_corrs),4) == 1.0 else np.round(np.nanmean(p_corrs),4) 
        #identical = (np.argwhere(f_l_corrs >= .99999).shape[0] / f_l_corrs.shape[0]) * 100
        print(f"\t - Temporal: \tFull Leakage: {np.argwhere(p_corrs >= 0.99999).shape[0]}/{p_corrs.shape[0]} channels \tAverage Partial Leakage {partial_leakage}") # \tIdentical: {identical}%
        #results = {"fl": full_leakage, "pl": partial_leakage}
        
        viz_spatiotemporal(p_corrs, s_corrs, f_l_corrs, file_shape=original.shape)
        
    else: print(f" - Unsupported file type.")

    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        if Dim4: 
            fl, pl = leak_detect(results, "func", dim4result=dim4result)
        if not Dim4:
            fl, pl = leak_detect(results, "anat")
    elif ext.endswith(".fif") or ext.endswith(".fif.gz") or ext.endswith(".vhdr") or ext.endswith(".vhdr.gz"):
        fl, pl = leak_detect(results, "vhdr")
    if Dim4:
        log_msg = ("\n"+
                   "\n" + "\t" * 2 + "#" *41 + "\n" +
                   "\t" * 2 + "#"+"  " + " |"+"    AVG    " + "|" +"    MIN    " + "|"+"    MAX    " + "#"+"\n" +
                   "\t" * 2 + "#"+ "_" * 39 + "#"+"\n" +
                   "\t" * 2 + "#"+" X " + "|" + str(round(results["x"]["p_corr_pl_avg"][0], 9)) + "|" + str(round(results["x"]["p_corr_pl_min"][0], 9)) + "|" +  str(round(results["x"]["p_corr_pl_max"][0], 9)) + "#"+"\n" +
                   "\t" * 2 + "#"+ "_" * 39 + "#"+"\n" +
                   "\t" * 2 + "#"+" Y " + "|" + str(round(results["y"]["p_corr_pl_avg"][0], 9)) + "|" + str(round(results["y"]["p_corr_pl_min"][0], 9)) + "|" +  str(round(results["y"]["p_corr_pl_max"][0], 9)) + "#"+"\n" +
                   "\t" * 2 + "#"+ "_" * 39 + "#"+"\n" +
                   "\t" * 2 + "#"+" Z " + "|" + str(round(results["z"]["p_corr_pl_avg"][0], 9)) + "|" + str(round(results["z"]["p_corr_pl_min"][0], 9)) + "|" +  str(round(results["z"]["p_corr_pl_max"][0], 9)) + "#"+"\n" +
                   "\t" * 2 + "#" * 41 +
                   "\n")
        logger.info(log_msg)     
    
    #########################################
    #             REPORT                    #
    #########################################
    
    if r:
        
        report_output = subject_name +"_"+ task +"_"+ run_ if task else subject_name
        spatiotemporal = True if Dim4 else False
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
            partial_leakage=round(pl,4) * 100,
            
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
           spatiotemporal=spatiotemporal, spatiotemporal_image_path="img/correlations dist along Time.png",          
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
            partial_leakage=round(pl,4) * 100,
            
           p_leakage_time_min=np.mean(results["time"]["p_corr_pl_min"]),
           p_leakage_time_max=np.mean(results["time"]["p_corr_pl_max"]),
           p_leakage_time_avg=np.mean(results["time"]["p_corr_pl_avg"]), 
            
           s_leakage_time_min=np.mean(results["time"]["s_corr_pl_min"]),
           s_leakage_time_max=np.mean(results["time"]["s_corr_pl_max"]),
           s_leakage_time_avg=np.mean(results["time"]["s_corr_pl_avg"]), 
             
           full_leakage=fl)    
     
    return np.round(pl,4),fl

def pair(o, s, data_type):
    if o.split("/")[-1] != s.split("/")[-1]:
    
        logger.info(f"Original and Scrambled filename did not match: \n\tOriginal: {o.split('/')[-1]}\tScrambled: {s.split('/')[-1]}")
        print(f" - Warning: Original and Scrambled filename does not match.\n\t - Original: {o.split('/')[-1]}\t - Scrambled: {s.split('/')[-1]}")
        print(f" - Mentioned subject will be skipped.")
        return

    
    
    file_ = neuro_reader(o).get_data() if "fif" in o or "vhdr" in o else nii_reader(o).get_data()
    if len(list(file_.shape)) == 3:
        data_type = "anat"
    subject_name, task, run_ = parse_info(o, data_type)
    
    log_msg = ( "\n"+
                "\n" + "\t" * 2 + "#" * 37 + "\n" +
                "\t" * 2 + "#"+"\t" + subject_name+"-"+task+"-"+run_+ "\t"+"#"+"\n" +
                "\t" * 2 + "#" * 37 +
                "\n") if data_type == "func" else ( "\n"+
                "\n" + "\t" * 2 + "#" * 13 + "\n" +
                "\t" * 2 + "#"+"\t" + subject_name + "\t"+"#"+"\n" +
                "\t" * 2 + "#" * 13 +
                "\n")
    
    if run_: logger.info(log_msg)
    if data_type == "anat": logger.info(log_msg)
    logger.info(f"Comparing files:\nOriginal: {o}\nScrambled: {s}")
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
        "r": sys.argv[3] if len(sys.argv) > 3 else False,
    }
    if task: args["task"] = task
    if run_: args["run_"] = run_

    pl, fl = main(**args)
    test(pl,fl)
    if pl == 1.0:
        pl = 'Null'
    print(f" - Partial Leakage: {pl}")
    print(f" - Full Leakage: {fl}")
    if fl > 0.0:
        print(" - Please consider applying scramble on your dataset again.")
        logger.info(f"Leakage Detected.")

def test(pl,fl):
    fl = 1 if fl else 0
    with open(os.path.dirname(os.path.abspath(__file__))+"/test/result.tsv", "a") as f:
        try:
           line = "\t".join(map(str, [pl,fl])) + "\n"
           f.write(line)
        except Exception as e:
            print(f"Error writing data to file {filename}: {str(e)}", file=sys.stderr) 
def process_(original_list, scrambled_list, data_type):
    for o, s in tqdm.tqdm(zip(original_list, scrambled_list), total=len(original_list)):
        try:
            pair(o, s, data_type)  
        except Exception as e:
            traceback.print_exc()
            pass
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

    #########################################
    #             SINGLE TEST               #
    #########################################
    from subprocess import call
    import wget, shutil
    usecase = None
    
    if usecase == 2.5:
        link_original = "https://s3.amazonaws.com/openneuro.org/ds004934/sub-SAXNES2s001/func/sub-SAXNES2s001_task-DOTS_run-001_bold.nii.gz?versionId=R0fwRS9fxw8CcPZnb4zYsw9I5v19aAbP"
        wget.download(link_original)
        os.makedirs("input_TEST/func/original", exist_ok=True)
        file_ = [os.path.join(root, file) for root, _, files in os.walk(os.getcwd()) for file in files if file.endswith(".gz")]
        shutil.copy2(file_[0], "input_TEST/func/original")
        os.remove(file_[0])
        print(f"\nOriginal file downloaded")
        original = "input_TEST/func/original/"
        scrambled = "input_TEST/func/scrambled/"
    elif usecase == 2.2:
        link_original = "https://s3.amazonaws.com/openneuro.org/ds003826/sub-02/anat/sub-02_T1w.nii.gz?versionId=2zRivZaztVdjsig2hWgHLJCj5542CdeK"
        wget.download(link_original)
        os.makedirs("input_TEST/anat/original", exist_ok=True)
        file_ = [os.path.join(root, file) for root, _, files in os.walk(os.getcwd()) for file in files if file.endswith(".gz")]
        shutil.copy2(file_[0], "input_TEST/anat/original")
        os.remove(file_[0])
        print(f"\nOriginal file downloaded")
        original = "input_TEST/anat/original/"
        scrambled = "input_TEST/anat/scrambled/"
    if usecase != None:
        s_args = ["scramble", original, scrambled, "nii", "permute", "t", "-i"]
        print(f"Scramble method: \n{' '.join(s_args)}")
        call(s_args) 													# "permute", "t", "-i"
    												 				# "wobble", "-a", "4"
    																# "scatter", "4" 
        orig = original
        scra = scrambled
        print(f"File is scrambled")
    
    orig = sys.argv[1]
    scra = sys.argv[2]             
             
    if detect_file(orig) == "nii":
        process_(fetch_files(orig).nii_(), fetch_files(scra).nii_(), data_type="func")
    if detect_file(orig) == "fif":
        process_(fetch_files(orig).fif_(), fetch_files(scra).fif_(), data_type="fif")
    if detect_file(orig) == "vhdr":
        process_(fetch_files(orig).vhdr_(), fetch_files(scra).vhdr_(), data_type="vhdr")

        
    total_time = time.time() - start_time
    print(f"\nTotal time taken: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    logger.info(f"Total time taken: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")	
