import numpy as np
import glob
from tqdm import tqdm
from multiprocessing import Pool
import time as t
import os
from .file_size_filter import *
from .create_list_files_day import *
from .limb_darkening import remove_limb_darkening, correct_limb_darkening_and_background
from .fits_utils import open_fits_file, open_fits_fz_file, save_fits_file
import sys
from .filename_helpers import filter_science_files, updated_filename as build_updated_filename, updated_filepath

#################
# DAY OF INTEREST
#################
if len(sys.argv) < 5:
    raise SystemExit(
        "Usage: python3 -m solfilosc.data_processing.preprocess_data "
        "<year> <month> <day> <n_proc> [raw_data_dir]\n"
        "Default raw_data_dir: data/raw/<month>/<day>/"
    )

year = sys.argv[1]
month = sys.argv[2]
day = sys.argv[3]
n_proc = sys.argv[4]
raw_data_dir = sys.argv[5] if len(sys.argv) > 5 else os.path.join('data', 'raw', month, day)
print(f'Day of calculations: {day} \n')


################################################################
# DIRECTORY WHERE ALL .FITS.FZ FILES ARE STORED FOR A SINGLE DAY
################################################################
directory_of_original_data = raw_data_dir
if not directory_of_original_data.endswith(os.sep):
    directory_of_original_data += os.sep

if not os.path.isdir(directory_of_original_data):
    raise FileNotFoundError(
        f"Raw FITS directory not found: {directory_of_original_data}\n"
        "Create this folder and place the .fits.fz files there, or pass a custom "
        "raw-data directory as the optional fifth argument to preprocess_data."
    )

print(f'Data is in directory {directory_of_original_data} \n')
directory_of_processed_data = f'data/{day}/'
os.makedirs(directory_of_processed_data, exist_ok=True)


########################################################
# LIST OF ALL USEFUL .FTIS.FZ FILE PATHS FOR A GIVEN DAY
########################################################
print('Creating file list and applying filters')
files = sorted(glob.glob(directory_of_original_data+'*.fits.fz'))
files = filter_science_files(files)
if len(files) == 0:
    raise FileNotFoundError(
        f"No .fits.fz files found in {directory_of_original_data}\n"
        "Place the raw daily files there, or pass the correct raw-data directory "
        "as the optional fifth argument."
    )
files = filter1(files)
if len(files) == 0:
    raise RuntimeError(
        f"All .fits.fz files in {directory_of_original_data} were rejected by "
        "the file-size filter. Check that the raw files are complete."
    )
obs_windows_telescopes = observation_windows_telescopes(files)
obs_windows_times = observation_windows_times(obs_windows_telescopes)
obs_windows_times_telescopes = observation_windows_times_telescopes(obs_windows_times, files)
obs_windows_times_telescopes = add_sharpness_entry(obs_windows_times_telescopes)
obs_windows_times_telescopes = filter_density(obs_windows_times_telescopes)
obs_windows_times_telescopes = filter_sharpness(obs_windows_times_telescopes)
final_files_list = final_list_of_files(obs_windows_times_telescopes)
print(f'Total of {len(final_files_list)} files to be preprocessed \n')
print('Files filtered \n')



####################
# Data preprocessing
####################

def preprocess_data(args):
    file = args
    updated_filename = updated_filepath(file, directory_of_processed_data)
    header, data = open_fits_fz_file(file)
    data = data.astype(np.float32)
    data = correct_limb_darkening_and_background(data)
    #data = median_adjust_intensity(data, reference_data, inside_indices_list)
    # float32 is enough downstream (the data cube is float32) and halves disk I/O
    save_fits_file(updated_filename, header, data.astype(np.float32))
    return f"{build_updated_filename(file)} created"


print('Prepocessing data, removing limb darkening, adjusting intensity, etc ')
preprocessing_time_start = t.perf_counter()

with Pool(int(n_proc)) as preprocess_pool:
    results = list(tqdm(preprocess_pool.imap(preprocess_data, final_files_list), total=len(final_files_list)))

preprocessing_time_end = t.perf_counter()
prepocessing_time_elapsed = preprocessing_time_end - preprocessing_time_start
print(f'Preprocessing took in total: {prepocessing_time_elapsed/60:.1f} min \n')
