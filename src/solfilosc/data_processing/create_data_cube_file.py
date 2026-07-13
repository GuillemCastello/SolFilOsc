import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
import numpy as np
import glob
import shutil
import sys
from multiprocessing import Pool
from tqdm import tqdm
import time as t
import h5py
from .derotate import derotate
from .create_times_array import create_tdeltas_array
from .filename_helpers import observatory_change_indices
from ..paths import day_dir



##############################################################
#CREATING THE DATA CUBE / DEROTATING IMAGES OF THE TIME SERIES
##############################################################
print('Derotating images')
print('Using updated files')
year = sys.argv[1]
month = sys.argv[2]
day = sys.argv[3]
n_threads = sys.argv[4]

directory_of_data = day_dir('data', day) + os.sep
files_updated = sorted(glob.glob(directory_of_data+'*.fits'))

drot_time_start = t.perf_counter()
# Function to fill a single slice from the data cube, (image derotated)


print(f'Parallelizing the derotation process with {n_threads} threads, if this makes the computer crash lower n_threads number')

print('Creating data cube')
dim_data_cube = 2048
data_cube = np.zeros(shape=(len(files_updated), dim_data_cube, dim_data_cube), dtype=np.float32)
print(f'Data cube has a shape of {np.shape(data_cube)}')

# Fill the cube as results stream in instead of buffering every derotated
# frame in a list first (halves the peak memory of this stage).
with Pool(int(n_threads)) as p:
    for i, drot_slice in enumerate(tqdm(p.imap(derotate, files_updated), total=len(files_updated))):
        data_cube[i] = drot_slice

drot_time_end = t.perf_counter()
drot_time_elapsed = drot_time_end - drot_time_start
print(f'Time elapsed for derotating and creating the data_cube: {(drot_time_elapsed/60):.1f} min \n')

# The cube stays in memory and is post-processed below, so we avoid writing a
# large intermediate .h5 only to immediately read it back.

##############################################################
#POST PROCESSING THE DATA CUBE
##############################################################


##############################################################
#DELETING BAD IMAGES
##############################################################
print(f'Deleting images that even with this fixed still are bad using correlations between vertical/horizontal - diag1/diag2 pixels')
vertical = data_cube[:, 1024, :]
horizontal = data_cube[:, :, 1024]
diagonal1 = np.diagonal(data_cube, axis1=1, axis2=2)
diagonal2 = np.diagonal(np.fliplr(data_cube), axis1=1, axis2=2)

print('Computing correlations')
vh_corr = [np.corrcoef(vertical[i,:], horizontal[i,:]) for i in range(vertical.shape[0])]
vh_corr = np.array(vh_corr)
vh_corr = vh_corr[:, 0, 1]
d1d2_corr = [np.corrcoef(diagonal1[i,:], diagonal2[i,:]) for i in range(diagonal1.shape[0])]
d1d2_corr = np.array(d1d2_corr)
d1d2_corr = d1d2_corr[:, 0, 1]

print('If corr coef is below 2 std of the mean of the corr coef, the image is bad')
mask_vh =  vh_corr < (np.mean(vh_corr) - 2*np.std(vh_corr))
mask_d1d2 = d1d2_corr < (np.mean(d1d2_corr) - 2*np.std(d1d2_corr))
mask = mask_d1d2 + mask_vh
delete = np.where(mask==True)[0]

print(f'Deleting total of {len(delete)} images that are bad')
#Delete the slices of the data cube that are bad
data_cube = np.delete(data_cube, delete, axis=0)

# Drop the bad frames from the in-memory file list so the surviving files (sorted,
# aligned with the trimmed cube) drive telescope-change detection and the tdeltas array.
delete_set = set(int(i) for i in delete)
files_updated = [f for i, f in enumerate(files_updated) if i not in delete_set]

print('Files deleted')
print('')


##############################################################
#MEDIAN ADJUSTMENT OF PIXELS AT TELESCOPE CHANGE INSTANTS
##############################################################
print('Computing telescope change instants')
change_telescope_indices = observatory_change_indices(files_updated)

print(f'Median adjustment of each pixel at telescope change instants, total of {len(change_telescope_indices)} adjustments will be done')
space_median_adj = 15
n_frames = data_cube.shape[0]

for id, change in enumerate(change_telescope_indices):
    previous_segment_start = change_telescope_indices[id - 1] if id > 0 else 0
    posterior_segment_end = (
        change_telescope_indices[id + 1]
        if id < len(change_telescope_indices) - 1
        else n_frames
    )

    previous_start = max(previous_segment_start, change - space_median_adj)
    previous_end = change

    posterior_start = change
    posterior_end = min(posterior_segment_end, change + space_median_adj)

    previous = np.median(data_cube[previous_start:previous_end, :, :], axis=0)
    posterior = np.median(data_cube[posterior_start:posterior_end, :, :], axis=0)

    delta = posterior - previous

    if id == len(change_telescope_indices) - 1:
        data_cube[change:, :, :] = data_cube[change:, :, :] - delta
    else:
        next_change = change_telescope_indices[id + 1]
        data_cube[change:next_change, :, :] = data_cube[change:next_change, :, :] - delta

print('Median adjustment finished')
print('')


##############################################################
#OUTSIDE DISK PIXELS SET TO ZERO
##############################################################
print('Setting pixels oputside the disk to zero')
yy, xx = np.ogrid[:2048, :2048]
outside_disk = (yy - 1024) ** 2 + (xx - 1024) ** 2 > 830 ** 2
data_cube[:, outside_disk] = 0

print('Pixels outside the disk set to zero')
print('')


##############################################################
#SAVING THE FINAL DATA CUBE
##############################################################
print('Creating the tdeltas array')
tdeltas = np.array(create_tdeltas_array(files_updated), dtype=np.int32)

data_cube_file_out = f'{directory_of_data}{day}.h5'
print(f'Saving the final data cube in {data_cube_file_out}')
saving_h5_time_start = t.perf_counter()
with h5py.File(data_cube_file_out, mode='w') as f:
    f.create_dataset('time_series', data=data_cube, dtype=np.float32,
                    compression='gzip', compression_opts=4, shuffle=True)
    f.create_dataset('tdeltas', data=tdeltas, dtype=np.int32, 
                     compression='gzip', compression_opts=4, shuffle=True)
saving_h5_time_end = t.perf_counter()
print(f'h5 data file saved in {((saving_h5_time_end - saving_h5_time_start)/60):.1f} min \n')


##############################################################
#REMOVING INTERMEDIATE FILES (final cube is the only artifact we keep)
##############################################################
print(f'Removing intermediate files, keeping only {data_cube_file_out}')
for file in glob.glob(directory_of_data + '*_updated.fits'):
    os.remove(file)
shutil.rmtree(directory_of_data + 'delete', ignore_errors=True)  # stale quarantine from old runs
print('Intermediate files removed')
print('')
print('Data file creation finished')
