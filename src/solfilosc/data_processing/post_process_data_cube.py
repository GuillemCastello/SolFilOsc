import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import sys
from tqdm import tqdm
import itertools
import h5py
from scipy.ndimage import median_filter
from .create_times_array import create_tdeltas_array
from .filename_helpers import observatory_change_indices

##############################################################
#CREATING THE DATA CUBE / DEROTATING IMAGES OF THE TIME SERIES
##############################################################

print('Using updated files')
year = sys.argv[1]
month = sys.argv[2]
day = sys.argv[3]

directory_of_data = f'data/{day}/updated/'
files_updated = sorted(glob.glob(directory_of_data+'*.fits'))

data_cube_file = f'{directory_of_data}{day}_data.h5'
data_cube_file_out = f'{directory_of_data}{day}_data_modified.h5'

print('Loading data cube')
with h5py.File(data_cube_file, mode='r') as f:
    data_cube = np.array(f['time_series'], dtype=np.float32)

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

# Delete the files of the deleted times
deleted_files = [files_updated[i] for i in delete]

if 'delete' not in os.listdir(directory_of_data):
    os.makedirs(directory_of_data+'delete')

for file in deleted_files:
    os.replace(file, f'{directory_of_data}delete/{os.path.basename(file)}')

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
xrange = range(2048)
yrange = range(2048)
indices_list = list(itertools.product(xrange, yrange))
inside_indices_list = list(filter(lambda x: np.sqrt((x[0]-1024)**2 + (x[1]-1024)**2) > 830, indices_list))

for inside in tqdm(inside_indices_list):
    i = inside[0]
    j = inside[1]
    data_cube[:, i, j] = 0

print('Pixels outside the disk set to zero')
print('')

print('Creating the modified tdeltas array')
tdeltas = np.array(create_tdeltas_array(files_updated), dtype=np.int32)

print('Saving the modified data cube')
with h5py.File(data_cube_file_out, mode='w') as f:
    f.create_dataset('time_series', data=data_cube, dtype=np.float32, compression='gzip')
    f.create_dataset('tdeltas', data=tdeltas, dtype=np.int32, compression='gzip')
print('')
