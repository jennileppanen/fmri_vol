#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:44:26 2020

Echoes combined suing simple average and variance

@author: 
"""

import os
os.chdir("/path/to/data/")

import numpy as np
import matplotlib.pyplot as plt
from nibabel.testing import data_path
import nibabel as nib
import nibabel.processing as nibp
from sklearn.linear_model import LinearRegression


# import nifti files
echo_1_file = os.path.join(data_path, '/path/to/data/Phantom2/rasub-Phantom2_task-rest_echo-1_bold.nii')
echo_2_file = os.path.join(data_path, '/path/to/data/Phantom2/rasub-Phantom2_task-rest_echo-2_bold.nii')
echo_3_file = os.path.join(data_path, '/path/to/data/Phantom2/rasub-Phantom2_task-rest_echo-3_bold.nii')
echo_4_file = os.path.join(data_path, '/path/to/data/Phantom2/rasub-Phantom2_task-rest_echo-4_bold.nii')
echo_5_file = os.path.join(data_path, '/path/to/data/Phantom2/rasub-Phantom2_task-rest_echo-5_bold.nii')
echo_6_file = os.path.join(data_path, '/path/to/data/Phantom2/rasub-Phantom2_task-rest_echo-6_bold.nii')
echo_7_file = os.path.join(data_path, '/path/to/data/Phantom2/rasub-Phantom2_task-rest_echo-7_bold.nii')
echo_8_file = os.path.join(data_path, '/path/to/data/Phantom2/rasub-Phantom2_task-rest_echo-8_bold.nii')

echo_1 = nib.load(echo_1_file)
echo_2 = nib.load(echo_2_file)
echo_3 = nib.load(echo_3_file)
echo_4 = nib.load(echo_4_file)
echo_5 = nib.load(echo_5_file)
echo_6 = nib.load(echo_6_file)
echo_7 = nib.load(echo_7_file)
echo_8 = nib.load(echo_8_file)

# read data
e1 = echo_1.get_fdata()
e2 = echo_2.get_fdata()
e3 = echo_3.get_fdata()
e4 = echo_4.get_fdata()
e5 = echo_5.get_fdata()
e6 = echo_6.get_fdata()
e7 = echo_7.get_fdata()
e8 = echo_8.get_fdata()

# prep realised vol and demeanded ts output arrays
RV = []
TS = []

# prep optimally combined ts and realised vol output images
echo_1.set_data_dtype(np.float64)
comb = echo_1.get_fdata()
vol = echo_1.get_fdata()

# read mask and prep output T2* image
mask_nifti = os.path.join(data_path, '/path/to/mask/Phantom2_mask.nii')
Mask = nib.load(mask_nifti)
Mask.set_data_dtype(np.float64)
mask = Mask.get_fdata()

# get dimensions to create nested loops
dims = e1.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]


# %%
# Combine echos using T2*-weights
     
# for saving demeaned ts
index = 0

# loop through each voxel and run optimal combination from Tedana
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # read ts
            e1_ts = e1[i,j,k,:]
            e2_ts = e2[i,j,k,:]
            e3_ts = e3[i,j,k,:]
            e4_ts = e4[i,j,k,:]
            e5_ts = e5[i,j,k,:]
            e6_ts = e6[i,j,k,:]
            e7_ts = e7[i,j,k,:]
            e8_ts = e8[i,j,k,:]
            
            # check voxel is inside mask and write output as zero if not inside mask
            if mask[i,j,k] == 0:
                comb[i,j,k,:] = 0
                vol[i,j,k,:] = 0
                continue
            
            
            e1_ts = e1_ts.reshape(len(e1_ts),1)
            e2_ts = e2_ts.reshape(len(e2_ts),1)
            e3_ts = e3_ts.reshape(len(e3_ts),1)
            e4_ts = e4_ts.reshape(len(e4_ts),1)
            e5_ts = e5_ts.reshape(len(e5_ts),1)
            e6_ts = e6_ts.reshape(len(e6_ts),1)
            e7_ts = e7_ts.reshape(len(e7_ts),1)
            e8_ts = e8_ts.reshape(len(e8_ts),1)
            
            data = np.concatenate((e1_ts,e2_ts,e3_ts,e4_ts,e5_ts,e6_ts,e7_ts,e8_ts),axis=1)
            
            combined = np.average(data, axis=1)
            comb[i,j,k,:] = combined
            
            
            # calculate realised vol
            # remove linear trend (by model fitting)
            X = [p for p in range(0, len(e1_ts))]
            X = np.reshape(X, (len(X), 1))
            
            model = LinearRegression()
            model.fit(X, e1_ts)
            trend = model.predict(X)
            e1_dts = [e1_ts[p] - trend[p] for p in range(0,len(e1_ts))]
            e1_dts = np.asarray(e1_dts)
            
            X = [p for p in range(0, len(e1_ts))]
            X = np.reshape(X, (len(X), 1))
            model.fit(X, e2_ts)
            trend = model.predict(X)
            e2_dts = [e2_ts[p] - trend[p] for p in range(0,len(e2_ts))]
            e2_dts = np.asarray(e2_dts)
            
            X = [p for p in range(0, len(e1_ts))]
            X = np.reshape(X, (len(X), 1))
            model.fit(X, e3_ts)
            trend = model.predict(X)
            e3_dts = [e3_ts[p] - trend[p] for p in range(0,len(e3_ts))]
            e3_dts = np.asarray(e3_dts)
            
            X = [p for p in range(0, len(e1_ts))]
            X = np.reshape(X, (len(X), 1))
            model.fit(X, e4_ts)
            trend = model.predict(X)
            e4_dts = [e4_ts[p] - trend[p] for p in range(0,len(e4_ts))]
            e4_dts = np.asarray(e4_dts)
            
            X = [p for p in range(0, len(e1_ts))]
            X = np.reshape(X, (len(X), 1))
            model.fit(X, e5_ts)
            trend = model.predict(X)
            e5_dts = [e5_ts[p] - trend[p] for p in range(0,len(e5_ts))]
            e5_dts = np.asarray(e5_dts)
            
            X = [p for p in range(0, len(e1_ts))]
            X = np.reshape(X, (len(X), 1))
            model.fit(X, e6_ts)
            trend = model.predict(X)
            e6_dts = [e6_ts[p] - trend[p] for p in range(0,len(e6_ts))]
            e6_dts = np.asarray(e6_dts)
            
            X = [p for p in range(0, len(e1_ts))]
            X = np.reshape(X, (len(X), 1))
            model.fit(X, e7_ts)
            trend = model.predict(X)
            e7_dts = [e7_ts[p] - trend[p] for p in range(0,len(e7_ts))]
            e7_dts = np.asarray(e7_dts)
            
            X = [p for p in range(0, len(e1_ts))]
            X = np.reshape(X, (len(X), 1))
            model.fit(X, e8_ts)
            trend = model.predict(X)
            e8_dts = [e8_ts[p] - trend[p] for p in range(0,len(e8_ts))]
            e8_dts = np.asarray(e8_dts)
            
            e1_dts = e1_dts.reshape(len(e1_dts),1)
            e2_dts = e2_dts.reshape(len(e2_dts),1)
            e3_dts = e3_dts.reshape(len(e3_dts),1)
            e4_dts = e4_dts.reshape(len(e4_dts),1)
            e5_dts = e5_dts.reshape(len(e5_dts),1)
            e6_dts = e6_dts.reshape(len(e6_dts),1)
            e7_dts = e7_dts.reshape(len(e7_dts),1)
            e8_dts = e8_dts.reshape(len(e8_dts),1)
            
            d_data = np.concatenate((e1_dts, e2_dts, e3_dts, e4_dts, e5_dts, e6_dts, e7_dts, e8_dts),axis=1)
            
            dmean_comb = np.average(d_data, axis=1)
            TS.append(dmean_comb)
            index += 1
            dmean_comb = dmean_comb.reshape(len(dmean_comb),1)
            
            rv = np.average((d_data - dmean_comb)**2, axis=1)
            vol[i,j,k,:] = rv
            RV.append(rv)

#%%
# Write output images

comb_img = nib.Nifti1Image(comb, echo_1.affine, echo_1.header)
nib.save(comb_img, 'Phantom2_comb.nii')    

vol_img = nib.Nifti1Image(vol.astype(np.float64), echo_1.affine, echo_1.header)
nib.save(vol_img, 'Phantom2_rv.nii')           

np.savetxt("Phantom2_RV.csv", RV, delimiter=",")     
np.savetxt("Phantom2_demean_TS.csv", TS, delimiter=",")        