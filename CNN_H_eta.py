#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:26:16 2020

Find the H and eta associated with a given volatility path using CNN method from

@author: 
"""

#%%
import os
os.chdir("/home/jenni/Documents/SHWPF/stats/fMRI_vol/phantom/")

import keras
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.special as special

#%%
def rBergomi_path_chol(grid_points, M, H, T, eta):
    # this function genrates M trajectories of the process Z in the rBergomi model.
    # source: https://github.com/amuguruza/RoughFCLT/blob/master/rDonsker.ipynb
    """
    @grid_points: # points in the simulation grid
    @H: Hurst Index
    @T: time horizon
    @M: # paths to simulate
    """
    
    assert 0<H<1.0
        
    ## Step1: create partition 
    
    X=np.linspace(0, 1, num=grid_points)
    
    # get rid of starting point
    X=X[1:grid_points]
    
    ## Step 2: compute covariance matrix
    Sigma=np.zeros((grid_points-1,grid_points-1))
    for j in range(grid_points-1):
        for i in range(grid_points-1):
            if i==j:
                Sigma[i,j]=np.power(eta,2)*(np.power(X[i],2*H))/2/H
            else:
                s=np.minimum(X[i],X[j])
                t=np.maximum(X[i],X[j])
                Sigma[i,j]=np.power(eta,2)*(np.power(t-s,H-0.5))/(H+0.5)*np.power(s,0.5+H)*special.hyp2f1(0.5-H, 0.5+H, 1.5+H, -s/(t-s))
        
    ## Step 3: compute Cholesky decomposition
    
    P=np.linalg.cholesky(Sigma)
    
    ## Step 4: draw Gaussian rv
    np.random.seed(0)
    Z=np.random.normal(loc=0.0, scale=1., size=[M,grid_points-1]) # 
    
    ## Step 5: get V
    
    W=np.zeros((M,grid_points))
    for i in range(M):
        W[i,1:grid_points]=np.dot(P,Z[i,:])
        
    #Use self-similarity to extend to [0,T] 
    
    return W*np.power(T,H)

def rBergomi_path_cholNS(grid_points, M, H, T, eta):
    # this function genrates M trajectories of the process Z in the rBergomi model.
    # source: https://github.com/amuguruza/RoughFCLT/blob/master/rDonsker.ipynb
    """
    @grid_points: # points in the simulation grid
    @H: Hurst Index
    @T: time horizon
    @M: # paths to simulate
    """
    
    #assert 0<H<1.0
        
    ## Step1: create partition 
    
    X=np.linspace(0, 1, num=grid_points)
    
    # get rid of starting point
    X=X[1:grid_points]
    
    ## Step 2: compute covariance matrix
    Sigma=np.zeros((grid_points-1,grid_points-1))
    for j in range(grid_points-1):
        for i in range(grid_points-1):
            if i==j:
                Sigma[i,j]=np.power(eta,2)*(np.power(X[i],2*H))/2/H
            else:
                s=np.minimum(X[i],X[j])
                t=np.maximum(X[i],X[j])
                Sigma[i,j]=np.power(eta,2)*(np.power(t-s,H-0.5))/(H+0.5)*np.power(s,0.5+H)*special.hyp2f1(0.5-H, 0.5+H, 1.5+H, -s/(t-s))
        
    ## Step 3: compute Cholesky decomposition
    
    P=np.linalg.cholesky(Sigma)
    
    ## Step 4: draw Gaussian rv
    np.random.seed()
    Z=np.random.normal(loc=0.0, scale=1., size=[M,grid_points-1]) # 
    
    ## Step 5: get V
    
    W=np.zeros((M,grid_points))
    for i in range(M):
        W[i,1:grid_points]=np.dot(P,Z[i,:])
        
    #Use self-similarity to extend to [0,T] 
    
    return W*np.power(T,H)


#%% Generate paths with seed

numPath = 25000 # set to same as total current size of rBergomi data 
numStep = 200 # length of input vector, change as required
Hs1=np.random.uniform(0.0,1., numPath)
eta1=np.random.uniform(0.0,3.0, numPath)
Data1=pd.DataFrame()

for i in range(0,numPath):
    H=Hs1[i]
    eta=eta1[i]
    path=rBergomi_path_chol(numStep, 1, H , 1, eta)
    dF=pd.DataFrame(path)
    dF['H']=H
    dF['eta']=eta
    Data1=pd.concat([Data1,dF])
    
Data1=Data1.sample(frac=1) #this randomly re-orders robustData
DataVal1=Data1.values

xValues1=DataVal1[:,:200] # inputs to CNN (split up into training/test/validation sets)
hValues1=DataVal1[:,-2] # outputs from CNN ie H values.
etaValues1=DataVal1[:,-1]

# save series and Hs
np.save('paths3.npy',xValues1)
np.save('H3.npy',hValues1)
np.save('eta3.npy',etaValues1)


#%% random, no seed

#numPath = 25000 # set to same as total current size of rBergomi data 
#numStep = 200 # length of input vector, change as required
Hs2=np.random.uniform(0.0,1., numPath)
eta2=np.random.uniform(0.0,3.0, numPath)
Data2=pd.DataFrame()

for i in range(0,numPath):
    H=Hs2[i]
    eta=eta2[i]
    path=rBergomi_path_cholNS(numStep, 1, H , 1, eta)
    dF=pd.DataFrame(path)
    dF['H']=H
    dF['eta']=eta
    Data2=pd.concat([Data2,dF])
    
Data2=Data2.sample(frac=1) #this randomly re-orders robustData
DataVal2=Data2.values

xValues2=DataVal2[:,:200] # inputs to CNN (split up into training/test/validation sets)
hValues2=DataVal2[:,-2] # outputs from CNN ie H values.
etaValues2=DataVal2[:,-1]

# save series and Hs
np.save('paths4.npy',xValues2)
np.save('H4.npy',hValues2)
np.save('eta4.npy',etaValues2)

#%% Upload training data (if already created)

paths = np.concatenate((np.load('/home/jenni/Documents/SHWPF/stats/fMRI_vol/phantom/paths3.npy'),np.load('/home/jenni/Documents/SHWPF/stats/fMRI_vol/phantom/paths4.npy')), axis=0)
Hs = np.concatenate((np.load('/home/jenni/Documents/SHWPF/stats/fMRI_vol/phantom/H3.npy'),np.load('/home/jenni/Documents/SHWPF/stats/fMRI_vol/phantom/H4.npy')), axis=0)
ETAs = np.concatenate((np.load('/home/jenni/Documents/SHWPF/stats/fMRI_vol/phantom/eta3.npy'),np.load('/home/jenni/Documents/SHWPF/stats/fMRI_vol/phantom/eta4.npy')), axis=0)

# normalise ETA to range(0,1) for CNN
ETAs2 = np.tanh(ETAs)

Params = np.transpose(np.vstack((Hs,ETAs2)))

#%%
from sklearn.model_selection import train_test_split
# split up the data
xTraining, xTest, yTraining, yTest = train_test_split(paths, Params, test_size=0.3)
xTrain, xValid, yTrain, yValid = train_test_split(xTraining, yTraining , test_size=0.2)
# reshape
xTrain=xTrain.reshape(xTrain.shape[0], xTrain.shape[1],1)
xValid=xValid.reshape(xValid.shape[0], xValid.shape[1],1)
xTest=xTest.reshape(xTest.shape[0], xTest.shape[1],1)
yTrain=yTrain.reshape(yTrain.shape[0], 2)
yValid=yValid.reshape(yValid.shape[0], 2)
yTest=yTest.reshape(yTest.shape[0], 2)


#%%
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report
import time

def trainCNN(xTrain, yTrain, xValid, yValid, xTest, yTest, nodes=[32,64,128,128], dropout=[0.25, 0.25, 0.25], alp=[0.01,0.01,0.01,0.01], batchSize=32, kSize=20, poolSize=3):
    cnn = Sequential()
    for ii in range(len(nodes)-1):
      if ii == 0:
        cnn.add(Conv1D(nodes[ii], kernel_size=kSize, activation='tanh', padding='same', input_shape=(xTrain.shape[1],1)))
      else:
        cnn.add(Conv1D(nodes[ii], kernel_size=kSize, activation='linear', padding='same'))
      cnn.add(LeakyReLU(alpha=alp[ii]))
      cnn.add(MaxPooling1D(pool_size=poolSize,padding='same'))
      cnn.add(Dropout(dropout[ii]))
    cnn.add(Flatten())
    cnn.add(Dense(nodes[-1], activation='linear'))
    cnn.add(LeakyReLU(alpha=alp[-1]))           
    cnn.add(Dropout(dropout[-1]))
    cnn.add(Dense(2, activation='sigmoid'))
    cnn.compile(loss=keras.losses.mean_squared_error, optimizer='Adam') 
    T = time.time()
    cnn.fit(xTrain,yTrain,batch_size=batchSize,epochs=30,validation_data=(xValid,yValid))
    print('training time: %f' %(time.time()-T))    
    T = time.time()
    testMSE = cnn.evaluate(xTest,yTest)
    print('test time: %f' %(time.time()-T))
    return cnn, testMSE


#%%
cnn, testMSE = trainCNN(xTrain, yTrain, xValid, yValid, xTest, yTest)
print(np.sqrt(testMSE))

yPred = cnn.predict(xTest)

#%%

print(np.sqrt(testMSE))

print(yPred[:5,0])

plt.scatter(yTest[:,0], yPred[:,0])
plt.ylabel('Predicted H')
plt.xlabel('True H')
plt.show()

#%%

print(yPred[1][:5])

plt.scatter(np.arctanh(yTest[:,1]), np.arctanh(yPred[:,1]))
plt.ylabel('Predicted eta')
plt.xlabel('True eta')
plt.show()


# save yPred
np.savetxt("hPred.csv",yPred[:,0],delimiter=",")
np.savetxt("etaPred.csv",np.arctanh(yPred[:,1]),delimiter=",")
np.savetxt("hTest.csv",yTest[:,0],delimiter=",")
np.savetxt("etaTest.csv",np.arctanh(yTest[:,1]),delimiter=",")

#%%
# OPTIMALLY COMBINED DATA

#%%
# Phantom data
#%%  phantom 1

volData=np.genfromtxt("Phantom1_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

#%%
final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("Phantom1_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("Phantom1_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write H nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/Phantom1_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'Phantom1_CNN_H.nii')

#%% phantom 2

volData=np.genfromtxt("Phantom2_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("Phantom2_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("Phantom2_CNN_ETA.csv", np.arctanh(final), delimiter=",") 
#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/Phantom2_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'Phantom2_CNN_H.nii')

#%%
#    DATASET ds000528

#%% sub-17821

volData=np.genfromtxt("sub-17821_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:200]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("sub-17821_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("sub-17821_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/sub-17821_vent_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'sub-17821_cnnH.nii')

#%% sub-21300

volData=np.genfromtxt("sub-21300_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:200]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("sub-21300_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("sub-21300_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/sub-21300_vent_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'sub-21300_cnnH.nii')

#%%
#   DATASET ds000210 

#%% sub-28

volData=np.genfromtxt("sub-28_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:200]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("sub-28_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("sub-28_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/sub-28_vent_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'sub-28_cnnH.nii')

#%% sub-30

volData=np.genfromtxt("sub-30_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:200]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("sub-30_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("sub-30_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/sub-30_vent_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'sub-30_cnnH.nii')

#%%

# COMBINED DATA (simple variance)
#%%
# Phantom data

#%% phantom 1

volData=np.genfromtxt("Phantom1_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("Phantom1_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("Phantom1_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write H nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/Phantom1_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'Phantom1_CNN_H.nii')

#%% phantom 2

volData=np.genfromtxt("Phantom2_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("Phantom2_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("Phantom2_CNN_ETA.csv", np.arctanh(final), delimiter=",") 
#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/Phantom2_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'Phantom2_CNN_H.nii')

#%%
#    DATASET ds000528

#%% sub-17821

volData=np.genfromtxt("sub-17821_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:200]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("sub-17821_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("sub-17821_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/sub-17821_vent_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'sub-17821_cnnH.nii')

#%% sub-21300

volData=np.genfromtxt("sub-21300_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:200]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("sub-21300_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("sub-21300_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/sub-21300_vent_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'sub-21300_cnnH.nii')

#%%
#   DATASET ds000210 

#%% sub-28

volData=np.genfromtxt("sub-28_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:200]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("sub-28_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("sub-28_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/sub-28_vent_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'sub-28_cnnH.nii')

#%% sub-30

volData=np.genfromtxt("sub-30_RV.csv", delimiter=",")
cnnH=[]
cnnETA=[]

dims = volData.shape
dim1 = dims[0]
dim2 = dims[1]

print(dim1)

for s in range(0, dim1):
    data=volData[s,:200]
    volSeries=np.log(data)
    volSeries=volSeries-np.mean(volSeries) # centre series on 0
    volSeries=volSeries.reshape(1,200,1)
    cnnPred=cnn.predict(volSeries)
    cnnH.append(cnnPred[0,0])
    cnnETA.append(cnnPred[0,1])
                        
print(np.mean(cnnH),np.max(cnnH),np.min(cnnH))
print(np.mean(np.arctanh(cnnETA)),np.max(np.arctanh(cnnETA)),np.min(np.arctanh(cnnETA)))

final=np.array(cnnH)
final=final.reshape(len(final),1)
np.savetxt("sub-30_CNN_H.csv", final, delimiter=",") 

final=np.array(cnnETA)
final=final.reshape(len(final),1)
np.savetxt("sub-30_CNN_ETA.csv", np.arctanh(final), delimiter=",") 

#%% write nifti

from nibabel.testing import data_path
import nibabel as nib

mask_nifti = os.path.join(data_path, '/path/to/mask/sub-30_vent_mask.nii')
mask = nib.load(mask_nifti)
mask.set_data_dtype(np.float64)
H_output = mask.get_fdata()

dims = H_output.shape
dim1 = dims[0]
dim2 = dims[1]
dim3 = dims[2]

# H vector index
ind = 0

# loop through each voxel and save H incorrect place
for i in range(0, dim1):
    for j in range(0, dim2):
        for k in range(0, dim3):
            
            # check voxel is inside mask
            if H_output[i,j,k] == 0:
                continue
            
            H_output[i,j,k] = cnnH[ind]
            ind += 1
            

#%% Save output images

H_img = nib.Nifti1Image(H_output.astype(np.float64), mask.affine, mask.header)
nib.save(H_img, 'sub-30_cnnH.nii')