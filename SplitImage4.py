#!/usr/bin/env python
# coding: utf-8

# In[146]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

import timeit

def chunkify4(img,figures):
    n_chunks = 4
    shape = img.shape
    
    #Get len/chunk
    x_len_per_chunk = int(np.floor(shape[0]/(n_chunks/2)))
    y_len_per_chunk = int(np.floor(shape[1]/(n_chunks/2)))
    
    #Compute Offset from (0,0) to center the chunk
    x_offset = int(np.floor((shape[0]%x_len_per_chunk)/2))
    y_offset = int(np.floor((shape[1]%y_len_per_chunk)/2))
    
    #Compute Indices for all points
    x_indices = [i + x_offset for i in range(0, shape[0]+1, x_len_per_chunk)] 
    y_indices = [i + y_offset for i in range(0, shape[1]+1, y_len_per_chunk)] 

    chunks = np.zeros((4,x_len_per_chunk,y_len_per_chunk))
    
    count = 0
    
    for i in range(0,len(x_indices)-1):
        for j in range(0,len(y_indices)-1):
            count += 1
            chunks[count-1] = img[x_indices[i]:x_indices[i+1],y_indices[j]:y_indices[j+1]]
    
    if(figures):
        fig, axs = plt.subplots(2,2)
        fig.suptitle('Split Image')
        axs[0,0].imshow(chunks[0], cmap='gray', vmin=0, vmax=255)
        axs[0,1].imshow(chunks[1], cmap='gray', vmin=0, vmax=255)
        axs[1,0].imshow(chunks[2], cmap='gray', vmin=0, vmax=255)
        axs[1,1].imshow(chunks[3], cmap='gray', vmin=0, vmax=255)

        plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    
    return chunks

if __name__ == '__main__':
    im1 = 'images/original_04.png'

    # # load the two input images
    imageA = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)

    split_img = chunkify4(imageA,True)
