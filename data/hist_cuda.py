#!/usr/bin/env python

"""
Basic 2d histogram.
"""

#import PYCUDA modules and libraries

import sys
from PIL import Image
import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pycuda.autoinit



# Function to load data, size could be 1, 10, 50, 100, 500
def load_data(size):
    R=size*1000
    C=1000
    img = np.memmap('/opt/data/random.dat', dtype=np.uint8, mode='r',shape=(R,C))
    return img, R, C

# Compute histogram in Python:
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins


kernel_code_template = """
__global__ void given_Hist(unsigned char *img, unsigned int *bins, int P)
{
                uint bx = blockIdx.x;
                uint tx = threadIdx.x;
                uint i = blockDim.x * bx + tx;
                uint k;
                volatile __shared__ unsigned char bins_loc[256];

                for (k=0; k<256; k++)
                    bins_loc[k] = 0;
                for (k=0; k<P; k++)
                    ++bins_loc[img[i*P+k]];
                __syncthreads();
                for (k=0; k<256; k++)
                    atomicAdd(&bins[k], bins_loc[k]);
}



"""

kernel_code = kernel_code_template
# compile the kernel code
mod = compiler.SourceModule(kernel_code)
# get the kernel function from the compiled module
given_hist = mod.get_function("given_Hist")
#advanced_hist=mod.get_function("advanced_hist")

if __name__ =='__main__':
    im_size=[1, 10, 50, 100, 500]
    for size in im_size:
        img, n_row, n_col=load_data(size)
        img_gpu = gpuarray.to_gpu(img)
        bin_gpu = gpuarray.zeros((1, 256), np.uint32)
        time_cpu=[]
        time_gpu1=[]
        time_gpu2=[]
        for i in range(1, 3):
            ##################################################################3
            #Convolution in CPU using signal.convolve2d
            ###################################################################
            start = time.time()
            bins_cpu = hist(img)
            end = time.time()
            time_cpu.append(end - start)

            #############################################################
            #Given histogram GPU code
            #############################################################
            start = time.time()
            given_hist(img_gpu, bin_gpu, np.uint32(32), grid=((n_row*n_col)//32+1, 1), block=(32, 1))
            #given_hist(img_gpu, bin_gpu, np.uint32(32), block=(32, 32, 1), grid=((n_row//32)+1,(n_col//32)+1, 1))
            end = time.time()
            time_gpu1.append(end - start)
            bin_gpu1 = bin_gpu.get()

            #############################################################
            #Advanced histogram GPU code
            #############################################################
            start = time.time()
            #advanced_hist(img_gpu, bin_gpu, np.uint32(32), grid=(N // 32, 1), block=(1, 1, 1))
            end = time.time()
            time_gpu2.append(end - start)
            bin_gpu2 = bin_gpu.get()

        print("This is a CPU result")
        print(bins_cpu)
        print(np.average(time_cpu))
        print '*' * 66
        print("This is a given GPU result")
        print(bin_gpu1)
        print 'Does GPU equal to CPU result?:', np.allclose(bin_gpu1, bins_cpu)#Validate the results
        print(np.average(time_gpu1))
        print("Given GPU time/AdvancedGPU time=={}".format(np.average(time_gpu2) / np.average(time_gpu1)))
        print '*' * 66
    # # Plot runtime graph
    # plt.ioff()
    # plt.gcf()
    # plt.plot(time_cpu, label="CPU")
    # plt.plot(time_gpu1, label="Given GPU")
    # plt.plot(time_gpu2, label="Advanced GPU")
    # plt.legend(loc='upper right')
    # plt.xlabel('Dimension of matrix')
    # plt.ylabel('time/second')
    # plt.gca().set_xlim(0, 500)
    # plt.savefig('hist_cuda.png')
