#!/usr/bin/env python

"""
Basic 2d histogram.
"""
import time
import pyopencl as cl
import pyopencl.array
import numpy as np
import matplotlib as mpl

mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)


# Compute histogram in Python:
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins


# Function to load data, size could be 1, 10, 50, 100, 500
def load_data(size):
    R = size * 1000
    C = 1000
    img = np.memmap('/opt/data/random.dat', dtype=np.uint8, mode='r', shape=(R, C))
    return img, R, C


# Compute histogram in Python:
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

glb_sz=2000
lc_sz=4

given_hist = cl.Program(ctx, """
      __kernel void given_hist(__global unsigned char *img, __global unsigned int *bins,
                       const unsigned int P) {
        unsigned int i = get_global_id(0);
        unsigned int k;
        volatile __local unsigned char bins_loc[256];

        for (k=0; k<256; k++)
            bins_loc[k] = 0;
        for (k=0; k<P; k++)
            ++bins_loc[img[i*P+k]];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (k=0; k<256; k++)
            atomic_add(&bins[k], bins_loc[k]);
    }
    """).build().given_hist

my_hist = cl.Program(ctx, """
    __kernel void my_hist(__global unsigned char *img, __global unsigned int *bins, const unsigned int N) {
        unsigned int i = get_global_id(0);
        unsigned int pvt_bin[256];
        unsigned int k;
        for (k=0; k<256; k++)
            pvt_bin[k] = 0;
        for (k=0; k<N; k+=%(GLOBAL_SIZE)s)
            ++pvt_bin[img[k+i]];
        for (k=0; k<256; k++)
            atomic_add(&bins[k], pvt_bin[k]);
    } """ % {'GLOBAL_SIZE': glb_sz}).build().my_hist

given_hist.set_scalar_arg_dtypes([None, None, np.uint32])

if __name__ == '__main__':
    im_size = [1, 10, 50, 100, 500]
    for size in im_size:
        # img, n_row, n_col=load_data(size)
        n_row = 128
        n_col = 128
        img = np.random.randint(0, 255, n_row * n_col).astype(np.uint8).reshape(n_row, n_col)
        img_gpu = cl.array.to_device(queue, img)
        bin_gpu1 = cl.array.zeros(queue, 256, np.uint32)
        bin_gpu2 = cl.array.zeros(queue, 256, np.uint32)

        time_cpu = []
        time_gpu1 = []
        time_gpu2 = []
        for i in range(1, 2):
            ##################################################################3
            # CPU histogram goes here
            ###################################################################
            start = time.time()
            res_cpu = hist(img)
            end = time.time()
            time_cpu.append(end - start)

            #############################################################
            # Given histogram GPU code
            #############################################################
            start = time.time()
            given_hist(queue, (n_col*n_row / 32,), (1,), img_gpu.data, bin_gpu1.data, 32)
            end = time.time()
            time_gpu1.append(end - start)
            res_gpu1 = bin_gpu1.get()

            #############################################################
            # My histogram GPU code
            #############################################################
            start = time.time()
            my_hist(queue, (glb_sz,), (lc_sz,), img_gpu.data, bin_gpu2.data, np.uint32(n_row*n_col))
            end = time.time()
            time_gpu2.append(end - start)
            res_gpu2 = bin_gpu2.get()

        print("This is a CPU result")
        print(res_cpu)
        print(np.average(time_cpu))
        print '*' * 66
        print("This is a given GPU result")
        print(res_gpu1)
        print 'Does given GPU equal to CPU result?:', np.allclose(res_gpu1, res_cpu)  # Validate the results
        print("This is my GPU result")
        print(res_gpu2)
        print 'Does my GPU equal to CPU result?:', np.allclose(res_gpu2, res_cpu)  # Validate the results
        print(np.average(time_gpu2))
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




