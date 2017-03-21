#!/usr/bin/env python


from PIL import Image
import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pycuda.autoinit


def getData(MB):
    # Create input image containing 8-bit pixels; the image contains N = R*C bytes;
    R = 1000 * MB
    C = 1000
    N = R * C
    img = np.memmap('/opt/data/random.dat', dtype=np.uint8, mode='r', shape=(R,C))
    return N, img


# Compute histogram in Python:
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

def build_kernel(glb_sz=2000, lc_sz = 4):
    ####################################
    # compile and build
    given_kernel_code = """
            __global__ void givenHist(unsigned char *img, unsigned int *bins, int P)
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
    mod = compiler.SourceModule(given_kernel_code)
    givenHist = mod.get_function("givenHist")
    ####################################
    # compile and build
    my_kernel_code = """
            __global__ void myHist(unsigned char *img, unsigned int *bins, const unsigned int N) {
                uint i = blockDim.x * blockIdx.x + threadIdx.x;
                unsigned int pvt_bin[256];
                unsigned int k;
                for (k=0; k<256; k++)
                    pvt_bin[k] = 0;
                for (k=0; k<N; k+=%(GLOBAL_SIZE)s)
                    ++pvt_bin[img[k+i]];
                for (k=0; k<256; k++)
                    atomicAdd(&bins[k], pvt_bin[k]);
            }
        """ % {'GLOBAL_SIZE': glb_sz}
    mod = compiler.SourceModule(my_kernel_code)
    myHist = mod.get_function("myHist")

    return givenHist, myHist

def given_kernel(givenHist, img_gpu, N, check=False, h_ck = None):
    print('given kernel')
    ####################################
    # running to get result
    bin_gpu = gpuarray.zeros((1, 256), np.uint32)
    givenHist(img_gpu, bin_gpu, np.uint32(32), grid = (N//32, 1),block= (1,1,1))
    result = bin_gpu.get()
    print('given kernel result')
    print result

    if check:
        print('are they all the same')
        print np.allclose(result, h_ck)
    ####################################
    # calculate time
    print('average time')
    start = time.time()
    for i in xrange(50):
        givenHist(img_gpu, bin_gpu, np.uint32(32), grid=(N // 32, 1), block=(1, 1, 1))
    gtime = (time.time() - start) / 50
    print(gtime)
    return result, gtime

def my_kernel(myHist, img_gpu,N,  check=False, h_ck = None, glb_sz=2000, lc_sz = 4):
    print "my kernel"
    ####################################
    # running to get result
    bin_gpu = gpuarray.zeros((1, 256), np.uint32)
    myHist(img_gpu, bin_gpu, np.uint32(N), grid = (glb_sz//lc_sz, 1),block= (lc_sz,1,1))
    h_mop = bin_gpu.get()
    print('my kernel result')
    print h_mop

    if check:
        print('are they all the same')
        print np.allclose(h_mop, h_ck)
    ####################################
    # calculate time
    print('average time')
    start = time.time()
    for i in xrange(50):
        myHist(img_gpu, bin_gpu, np.uint32(N), grid=(glb_sz // lc_sz, 1), block=(lc_sz, 1, 1))
    mtime = (time.time() - start) / 50
    print(mtime)
    return h_mop, mtime


def cmpkernel(givenHist, myHist):
    print('loading 1MB data')
    N, img = getData(MB=1)
    print('python hist time')
    start = time.time()
    h_py = hist(img)
    ctime = time.time() - start
    print(ctime)
    print('python result')
    print h_py
    # initial data
    img_gpu = gpuarray.to_gpu(img)
    # run given kernel
    h_given, gtime = given_kernel(givenHist, img_gpu,N, check=True, h_ck=h_py)
    # run my kernel
    h_my, mtime = my_kernel(myHist, img_gpu,N,  check=True, h_ck=h_given)
    print "speed up"
    print gtime / mtime
    return gtime, mtime

def cmpkernelWithoutCpu(givenHist, myHist, MB=10):
    print '-' *30
    print('loading '+str(MB)+'MB data')
    N, img = getData(MB=MB)
    # initial data
    img_gpu = gpuarray.to_gpu(img)
    # run given kernel
    h_given, gtime = given_kernel(givenHist, img_gpu,N, check=False)
    # run my kernel
    h_mop, mtime = my_kernel(myHist, img_gpu,N,  check=True, h_ck=h_given)
    print "speed up"
    print gtime / mtime
    return gtime, mtime


if __name__ == '__main__':
    givenHist, myHist = build_kernel()
    # cmpkernel(givenHist, myHist)
    # print cmpkernelWithoutCpu(givenHist, myHist, 500)
    # N, img = getData(100)
    # img_gpu = gpuarray.to_gpu(img)
    # h_given = given_kernel(givenHist, img_gpu, False)
    # h_my = my_kernel(myHist, img_gpu, check=True, h_ck=h_given)
    given_time = []
    my_time = []
    x, y = cmpkernel(givenHist, myHist)
    given_time.append(x)
    my_time.append(y)
    x, y = cmpkernelWithoutCpu(givenHist, myHist, MB=10)
    given_time.append(x)
    my_time.append(y)
    x, y = cmpkernelWithoutCpu(givenHist, myHist, MB=50)
    given_time.append(x)
    my_time.append(y)
    x, y = cmpkernelWithoutCpu(givenHist, myHist, MB=100)
    given_time.append(x)
    my_time.append(y)
    x, y = cmpkernelWithoutCpu(givenHist, myHist, MB=500)
    given_time.append(x)
    my_time.append(y)

    plt.gcf()
    plt.plot([1, 10, 50, 100, 500], my_time, 'o-', label="my time")
    plt.plot([1, 10, 50, 100, 500], given_time, 'o-', label="given time")
    plt.xlim(0, 600)
    plt.title("Compare histogram time with given kernel and my kernel")
    plt.xlabel('size')
    plt.ylabel('time/second')
    plt.legend()
    plt.savefig('plotCuda.png')
    plt.close()
