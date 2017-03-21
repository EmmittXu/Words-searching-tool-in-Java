#This is HW3 for hybrid computing class
#Auther: Guowei Xu
#UNI:gx2127

from PIL import Image
import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
import matplotlib as mpl
from scipy import signal
mpl.use('agg')
import matplotlib.pyplot as plt



## Use the function below to dynamically change image size.
def create_img(filename, cols, rows):
    size = (cols, rows)
    im = Image.open(filename).convert('L')  # .convert('L') converts the image to grayscale
    im = im.resize(size)
    return np.array(im)


# -- initialize the device
import pycuda.autoinit
#This is the cuda kernel code
kernel_code_con = """
            __global__ void conv(float *A, float *K, float *C, int M, int N, int MASK_WIDTH)
            {
                const uint tile = %(TILE_SIZE)s;
                const uint bx = blockIdx.x;
                const uint by = blockIdx.y;
                const uint tx = threadIdx.x;
                const uint ty = threadIdx.y;
                const uint row = tile * by + ty;
                const uint col = tile * bx + tx;
                int row_start = row - MASK_WIDTH/2; //Row of the upper-left starting point of the mask
                int col_start = col - MASK_WIDTH/2; //Column of the upper-left starting point of the mask
                float sum = 0.0;
                float tmp = 0.0;

                for(int x = row_start;x<row_start+MASK_WIDTH;x++)
                {
                    for(int y = col_start;y<col_start+MASK_WIDTH;y++)
                    {
                        if((x<0 ||x>(M-1)) || (y<0 ||y>(N-1)))//boundary condition
                        {
                            tmp = 0.0;//For thoes beyond boundary, we set it to zero
                        }else{
                            tmp = A[x*N+y];
                        }
                        sum += tmp * K[(x-row_start)*MASK_WIDTH+(y-col_start)];//sum up all products of mask elements and input block
                    }
                }
                if ((row<M)&&(col<N))//boundary check
                {
                    C[row*N+col]=sum;
                }

            }
            """


# get the kernel code from the template
kernel_code = kernel_code_con
#Set tile size
TILE_SIZE = 2
kernel_code = kernel_code % {'TILE_SIZE': TILE_SIZE}
# compile the kernel code
mod = compiler.SourceModule(kernel_code)
# get the kernel function from the compiled module
convolution = mod.get_function("conv")

if __name__ =='__main__':
    ##################################################################################################
    #########Part 1, problem 1 and 2 start here
    ##################################################################################################
    print("Part 1 problem 1 and 2")
    print("Compare the performance of your algorithm against the scipy.signal.convolve2d function.")
    print("Compare the speed of your function for different 'A' matrix sizes, keeping kernel size constant.")
    time_cpu = []#to record cpu run time
    time_gpu = []#to record gpu runtime
    # change the size of matrix from 4*5 to 400*500
    for k in range(1, 100):
        M = 4 * k
        N = 5 * k
        Mask_Dim = 3#mask size 3*3
        #Initialize a random M*N dimension matrix
        matrix = np.random.rand(M, N).astype(np.float32)
        #Here we temporarily define a mask with all elements equal to 1
        mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32)
        # Declare gpu memory for input matrix, mask and output matrix
        A_gpu = gpuarray.to_gpu(matrix)
        K_gpu = gpuarray.to_gpu(mask)
        c_gpu = gpuarray.empty((M,N), matrix.dtype)

        for i in range(1, 9):
            #############################################################
            #Convolution in GPU
            #############################################################
            start = time.time()
            convolution(A_gpu, K_gpu, c_gpu, np.uint32(M), np.uint32(N), np.uint32(Mask_Dim),
                      grid=((N - 1) // TILE_SIZE + 1, (M - 1) // TILE_SIZE + 1),
                      block=(TILE_SIZE, TILE_SIZE, 1), )
            end = time.time()
            time_gpu.append(end - start)
            c = c_gpu.get()

            ##################################################################3
            #Convolution in CPU using signal.convolve2d
            ###################################################################
            start = time.time()
            con_cpu = signal.convolve2d(matrix, mask, mode='same')
            end = time.time()
            time_cpu.append(end - start)

        print("This is a CPU convolution result")
        print(con_cpu)
        print(np.average(time_cpu))
        print '*' * 66
        print("This is a GPU convolution result")
        print(c)
        print 'Does GPU equal to CPU result?:', np.allclose(con_cpu, c)#Validate the results
        print(np.average(time_gpu))
        print("GPU time/CPU time=={}".format(np.average(time_gpu) / np.average(time_cpu)))
        print '*' * 66
    # Plot runtime graph
    plt.ioff()
    plt.gcf()
    plt.plot(time_cpu, label="CPU")
    plt.plot(time_gpu, label="GPU")
    plt.legend(loc='upper right')
    plt.xlabel('Dimension of matrix')
    plt.ylabel('time/second')
    plt.gca().set_xlim(0, M)
    plt.savefig('part1_cuda.png')

    ############################################################################################
    ########Part 1, problem 3 starts from here
    #######Repeat this process for different 'mask' sizes, keeping 'A' matrix size constant
    #######(preferably keep 'A' matrix size as atleast 200x100).
    ############################################################################################
    print '#' * 66
    print '#' * 66
    print '#' * 66
    print("This is part 3")
    M3 = 200  # Row dimension of mask
    N3 = 300  # Column dimension of mask
    time_cpu = []
    time_gpu = []
    # Initialize a random matrix with the dimention of M3*N3 that is to do convolution
    matrix3 = np.random.rand(M3, N3).astype(np.float32)
    # Declare gpu memory
    A_gpu = gpuarray.to_gpu(matrix3)
    c_gpu = gpuarray.empty((M3, N3), matrix3.dtype)
    # Enlarge mask dimention from 3*3 to 90*90
    for m in range(1, 30):
        Mask_Dim3 = 3 * m
        #For simplicity we temporarily set mask elements to 1
        mask3 = np.array([[1, 1, 1] * m, [1, 3, 1] * m, [1, 1, 1] * m] * m).astype(np.float32)
        #Declare gpu memory for mask
        K_gpu = gpuarray.to_gpu(mask3)
        for i in range(1, 9):
            c3 = 0.0  # gpu result variable
            con_cpu3 = 0.0  # cpu result variable
            ########################################################################
            # Convolution using GPU
            #########################################################################
            start = time.time()
            convolution(A_gpu, K_gpu, c_gpu, np.uint32(M3), np.uint32(N3), np.uint32(Mask_Dim3),
                      grid=((N3 - 1) // TILE_SIZE + 1, (M3 - 1) // TILE_SIZE + 1),
                      block=(TILE_SIZE, TILE_SIZE, 1), )
            end = time.time()
            time_gpu.append(end - start)
            c3 = c_gpu.get()
            #############################################################################
            # Convolution using CPU
            ###############################################################################
            start = time.time()
            con_cpu3 = signal.convolve2d(matrix3, mask3, mode='same')
            end = time.time()
            time_cpu.append(end - start)

        print("This is a CPU convolution result")
        print(con_cpu3)
        print 'CPU Time:', np.average(time_cpu)
        print '*' * 66
        print("This is a GPU convolution result")
        print(c3)
        print 'Does GPU equal to CPU result?:', np.allclose(con_cpu3, c3)  # Validate the results
        print  'GPU Time:', np.average(time_gpu)
        # Compare the speed of gpu code when increasing mask dimension
        print("GPU time/CPU time=={}".format(np.average(time_gpu) / np.average(time_cpu)))
        print '*' * 66
    # Plot runtime graph
    plt.ioff()
    plt.gcf()
    plt.plot(time_cpu, label="CPU")
    plt.plot(time_gpu, label="GPU")
    plt.legend(loc='upper right')
    plt.xlabel('Mask dimension')
    plt.ylabel('time/second')
    plt.gca().set_xlim(0, 100)
    plt.savefig('part2_cuda.png')

    ############################################################################################
    ########Part 2 starts from here
    ########Image filtering
    ############################################################################################
    filters = {
        'identity': np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]).astype(np.float32),
        'sharpen': np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0]]).astype(np.float32),
        'blur': np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]).astype(np.float32),
        'edge_det': np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0]]).astype(np.float32),
        'emboss': np.array([[2., 1., 0.], [1., 1., -1.], [0., -1., -2]]).astype(np.float32),
        'sob_x': np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1]]).astype(np.float32),
        'sob_y': np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).astype(np.float32),
        'smooth_5x5': np.array([[0., 1., 2., 1., 0.], [1., 4., 8., 4., 1.], [2., 8., 16., 8., 2.], [1., 4., 8., 4., 1.],
                                [0., 1., 2., 1., 0.]]).astype(np.float32)
    }

    # Create a np.array matrix of the image
    im = create_img("./thrones-002-2.jpg", 612, 380)
    im = im.astype(np.float32)
    #Because the original image is 380*612, I'll just keep it
    height = 380
    width = 612

    # For each filter provided in filters, we do both cpu and gpu convolution
    # and plot the original, cpu version and gpu version in the same graph
    for filter in filters:
        print "Using filter: " + filter
        print(filters[filter])
        if filter == 'smooth_5x5':
            Mask_Dim = 5
        else:
            Mask_Dim = 3
        time_cpu = []
        time_gpu = []
        A_gpu = gpuarray.to_gpu(im)
        c_gpu = gpuarray.empty((height, width), im.dtype)
        K_gpu = gpuarray.to_gpu(filters[filter])
        # Declare gpu memory for different mask matrix
        #################################################################
        # Do filtering using gpu
        ##################################################################
        start = time.time()
        convolution(A_gpu, K_gpu, c_gpu, np.uint32(height), np.uint32(width), np.uint32(Mask_Dim),
                      grid=((width - 1) // TILE_SIZE + 1, (height - 1) // TILE_SIZE + 1),
                      block=(TILE_SIZE, TILE_SIZE, 1), )
        end = time.time()
        time_gpu.append(end - start)
        c = c_gpu.get()
        #################################################################
        # Do filtering using cpu
        ##################################################################
        start = time.time()
        con_cpu = signal.convolve2d(im, filters[filter], mode='same')
        end = time.time()
        time_cpu.append(end - start)

        # adjust min max range for visual effect
        if filter == 'blur' or filter == 'smooth_5x5':
            c = (c - np.amin(c)) / (np.amax(c) - np.amin(c)) * 255
            con_cpu = (con_cpu - np.amin(con_cpu)) / (np.amax(con_cpu) - np.amin(con_cpu)) * 255
        # Plot all figures side by side
        fig = plt.figure()
        plt.subplot(221)
        plt.title("Original")
        plt.imshow(im, cmap='Greys_r', vmax=255, vmin=0)
        plt.subplot(222)
        plt.title("GPU")
        plt.imshow(c, cmap='Greys_r', vmax=255, vmin=0)
        plt.subplot(223)
        plt.title("CPU")
        plt.imshow(con_cpu, cmap='Greys_r', vmax=255, vmin=0)
        plt.subplot_tool()
        plt.savefig('cuda ' + filter + '.png')
