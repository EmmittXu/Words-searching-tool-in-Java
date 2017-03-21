#This is HW3 for hybrid computing class
#Auther: Guowei Xu
#UNI:gx2127


import time
import pyopencl as cl
import matplotlib as mpl
from scipy import signal
mpl.use('Agg')
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()
# Set up a command queue; we need to enable profiling to time GPU operations:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


################################################################################
#Convolution kernel
################################################################################
kernel = """
      __kernel void conv_opencl(__global float* input, __global float* Mask, __global float* output, const int M, const int N, const int M_Dim){
        int ou_i=get_global_id(0); //output row
        int ou_j=get_global_id(1); //output column
        int in_i=ou_i-M_Dim/2; //input row
        int in_j=ou_j-M_Dim/2; //input column
        float tmp=0.0;
        float sum=0.0;
        for(int p=in_i;p<M_Dim+in_i;p++){
            for(int q=in_j;q<M_Dim+in_j;q++){
                if((p<0 || p>=M) || (q<0 || q>=N)) //boundary condition
                {    tmp=0.0;
                }else{
                    tmp=input[p*N+q];
                }
                sum+=Mask[(p-in_i)*M_Dim+(q-in_j)]*tmp; //sum up all products of mask elements and input block
            }
            output[ou_i*N+ou_j]=sum;
        }

        }

        """
#compile the kernel
prg = cl.Program(ctx, kernel).build()


## Use the function below to dynamically change image size.
def create_img(filename, cols, rows):
    size = (cols, rows)
    im = Image.open(filename).convert('L')  # .convert('L') converts the image to grayscale
    im = im.resize(size)
    return np.array(im)




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
        mf = cl.mem_flags
        #Declare gpu memory for input matrix, mask and output matrix
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix)#input matrix
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask)#mask
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)#output convolution matrix
        for i in range(1, 9):
            #############################################################
            #Convolution in GPU
            #############################################################
            start = time.time()
            prg.conv_opencl(queue, (M, N), None, a_buf, b_buf, c_buf, np.uint32(M), np.uint32(N), np.uint32(Mask_Dim))
            end = time.time()
            time_gpu.append(end - start)
            c = np.zeros((M, N)).astype(np.float32)
            cl.enqueue_copy(queue, c, c_buf)#load output from gpu back to host
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
    plt.savefig('part1_opencl.png')










    ############################################################################################
    ########Part 1, problem 3 starts from here
    #######Repeat this process for different 'mask' sizes, keeping 'A' matrix size constant
    #######(preferably keep 'A' matrix size as atleast 200x100).
    ############################################################################################
    print '#' * 66
    print '#' * 66
    print '#' * 66
    print("This is part 3")
    M3 = 200#Row dimension of mask
    N3 = 300#Column dimension of mask
    time_cpu=[]
    time_gpu=[]
    #Initialize a random matrix with the dimention of M3*N3 that is to do convolution
    matrix3 = np.random.rand(M3, N3).astype(np.float32)
    #mf = cl.mem_flags
    a_buf3 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix3)
    c_buf3 = cl.Buffer(ctx, mf.WRITE_ONLY, M3 * N3 * 4)
    #Enlarge mask dimention
    for m in range(1, 30):
        Mask_Dim3 = 3 * m
        mask3=np.array([[1,1,1]*m,[1,3,1]*m,[1,1,1]*m]*m).astype(np.float32)
        b_buf3 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask3)

        for i in range(1, 9):
            c3=0.0#gpu result variable
            con_cpu3=0.0#cpu result variable
            ########################################################################
            #Convolution using GPU
            #########################################################################
            start = time.time()
            prg.conv_opencl(queue, (M3, N3), None, a_buf3, b_buf3, c_buf3, np.uint32(M3), np.uint32(N3),
                            np.uint32(Mask_Dim3))
            end = time.time()
            time_gpu.append(end - start)
            c3 = np.zeros((M3, N3)).astype(np.float32)
            cl.enqueue_copy(queue, c3, c_buf3)
            #############################################################################
            #Convolution using CPU
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
        print 'Does GPU equal to CPU result?:', np.allclose(con_cpu3, c3)#Validate the results
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
    plt.savefig('part2_opencl.png')










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

    #Create a np.array matrix of the image
    im = create_img("./thrones-002-2.jpg", 612,380)
    im=im.astype(np.float32)
    height=380
    width=612
    #Declare gpu memory for image input matrix
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=im)
    #Declare gpu memory for output image matrix
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, height * width * 4)
    #For each filter provided in filters, we do both cpu and gpu convolution
    #and plot the original, cpu version and gpu version in the same graph
    for filter in filters:
        print "Using filter: "+filter
        print(filters[filter])
        if filter=='smooth_5x5':
            Mask_Dim=5
        else:
            Mask_Dim=3
        time_cpu = []
        time_gpu = []
        #Declare gpu memory for different mask matrix
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=filters[filter])
        #################################################################
        #Do filtering using gpu
        ##################################################################
        start = time.time()
        prg.conv_opencl(queue, (height, width), None, a_buf, b_buf, c_buf, np.uint32(height), np.uint32(width),
                        np.uint32(Mask_Dim))
        end = time.time()
        time_gpu.append(end - start)
        c = np.zeros((height, width)).astype(np.float32)
        cl.enqueue_copy(queue, c, c_buf)
        #################################################################
        #Do filtering using cpu
        ##################################################################
        start = time.time()
        con_cpu = signal.convolve2d(im, filters[filter], mode='same')
        end = time.time()
        time_cpu.append(end - start)

        # adjust min max range for visual effect
        if filter == 'blur' or filter == 'smooth_5x5':
            c = (c- np.amin(c)) / (np.amax(c) - np.amin(c)) * 255
            con_cpu= (con_cpu - np.amin(con_cpu)) / (np.amax(con_cpu) - np.amin(con_cpu)) * 255
        #Plot all figures side by side
        fig=plt.figure()
        plt.subplot(221)
        plt.title("Original")
        plt.imshow(im,cmap='Greys_r',vmax=255, vmin=0)
        plt.subplot(222)
        plt.title("GPU")
        plt.imshow(c, cmap='Greys_r', vmax=255, vmin=0)
        plt.subplot(223)
        plt.title("CPU")
        plt.imshow(con_cpu, cmap='Greys_r', vmax=255, vmin=0)
        plt.subplot_tool()
        plt.savefig('Opencl '+filter+'.png')







