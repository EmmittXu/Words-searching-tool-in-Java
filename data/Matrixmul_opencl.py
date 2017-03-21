import time
import pyopencl as cl
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def matrixmul_cpu(a, b):
    c = np.dot(a, b)
    return c



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

kernel = """
        __kernel void matrixmul_naive(__global float* a,__global float* b, __global float* c,
    const int Mdim, const int Ndim) {
        int i,j;
        i=get_global_id(0);
        j=get_global_id(1);
        float tmp;
        for(int k=0;k<Ndim;k++){
            tmp +=a[i*Ndim+k]*b[k*Mdim+j];
        }
        c[i*Mdim+j]=tmp;
    }

        __kernel void matrixmul_adv1(__global float* a,__global float* b, __global float* c,
    const int Mdim, const int Ndim) {
        int i,j,k;
        float temp;
        i= get_global_id(0);
        float wholerow[1024];
        // Fetch a row each time
        for(j=0;j<Ndim;j++){
            wholerow[j]=a[i*Ndim+j];
            }
            for(j=0;j<Mdim;j++){
                temp=0.0f;
                for(k=0;k<Ndim;k++){
                    temp+=wholerow[k]*b[k*Mdim+j];
                }
                c[i*Mdim+j]=temp;
            }
    }

        __kernel void matrixmul_adv2(__global float* a,__global float* b, __global float* c,
        const int Mdim, const int Ndim, __local float *Bwrk) {
            int k, j;
            int i = get_global_id(0);
            int iloc =get_local_id(0);
            int nloc = get_local_size(0);
            float Awrk[1024];
            float tmp;
            for(k=0; k<Ndim; k++){
                Awrk[k]=a[i*Ndim+k];
            }
            for(j=0;j<Mdim;j++){
                for(k=iloc;k<Ndim;k+=nloc)
                    Bwrk[k]=b[k*Mdim+j];
                barrier(CLK_LOCAL_MEM_FENCE);
                tmp=0.0f;
                for(k=0;k<Ndim;k++)
                    tmp +=Awrk[k] * Bwrk[k];
                c[i*Mdim+j] =tmp;
                barrier(CLK_LOCAL_MEM_FENCE);
                }
        }
    """


prg = cl.Program(ctx, kernel).build()

if __name__ == '__main__':
        time_cpu =[]
        time_naive = []
        time_adv1 = []
        time_adv2 = []
        for k in range(2, 20):
            M = 2 * k
            N = 3 * k
            matrix = np.array([[1, 2, 3] * k, [4, 5, 6] * k] * k).astype(np.float32)
            matrix_tran = np.transpose(matrix)
            matrix_tran_t = np.array([[1, 4] * k, [2, 5] * k, [3, 6] * k] * k).astype(np.float32)

            mf = cl.mem_flags
            a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix)
            b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix_tran_t)
            c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * M * 4)


            for i in xrange(4):
                start_cpu = time.time()
                # print(np.dot(matrix, matrix_tran))
                result_cpu=matrixmul_cpu(matrix, matrix_tran_t)
                time_cpu.append(time.time() - start_cpu)
            print('$'*66)
            print("CPU result")
            print(matrixmul_cpu(matrix, matrix_tran_t))
            print("CPU time")
            print(np.average(time_cpu))


            for i in xrange(4):
                start_naive = time.time()
                # print(np.dot(matrix, matrix_tran))
                prg.matrixmul_naive(queue, (M, M), None, a_buf, b_buf, c_buf, np.uint32(M), np.uint32(N))
                time_naive.append(time.time() - start_naive)
                c = np.zeros((M, M)).astype(np.float32)
                cl.enqueue_copy(queue, c, c_buf)
            print('$' * 66)
            print("GPU naive result")
            print(c)
            print("Naive time ")
            print(np.average(time_naive))
            print 'equal to cpu result?:', np.allclose(result_cpu, c)


            for i in xrange(4):
                start_adv1 = time.time()
                prg.matrixmul_adv1(queue, (M,), None, a_buf, b_buf, c_buf, np.uint32(M), np.uint32(N))
                time_adv1.append(time.time() - start_adv1)
                c = np.zeros((M, M)).astype(np.float32)
                cl.enqueue_copy(queue, c, c_buf)
            print'$' * 66
            print("GPU advanced 1 result")
            print(c)
            print("GPU advaned 1 time ")
            print(np.average(time_adv1))
            print 'equal to cpu result?:', np.allclose(result_cpu, c)


            for i in xrange(4):
                start_adv2 = time.time()
                prg.matrixmul_adv2(queue, (M,), None, a_buf, b_buf, c_buf, np.uint32(M), np.uint32(N), cl.LocalMemory(4 * N))
                time_adv2.append(time.time() - start_adv2)
                c = np.zeros((M, M)).astype(np.float32)
                cl.enqueue_copy(queue, c, c_buf)
            print'$' * 66
            print("GPU advanced 2 result")
            print(c)
            print("GPU advaned 2 time ")
            print(np.average(time_adv2))
            print 'equal to cpu result?:', np.allclose(result_cpu, c)


# Plot runtime graph
plt.ioff()
plt.gcf()
plt.plot(time_cpu, label="cpu")
plt.plot(time_naive, label="naive")
plt.plot(time_adv1, label="adv1")
plt.plot(time_adv2, label="adv2")
plt.legend(loc='upper right')
plt.xlabel('N')
plt.ylabel('time/second')
plt.gca().set_xlim(0, M)
plt.savefig('runtime_opencl.png')



