
#define PPROG_IMPLEMENTATION
#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#define N 1024
#define BLOCK_SIZE 32
#define MEMORY_TRANSFER_TIMER 0
#define TILE_WIDTH 32

#define cuda_check_error()                                           \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess)                                        \
        {                                                            \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            std::exit(EXIT_FAILURE);                                 \
        }                                                            \
    }

// NOTE: use this function to visualize the matrices
void print_matrix(const std::vector<float> &matrix)
{
    printf("\n");
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            printf("%5.2f ", matrix[y * N + x]);
        }
        printf("\n");
    }
}

__global__ void matmul_gpu(float *A, float *B, float *O, int n)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    if (Row >= n || Col >= n)
        return;

    float Pvalue = 0;

    for (int p = 0; p < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++p)
    {
        if (Row < n && (p * TILE_WIDTH + tx) < n)
            ds_M[ty][tx] = A[Row * n + p * TILE_WIDTH + tx];
        else
            ds_M[ty][tx] = 0.0f;

        if (Col < n && (p * TILE_WIDTH + ty) < n)
            ds_N[ty][tx] = B[(p * TILE_WIDTH + ty) * n + Col];
        else
            ds_N[ty][tx] = 0.0f;

        __syncthreads();

        // FIX: Evita accessi fuori dalla shared memory
        int valid_width = min(TILE_WIDTH, n - p * TILE_WIDTH);
        for (int i = 0; i < valid_width; ++i)
        {
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
        }

        __syncthreads();
    }

    if (Row < n && Col < n)
        O[Row * n + Col] = Pvalue;
}

//@@ Implement the CPU version of matmul
std::vector<float> matmul_cpu(const std::vector<float> &A, const std::vector<float> &B)
{
    std::vector<float> vec(N * N, 0);

    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            float sum = 0;
            for (int k = 0; k < N; ++k)
            {
                sum += A[r * N + k] * B[k * N + c];
            }
            vec[r * N + c] = sum;
        }
    }
    return vec;
}

int main(int argc, char **argv)
{

    // Generates random matrix data
    printf("generating random data...\n");
    auto A = random_square_matrix<float>(N, 1999);
    auto B = random_square_matrix<float>(N, 2000);
    printf("done\n");

    std::vector<float> result_gpu;
    result_gpu.resize(A.size());

    // print_matrix(A);
    // print_matrix(B);

    //@@ Setup blocks and threads count
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //@@ Allocate all necessary matrices in GPU memory
    float *d_A, *d_B, *d_O;
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_O, N * N * sizeof(float));
    cuda_check_error();
    auto timer_gpu = timerGPU{};

#if MEMORY_TRANSFER_TIMER == 1
    timer_gpu.start();
#endif

    //@@ Copy data from CPU to GPU
    cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, N * N * sizeof(float)); // Inizializza a zero la matrice di output
    cuda_check_error();

#if MEMORY_TRANSFER_TIMER == 0
    timer_gpu.start();
#endif

    //@@ Invoke kernel
    matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_O, N);
    cudaDeviceSynchronize();
    cuda_check_error()

#if MEMORY_TRANSFER_TIMER == 0
        timer_gpu.stop();
#endif

    //@@ Move result matrix to CPU memory
    cudaMemcpy(result_gpu.data(), d_O, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check_error();

#if MEMORY_TRANSFER_TIMER == 1
    timer_gpu.stop();
#endif

    // print_matrix(result);

    //@@ Free cuda memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_O);
    cuda_check_error();

    auto timer_cpu = timerCPU{};
    timer_cpu.start();
    auto solution_cpu = matmul_cpu(A, B);
    // print_matrix(solution_cpu);
    timer_cpu.stop();

    auto gpu_ms = timer_gpu.elapsed_ms();
    auto cpu_ms = timer_cpu.elapsed_ms();

    printf("GPU matmul elapsed time: %f ms\n", gpu_ms);
    printf("CPU matmul elapsed time: %f ms\n", cpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);

    // Check solution
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            auto rval = result_gpu[y * N + x];
            auto cval = solution_cpu[y * N + x];
            if (!float_approx_equal(rval, cval, 1e-6))
            {
                printf("invalid element at (%d, %d):\t: gpu = %f\tcpu = %f\n",
                       y, x, rval, cval);
                exit(1);
            }
        }
    }

    printf("Solution is similar!\n");
    return 0;
}
