
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define PPROG_TIMER
#include <timer.cuh>

#define BLOCK_SIDE_LEN 32
#define BLUR_SIZE 5

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

//@@ Write the blur kernel
__global__ void blur_gpu(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int pixels = 0;
    int somma_red = 0;
    int somma_green = 0;
    int somma_blue = 0;
    int index = 0;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("%d %d", Col, Row);
    if (Col < width && Row < height)
    {
        // printf("colonna %d riga %d", Col, Row);
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
        {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
            {
                index = ((Row + blurRow) * width + (Col + blurCol)) * channels;
                if (index > -1 && index < width * height * channels)
                {
                    somma_red += input[index];
                    somma_green += input[index + 1];
                    somma_blue += input[index + 2];
                    pixels++;
                }
            }
        }
        output[(Row * width + Col) * channels] = (unsigned char)(somma_red / pixels);
        output[((Row * width + Col) * channels) + 1] = (unsigned char)(somma_green / pixels);
        output[((Row * width + Col) * channels) + 2] = (unsigned char)(somma_blue / pixels);
    }
}

//@@ Write the CPU version of the blur kernel
void blur_cpu(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int pixels = 0;
    int somma_red = 0;
    int somma_green = 0;
    int somma_blue = 0;
    int index = 0;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
            {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
                {
                    index = ((i + blurRow) * width + (j + blurCol)) * channels;
                    if (index > -1 && index < width * height * channels)
                    {
                        somma_red += input[index];
                        somma_green += input[index + 1];
                        somma_blue += input[index + 2];
                        pixels++;
                    }
                }
            }
            output[(i * width + j) * channels] = (unsigned char)(somma_red / pixels);
            output[((i * width + j) * channels) + 1] = (unsigned char)(somma_green / pixels);
            output[((i * width + j) * channels) + 2] = (unsigned char)(somma_blue / pixels);
            somma_red = 0;
            somma_green = 0;
            somma_blue = 0;
            pixels = 0;
        }
    }
}

int main(int argc, char **argv)
{

    // Load image data from file
    int w, h, c;
    unsigned char *h_img = stbi_load("image.png", &w, &h, &c, 0);
    if (h_img == NULL)
    {
        printf("Error in loading the image\n");
        exit(1);
    }

    const auto img_size = sizeof(unsigned char) * w * h * c;

    unsigned char *result_cpu = new unsigned char[img_size];

    auto timer_cpu = timerCPU{};
    timer_cpu.start();
    blur_cpu(h_img, result_cpu, w, h, c);
    timer_cpu.stop();

    if (!stbi_write_png("result_cpu.png", w, h, c, result_cpu, w * c))
    {
        printf("Error in saving the image\n");
        exit(1);
    }

    delete result_cpu;

    //@@ Allocate GPU memory
    unsigned char *d_img, *d_output;
    cudaMalloc(&d_img, img_size);
    // cudaMalloc(&d_output, img_size);
    cudaMallocManaged(&d_output, img_size);

    //@@ Transfer memory to GPU
    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);

    //@@ Setup blocks and threads count
    dim3 threadsPerBlock(BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
    dim3 blocksPerGrid((w + BLOCK_SIDE_LEN - 1) / BLOCK_SIDE_LEN, (h + BLOCK_SIDE_LEN - 1) / BLOCK_SIDE_LEN);

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    //@@ Invoke kernel
    blur_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_output, w, h, c);
    cudaDeviceSynchronize();
    cuda_check_error();

    timer_gpu.stop();

    delete[] result_cpu;

    //@@ Move result to CPU memory
    // cudaMemcpy(result_cpu, d_output, img_size, cudaMemcpyDeviceToHost);
    cuda_check_error();

    // Save image to disk
    if (!stbi_write_png("result_gpu.png", w, h, c, d_output, w * c))
    {
        printf("Error in saving the image\n");
        exit(1);
    }

    auto gpu_ms = timer_gpu.elapsed_ms();
    auto cpu_ms = timer_cpu.elapsed_ms();

    printf("GPU matmul elapsed time: %f ms\n", gpu_ms);
    printf("CPU matmul elapsed time: %f ms\n", cpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);

    //@@ Free cuda memory
    cudaFree(d_img);
    cudaFree(d_output);

    stbi_image_free(h_img);
    return 0;
}
