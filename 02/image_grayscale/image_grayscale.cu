
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define PPROG_TIMER
#include <timer.cuh>

#define BLOCK_SIDE_LEN 32

//@@ Write the grayscale kernel
__global__ void grayscale_gpu(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // scorro righe, e poi scorro all'interno della riga
    // indice grey img
    int index = y * width + x;

    if (index < width * height)
    {
        output[index] = input[index * channels] * 0.21 + (input[index * channels + 1]) * 0.71 + (input[index * channels + 2]) * 0.07;
    }
}

//@@ Write the sequential version of grayscale
void grayscale_cpu(unsigned char *input, unsigned char *output, int width, int height, int channels)
{

    for (int i = 0; i < width * height * channels; i += channels)
    {
        output[i / channels] = input[i] * 0.21 + (input[i + 1]) * 0.71 + (input[i + 2]) * 0.07;
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
    grayscale_cpu(h_img, result_cpu, w, h, c);
    timer_cpu.stop();

    if (!stbi_write_png("result_cpu.png", w, h, 1, result_cpu, w * 1))
    {
        printf("Error in saving the image\n");
        exit(1);
    }

    delete result_cpu;

    unsigned char *d_img, *d_output;

    //@@ Allocate GPU memory
    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_output, img_size / c);

    //@@ Transfer memory from CPU to GPU
    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);

    //@@ Setup blocks and threads count
    int threadsPerBlock = 1024;
    int blocksPerGrid = (((w * h) * c) + threadsPerBlock - 1) / threadsPerBlock;

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    //@@ Invoke kernel
    grayscale_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_output, w, h, c);

    timer_gpu.stop();

    //@@ Move result image to CPU memory
    cudaMemcpy(result_cpu, d_output, img_size / c, cudaMemcpyDeviceToHost);

    // Save image to disk
    if (!stbi_write_png("result_gpu.png", w, h, 1, result_cpu, w * 1))
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
