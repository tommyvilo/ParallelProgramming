#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

const int N = 1 << 22; //shif a sinistra di 22, 2^22 -> 4194304

using timer = std::chrono::high_resolution_clock;

#define cuda_check_error()                                                     \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define cuda_safe_call(x)                                                      \
  {                                                                            \
    x;                                                                         \
    cuda_check_error();                                                        \
  }

//@@ Insert code to implement vector addition
__global__ void mainKernel(int* d_a,int* d_b,int* d_o,int N) {

  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalIndex<N) //evito di scrivere in aree di memoria no allocate (parlo per i threads di troppo che ho)
    d_o[globalIndex] = d_a[globalIndex] + d_b[globalIndex];

}

int main() {

  // HOST memory
  int *h_a = new int[N];
  int *h_b = new int[N];
  int *h_o = new int[N];

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(1, 100);

  for (int i = 0; i < N; i++) {
    h_a[i] = distribution(generator);
    h_b[i] = distribution(generator);
  }

  int *h_o_real = new int[N]; // Used to check the result
  auto t0 = timer::now();

  for (int i = 0; i < N; i++) {
    h_o_real[i] = h_a[i] + h_b[i];
  }

  auto t1 = timer::now();
  auto host_duration = t1 - t0;
  std::cout << "HOST version time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   host_duration)
                   .count()
            << " us" << std::endl;

  // -------------------------------------------------------------------------
  // Exercise

  int *d_a, *d_b, *d_o; // <- Device memory pointers

  //@@ Allocate GPU memory
  cuda_safe_call(cudaMalloc(&d_a, N*sizeof(int)));
  cuda_safe_call(cudaMalloc(&d_b, N*sizeof(int)));
  cuda_safe_call(cudaMalloc(&d_o, N*sizeof(int)));

  //@@ Copy memory to the GPU
  cuda_safe_call(cudaMemcpy(d_a,h_a,N*sizeof(int),cudaMemcpyHostToDevice));
  cuda_safe_call(cudaMemcpy(d_b,h_b,N*sizeof(int),cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 

  t0 = timer::now();

  //@@ Launch the GPU Kernel
  cuda_safe_call((mainKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_o, N)));
  cudaDeviceSynchronize();

  t1 = timer::now();
  auto gpu_duration = t1 - t0;

  std::cout << "GPU version time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   gpu_duration)
                   .count()
            << " us" << std::endl;

  std::cout << std::setprecision(1)
            << "speedup: " << host_duration / gpu_duration << "x\n\n";

  //@@ Copy the GPU memory back to the CPU here
  cuda_safe_call(cudaMemcpy(h_o,d_o,N*sizeof(int),cudaMemcpyDeviceToHost));

  //@@ Free the GPU memory here
  cuda_safe_call(cudaFree(d_a));
  cuda_safe_call(cudaFree(d_b));
  cuda_safe_call(cudaFree(d_o));


  // -------------------------------------------------------------------------
  // RESULT CHECK

  for (int i = 0; i < N; i++) {
    if (h_o[i] != h_o_real[i]) {
      std::cerr << "wrong result at: " << i << "\nhost:   " << h_o_real[i]
                << "\ndevice: " << h_o[i] << "\n\n";
      return 1;
    }
  }
  std::cout << "<> Correct\n\n";

  delete[] h_a;
  delete[] h_b;
  delete[] h_o;

  return 0;
}
