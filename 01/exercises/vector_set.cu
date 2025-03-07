#include <cstdio>

#define cuda_check_error()                                                     \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

//@@ Write kernel for vector initialization
__global__ void mainKernel(int* vector,int N) {
  /*
    Per assegnare ad ogni thread uno spazio in memoria sul quale lavorare,
    devi calcolare l'indice globale del thread all'interno del kernel. 
    Questo indice globale è dato dalla combinazione dell'indice del blocco (blockIdx.x), 
    del numero di thread per blocco (blockDim.x) e dell'indice del thread all'interno del 
    blocco (threadIdx.x).
  */
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if(globalIndex<N) //evito di scrivere in aree di memoria no allocate (parlo per i threads di troppo che ho)
    vector[globalIndex] = 4;
}


int main() {

  const int VALUE = 4;
  const int N = 4097;

  int *vector;

  //@@ Allocate managed memory
  cudaError_t error = cudaMallocManaged(&vector, N * sizeof(int));
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
    return -1;
  }

  //@@ Initialize the grid and block dimensions
  int threadsPerBlock = 256; //numero comunemente usato perchè spesso ottimale per le gpu (nonche threads max per blocco dice chatGPT)
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; //formula da usare per capire quanti blocchi mi servono in totale per avere un completo parallelismo, il + threadsPerBlock - 1 serve per arrotondare per eccesso, in questo caso l'ultimo blocco avrà solo 41 threads attivi

  //@@ Launch kernel to set all elements of vector
  mainKernel<<<blocksPerGrid, threadsPerBlock>>>(vector,N); //primo valore il numero di bloocchi, il secondo il numero di threads
  
  // aspetto che la gpu abbia finito prima di terminare il programma
  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++) {
    printf("[%d]: %d\n", i, vector[i]);
    if (vector[i] != VALUE) {
      printf(">< INCORRECT: not equal to %d", VALUE);
      return 1;
    }
  }

  //@@ Free memory
  cudaFree(vector);

  return 0;
}
