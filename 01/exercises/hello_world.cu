#include <cstdio>
#include <iostream>

//@@ Write GPU kernel and launch it from the main function with a single block
// and single thread

/* Definisco il GPU kernel -> e' una funzione che viene
   chiamata dalla cpu (host) e viene eseguita sulla gpu (CUDA device)
*/
__global__ void helloWorldKernel() {
  printf("Hello, World from the GPU!\n");
}

int main() {
  printf("Hello World!");

  //blocco: gruppo di threads che condividono memoria e altre proprieta' come la sincornizzazione
  helloWorldKernel<<<1, 1>>>(); //primo valore il numero di bloocchi, il secondo il numero di threads
  
  // aspetto che la gpu abbia finito prima di terminare il programma
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
    return -1;
  }

  return 0;
}
