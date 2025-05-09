#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n)
        c[index] = a[index] + b[index];
}

int main() {
    int n = 512;
    int size = n * sizeof(int);
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    for(int i = 0; i < n; i++){
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    add<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++){
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}
