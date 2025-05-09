#include <iostream>
#include <cstdlib>
using namespace std;

// CUDA kernel for matrix multiplication
__global__ void multiply(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Function to initialize matrix with random integers [0-9]
void initialize(int* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = rand() % 10;
    }
}

// Function to print matrix
void print(int* mat, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            cout << mat[row * N + col] << " ";
        }
        cout << '\n';
    }
    cout << '\n';
}

int main() {
    int N = 4; // ✅ You can change this to any size (2, 4, 16, etc.)
    int SIZE = N * N;
    size_t BYTES = SIZE * sizeof(int);

    // Allocate host matrices
    int* A = new int[SIZE];
    int* B = new int[SIZE];
    int* C = new int[SIZE];

    // Initialize matrices A and B
    initialize(A, N);
    initialize(B, N);

    cout << "Matrix A:\n";
    print(A, N);
    cout << "Matrix B:\n";
    print(B, N);

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, BYTES);
    cudaMalloc(&d_B, BYTES);
    cudaMalloc(&d_C, BYTES);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, BYTES, cudaMemcpyHostToDevice);

    // Define thread and block sizes
    int THREADS = 16;
    dim3 threads(THREADS, THREADS);
    dim3 blocks((N + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

    // Launch the kernel
    multiply<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    // Copy result matrix C from device to host
    cudaMemcpy(C, d_C, BYTES, cudaMemcpyDeviceToHost);

    cout << "Result of Matrix Multiplication:\n";
    print(C, N);

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
