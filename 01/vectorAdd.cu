#include <stdio.h>

#include <chrono>
using namespace std::chrono;

constexpr int N = 100'000'000;
constexpr double eps = 1e-7;
constexpr unsigned int siz = N*sizeof(float);
constexpr int threads = 512;
constexpr int blocks = (N + threads - 1) / threads;

// kernel vectorAdd for calculate C=A+B
__global__ void vectorAdd(const float * A, const float * B, float * C) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int main(void) {
    // set seed to zero for reproducing result
    srand(0);

    // allocate host memory
    float * A_host = (float *)malloc(siz);
    float * B_host = (float *)malloc(siz);
    float * C_host_output = (float *)malloc(siz);
    float * C_host_answer = (float *)malloc(siz);

    // allocate device memory
    float * A_device = NULL, * B_device = NULL, * C_device = NULL;
    cudaMalloc((void **)&A_device, siz);
    cudaMalloc((void **)&B_device, siz);
    cudaMalloc((void **)&C_device, siz);


    float cpu_time = 0.0f, gpu_time = 0.0f;

    // init A and B vector
    for (int i = 0; i < N; i++) {
        A_host[i] = (float)rand()/RAND_MAX;
        B_host[i] = (float)rand()/RAND_MAX;
    }


    // copy A and B to device memory
    cudaMemcpy(A_device, A_host, siz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, siz, cudaMemcpyHostToDevice);

    // check start and end point to calculate gpu_time
    cudaEvent_t gpu_st, gpu_ed;
    cudaEventCreate(&gpu_st);
    cudaEventCreate(&gpu_ed);
    cudaEventRecord(gpu_st, 0);

    // calculate C=A+B by gpu
    vectorAdd<<<blocks, threads>>>(A_device, B_device, C_device);

    cudaEventRecord(gpu_ed, 0);
    cudaEventSynchronize(gpu_ed);
    cudaEventElapsedTime(&gpu_time, gpu_st, gpu_ed);

    // copy C to host memory
    cudaMemcpy(C_host_output, C_device, siz, cudaMemcpyDeviceToHost);


    // check start and end point to calculate cpu_time
    steady_clock::time_point cpu_st = steady_clock::now();

    // calculate C=A+B by cpu
    for (int i = 0; i < N; i++)
        C_host_answer[i] = A_host[i] + B_host[i];

    steady_clock::time_point cpu_ed = steady_clock::now();
    cpu_time = (float)duration_cast<nanoseconds>(cpu_ed - cpu_st).count() / 1e6;


    // check the answer is correct
    bool correct = true;
    for (int i = 0; i < N; i++)
        if (abs(C_host_answer[i] - C_host_output[i]) / max(1.0f, C_host_answer[i]) > eps)
            correct = false;


    // free allocated device memory
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    // free allocated host memory
    free(A_host);
    free(B_host);
    free(C_host_answer);
    free(C_host_output);


    // print the result
    if (correct) printf("The answer is correct! CPU: %f ms, GPU: %f ms\n", cpu_time, gpu_time);
    else printf("The answer is not correct. Please check CUDA installation.\n");
}
