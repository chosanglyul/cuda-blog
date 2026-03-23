#include <stdio.h>

#include <chrono>
using namespace std::chrono;

#include "conv1D.h"

constexpr int N = 1'000'000;
constexpr int M = 10'000;
constexpr int sizN = sizeof(float)*N;
constexpr int sizM = sizeof(float)*M;
constexpr double eps = 1e-7;

int main(void) {
    // set seed to zero for reproducing result
    srand(0);

    // allocate host memory
    float * A_host = (float *)malloc(sizN);
    float * B_host = (float *)malloc(sizN);
    float * C_host_output = (float *)malloc(sizM);
    float * C_host_answer = (float *)malloc(sizM);

    // allocate device memory
    float * A_device = NULL, * B_device = NULL, * C_device = NULL;
    cudaMalloc((void **)&A_device, sizN);
    cudaMalloc((void **)&B_device, sizN);
    cudaMalloc((void **)&C_device, sizM);

    conv1Dcpu(A_host, B_host, C_host_answer);

    conv1D(N, M, A_host, B_host, C_host_output);

    for (int i = 0)

}

void conv1Dcpu(const float * A, const float * B, float * C) {
    for (int i = 0; i < N; i++) {
        C[i] = 0;
        int idx = i-M/2;
        for (int j = 0; j < M; j++) {
            int jdx = idx+j;
            if (jdx >= 0 && jdx < N)
                C[i] += A[jdx]*B[j];
        }
    }
}
