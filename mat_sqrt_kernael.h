#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float mat_sqrt(float a, float alpha);

__global__ void mat_sqrt_kernel(const float* __restrict__ src,
                                float* __restrict__ dst, int m, int n, float alpha);

/*
C++コンパイラに対して、この関数がC言語の名前修飾規則を使用することを示します。これにより、C++コードからもC言語の関数として呼び出すことができます。
*/
#ifdef __cplusplus
extern "C" {
#endif

void mat_sqrt_kernel_exec(const float* src, float* dst, int m, int n, float alpha);

#ifdef __cplusplus
}
#endif