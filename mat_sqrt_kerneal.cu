#include "mat_sqrt_kernel.h"
#define BLOCK_SIZE 32

/*
device:
    デバイス(GPU)上で実行される関数につける修飾子
    __device__関数は、カーネル関数(__global__関数)や他の__device__関数から呼び出すことができる
    ホスト(CPU)コードからは直接呼び出せない
global:
    カーネル関数につける修飾子。カーネル関数はデバイス(GPU)上で実行される
    ホスト(CPU)コードから呼び出すことができる。その際、実行設定(グリッド、ブロックサイズなど)を指定する
    __global__関数からは、他の__global__関数を呼び出せない
    戻り値の型はvoidでなければならない
*/

// GPU上で実行,インライン化を強制
__device__ __forceinline__ float mat_sqrt (float a, float alpha){
    return std::sqrt(a+alpha);
}

// カーネル関数を定義
/*
__restrict__ を使用することで、プログラマはコンパイラに対して「このポインタが指すメモリ領域は他のポインタと重ならない」と保証します。
これにより、コンパイラはより積極的な最適化を行うことができます。
*/
__global__ void mat_sqrt_kernel(const float* __restrict__ src, float* __restrict__ dst, int m, int n, float alpha) {
    // Block Index:グリッド内のスレッドブロックのインデックスを示します。
    // Block Dimension:1つのスレッドブロック内のスレッドの数
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        dst[row * n + col] = mat_sqrt(src[row * n + col], alpha);
    }
}

void mat_sqrt_kernel_exec(const float *src, float *dst, int m, int n, float alpha){
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);

    // カーネル関数の完了を待機
    mat_sqrt_kernel<<<grid, block>>>(src, dst, m, n, alpha);
    cudaThreadSynchronize();
}
