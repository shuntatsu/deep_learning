#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello(){
    printf("Hello CUDA World !!\n");
}

int main() {
    // hello関数を2ブロック, 各ブロック4スレッドで実行
    hello<<< 2, 4 >>>();
    // 全てのCUDAカーネルの実行が完了するまで待機
    cudaDeviceSynchronize();
    return 0;
}

/*
Hello CUDA World !!
Hello CUDA World !!
Hello CUDA World !!
Hello CUDA World !!
Hello CUDA World !!
Hello CUDA World !!
Hello CUDA World !!
Hello CUDA World !!
*/