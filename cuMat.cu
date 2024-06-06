// BLAS:Basic Linear Algebra Subprograms
#include <cublas_v2.h>
// exception用
#include <stdexcept>
#include <iostream>
#include "cuMat.h"
using namespace std;

cuMat::cuMat::cuMat(int rows, int cols) : rows(rows), cols(cols)
{
    cublasCreate(&cudaHandle);
    cudaMallocManaged(&mDevice, rows * cols * sizeof(float));
    mHost = mDevice;
    cudaDeviceSynchronize();
}


cuMat::cuMat(const cuMat &a) : rows(a.rows), cols(a.cols)
{
    cublasCreate(&cudaHandle);
    cudaMallocManaged(&mDevice, rows * cols * sizeof(float));
    mHost = mDevice;
    cudaDeviceSynchronize();

    cudaError_t error = cudaMemcpy(mDevice, a.mDevice, 
                rows * cols * sizeof(float),
                cudaMemcpyDeviceToDevice);
    
    if (error != cudaSuccess)
    {
        // エラー処理を行う
        throw std::exception("cudaMemcpy failed!");
    }
}

cuMat::~cuMat()
{
    cudaFree(mDevice);
    cublasDestroy(cudaHandle);
}

void cuMat::new_matrix(int rows, int cols)
{
    
}

void cuMat::print() const
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << mHost[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void cuMat::plus(const cuMat &b, cuMat &r)
{
    float alpha = 1;
    float beta = 1;

    /*
    cublasSgeam:
        C=α⋅A+β⋅B
        ここで、A と B は入力行列、C は結果行列、alpha と beta はスカラー値です。

    引数:
    r.cudaHandle: CUBLASライブラリのハンドル。
    CUBLAS_OP_N: 行列の転置を行わないことを示します。
    rows: 行列の行数。
    cols: 行列の列数。
    &alpha: スカラー値 alpha のポインタ。
    mDevice: 元の行列のデータを指すデバイスポインタ。
    rows: 元の行列のリーディングディメンション（行数）。
    &beta: スカラー値 beta のポインタ。
    b.mDevice: 加算する行列 b のデータを指すデバイスポインタ。
    rows: 加算する行列 b のリーディングディメンション（行数）。
    r.mDevice: 結果を格納する行列 r のデータを指すデバイスポインタ。
    r.rows: 結果を格納する行列 r のリーディングディメンション（行数）。
    */
    cublasStatus_t stat = cublasSgeam(r.cudaHandle, CUBLAS_OP_N,
        CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta,
        b.mDevice, rows, r.mDevice, r.rows);
    
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cout << "cannot cublasSgeam" << endl;
    }
    cudaThreadSynchronize();
}

cuMat& cuMat::operator=(const cuMat &a) {
    new_matrix(a.rows, a.cols);

    cudaError_t error = cudaMemcpy(mDevice, a.mDevice,
            rows * cols * sizeof(*mDevice), cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess)
        printf("cuMat operator= cudaMemcpy error\n");

    return *this;
}

cuMat operator+(const cuMat &a, const cuMat &b) {
    cuMat r = a;
    r.plus(b, r);

    return r;
}