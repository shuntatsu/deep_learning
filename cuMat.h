#pragma once
// BLAS:Basic Linear Algebra Subprograms
#include <cublas_v2.h>
// exception用
#include <stdexcept>
#include <iostream>

/*
C言語やC++では、通常、配列は行優先（row-major）で格納されますが、
Fortranでは列優先（column-major）で格納されます。
IDX2F マクロは、Fortranスタイルの列優先配列のインデックス計算を行います。
*/
#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))

class cuMat
{
private:
    float *mDevice = nullptr;
    float *mHost = nullptr;
    // row:行
    int rows = 0;
    // column:列
    int cols = 0;
    cublasHandle_t cudaHandle;

public:
    cuMat(int rows, int cols);
    ~cuMat();

    cuMat(const cuMat &a);
    void new_matrix(int rows, int cols);
    void print() const;
    
    // 算術用
    void plus(const cuMat &b, cuMat &r);
    cuMat &operator=(const cuMat &a);

    friend cuMat operator+(const cuMat &a, const cuMat &b);
};