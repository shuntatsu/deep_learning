// BLAS:Basic Linear Algebra Subprograms
#include <cublas_v2.h>
// exception用
#include <stdexcept>
#include <iostream>
#include "cuMat.h"
using namespace std;

/*
コピーコンストラクタです。既存のcuMatオブジェクトを引数に取り、そのオブジェクトと同じ内容の新しいcuMatオブジェクトを作成します。
コピーコンストラクタの特徴は:
  クラス自身の型の参照を1つだけ引数に取る
  引数はconst参照である
  新しいオブジェクトを引数のオブジェクトと同じ状態に初期化する
コピーコンストラクタは、以下のような場面で暗黙的に呼び出されます:
  関数の引数渡しでオブジェクトがコピーされるとき
  関数の戻り値でオブジェクトが返されるとき
  オブジェクトを別のオブジェクトで初期化するとき (例: cuMat mat2 = mat1;)
*/
/* 
関数: cuMat
    コンストラクタ。行列のサイズを指定して初期化する。
引数:
    rows: 行列の行数
    cols: 行列の列数  
戻り値:
    なし
*/
cuMat(int rows, int cols)
{
    cublasCreate(&cudaHandle);
    cudThreadSynchronize();
    new matrix(rows, cols); // ここは new_matrix(rows, cols) の呼び出しにすべき
}

/*
関数: cuMat 
    コピーコンストラクタ。他の cuMat オブジェクトからディープコピーを作成する。
引数:
    a: コピー元の cuMat オブジェクト
戻り値:
    なし
*/
cuMat::cuMat(const cuMat &a) : rows(a.rows), cols(a.cols)
{
    cublasCreate(&cudaHandle);
    // ホストとデバイスの両方からアクセス可能なメモリを割り当て 
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

/*
関数: ~cuMat
    デストラクタ。割り当てたメモリを解放する。 
引数:
    なし
戻り値:
    なし
*/
cuMat::~cuMat()
{
    cudaFree(mDevice);
    cublasDestroy(cudaHandle);
}

/*
関数: new_matrix
    行列のサイズを変更する。
引数: 
    rows: 新しい行数
    cols: 新しい列数
戻り値:
    なし
*/
void cuMat::new_matrix(int rows, int cols)
{
    // ここに行列のサイズ変更処理を実装する必要がある
    // 古いメモリを解放し、新しいサイズでメモリを割り当てる
}

/*
関数: print
    行列の内容を標準出力に表示する。
引数:
    なし 
戻り値:
    なし
*/
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

/*
関数: plus
    2つの行列の要素ごとの和を計算する。
引数:
    b: 加算する行列
    r: 結果を格納する行列
戻り値:
    なし
*/
void cuMat::plus(const cuMat &b, cuMat &r)
{
    float alpha = 1;
    float beta = 1;

    // 2つの単精度実数行列の和を計算します。この関数を使うと、行列の足し算と同時に、転置や共役転置も効率的に行えます。
    cublasStatus_t stat = cublasSgeam(r.cudaHandle, CUBLAS_OP_N,
        CUBLAS_OP_N, rows, cols, &alpha, mDevice, rows, &beta,
        b.mDevice, rows, r.mDevice, r.rows);
    
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cout << "cannot cublasSgeam" << endl;
    }
    cudaThreadSynchronize();
}

/*
関数: operator=
    代入演算子のオーバーロード。ディープコピーを実行する。
引数:
    a: 代入元の cuMat オブジェクト
戻り値:
    *this
*/
cuMat& cuMat::operator=(const cuMat &a) {
    new_matrix(a.rows, a.cols);

    cudaError_t error = cudaMemcpy(mDevice, a.mDevice,
            rows * cols * sizeof(*mDevice), cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess)
        printf("cuMat operator= cudaMemcpy error\n");

    return *this;
}

/*
関数: operator+
    加算演算子のオーバーロード。2つの cuMat オブジェクトの和を計算する。
引数:
    a: 左辺値の cuMat オブジェクト
    b: 右辺値の cuMat オブジェクト 
戻り値:
    a と b の和である新しい cuMat オブジェクト
*/
friend cuMat operator+(const cuMat &a, const cuMat &b) {
    cuMat r = a;
    r.plus(b, r);

    return r;
}
