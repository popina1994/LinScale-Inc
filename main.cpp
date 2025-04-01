#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <boost/program_options.hpp>
#include <mkl_lapacke.h>
#include <mkl_cblas.h>
#include <mkl_vml.h>
#include <omp.h>  // OpenMP header

#include <chrono>  // For timing
namespace po = boost::program_options;


enum class MajorOrder
{
    ROW_MAJOR = 0,
    COL_MAJOR = 1
};

#define IDX(rowIdx, colIdx, width) ((rowIdx) * (width) + (colIdx))
#define IDX_R(rowIdx, colIdx, numRows, numCols) ((rowIdx) * (numCols) + (colIdx) )
#define IDX_C(rowIdx, colIdx, numRows, numCols) ((rowIdx)  + (colIdx) * (numRows))

template<MajorOrder majorOrder>
int64_t getPosId(int64_t rowIdx, int64_t colIdx, int64_t numRows, int64_t numCols)
{
    if (MajorOrder::ROW_MAJOR == majorOrder)
    {
        return IDX_R(rowIdx, colIdx, numRows, numCols);
    }
    else
    {
        return IDX_C(rowIdx, colIdx, numRows, numCols);
    }
}

template <typename T, MajorOrder majorOrder>
struct Matrix
{
    int numRows = 0;
    int numCols = 0;
    T* pArr = nullptr;
    Matrix(int _numRows, int _numCols): numRows(_numRows), numCols(_numCols)
    {
        pArr = new T[int64_t(numRows) * int64_t(numCols)];
        // std::cout << "CREATE" << pArr << std::endl;
    }

    Matrix(const Matrix& matIn) = delete;
    Matrix& operator=(const Matrix& matIn) = delete;
    Matrix(Matrix&& matIn)
    {
        if (pArr != nullptr)
        {
            delete [] pArr;
        }
        pArr = matIn.pArr;
        // std::cout << "MOVE " << pArr << std::endl;
        numRows = matIn.numRows;
        numCols = matIn.numCols;
        matIn.pArr = nullptr;
    }

    Matrix& operator=(Matrix&& matIn)
    {
        if (pArr != nullptr)
        {
            delete [] pArr;
        }
        pArr = matIn.pArr;
        // std::cout << "ASSIGN " << pArr << std::endl;
        numRows = matIn.numRows;
        numCols = matIn.numCols;
        matIn.pArr = nullptr;
        return *this;
    }

    ~Matrix()
    {
        if (pArr != nullptr)
        {
            // std::cout << "DELETE" << pArr  << std::endl;
            delete [] pArr;
            pArr = nullptr;
        }
    }

    T& operator()(int rowIdx, int colIdx)
    {
        int64_t posId = getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols);
        return pArr[posId];
    }
    T*& getData()
    {
        return pArr;
    }

    const T* getDataC() const
    {
        return pArr;
    }

    int getNumRows(void) const {
        return numRows;

    }
    int getNumCols(void) const
    {
        return numCols;
    }
};

using MatrixDCol = Matrix<double, MajorOrder::COL_MAJOR>;
using MatrixDRow = Matrix<double, MajorOrder::ROW_MAJOR>;

template <typename T, MajorOrder order>
void printMatrix(T* pArr, int numRows, int numCols, int numRowsCut, const std::string& fileName, bool upperTriangular = false)
{
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "WTF?" << fileName << std::endl;
    }
    for (int rowIdx = 0; rowIdx < std::min(numRows, numRowsCut); rowIdx++)
    {
        for (int colIdx = 0; colIdx < numCols; colIdx++)
        {
            if (colIdx > 0)
            {
                outFile << ",";
            }
            if (upperTriangular and (rowIdx > colIdx))
            {
                outFile << "0";
            }
            else
            {
                if constexpr (order == MajorOrder::ROW_MAJOR)
                {
                    outFile << pArr[IDX_R(rowIdx, colIdx, numRows, numCols)];
                }
                else
                {
                    outFile << pArr[IDX_C(rowIdx, colIdx, numRows, numCols)];
                }
            }

        }
        outFile << std::endl;
    }
}

template <typename T, MajorOrder order>
void copyMatrix(const T* pArr, T*& pArrOut, int numRows, int numCols, int numRowsCopy, int numColsCopy, bool upperTriangular = false)
{
    // pArrOut = new T[numRowsCopy * numColsCopy];
    for (int rowIdx = 0; rowIdx < numRowsCopy; rowIdx++)
    {
        for (int colIdx = 0; colIdx < numColsCopy; colIdx++)
        {
            if (upperTriangular and (rowIdx > colIdx))
            {
                pArrOut[getPosId<order>(rowIdx, colIdx, numRowsCopy,  numColsCopy)]  = 0;
            }
            else
            {
                pArrOut[getPosId<order>(rowIdx, colIdx, numRowsCopy,  numColsCopy)]  = pArr[getPosId<order>(rowIdx, colIdx, numRows,  numCols)];
            }
        }
    }
}

template <typename T, MajorOrder order>
void concatenateMatrices(Matrix<T, order>& mat1, Matrix<T, order>& mat2,
    Matrix<T, order>& matOut,
    bool horizontal = true)
{
    matOut = Matrix<T, order> {mat1.getNumRows() + mat2.getNumRows(), mat1.getNumCols()};
    int numRowsOut = matOut.getNumRows();
    int numColsOut = matOut.getNumCols();
    for (int rowIdx = 0; rowIdx < mat1.getNumRows(); rowIdx++)
    {
        for (int colIdx = 0; colIdx < numColsOut; colIdx++)
        {
            matOut(rowIdx, colIdx) = mat1(rowIdx, colIdx);
        }
    }

    for (int rowIdx = 0; rowIdx < mat2.getNumRows(); rowIdx++)
    {
        for (int colIdx = 0; colIdx < numColsOut; colIdx++)
        {
            matOut(rowIdx + mat1.getNumRows(), colIdx) = mat1(rowIdx, colIdx);
        }
    }
}


// column major version
template <typename T>
Matrix<T, MajorOrder::ROW_MAJOR> generateRandom(int numRows, int numCols, int seed)
{
    std::mt19937 gen(seed); // Fixed seed
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    auto matA = std::move(Matrix<T, MajorOrder::ROW_MAJOR>{numRows, numCols});
    // col_major
    for (int colIdx = 0; colIdx < numCols; colIdx++)
    {
        for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
        {
            matA(rowIdx, colIdx) = dist(gen);
        }
    }
    return std::move(matA);
}

template<typename T, MajorOrder orderInput, MajorOrder orderOutput>
void generateCartesianProduct(Matrix<T, orderInput>& mat1,  Matrix<T, orderInput>& mat2,
    Matrix<T, orderOutput>& matOut)
{
    int numRows = mat1.getNumRows() * mat2.getNumRows();
    int numCols =  mat1.getNumCols() + mat2.getNumCols();
    matOut = std::move(Matrix<T, orderOutput>{numRows, numCols});
    // pArr = new T[numRows * numCols];
    for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
    {
        int rowIdx1 = rowIdx / mat2.getNumRows();
        int rowIdx2 = rowIdx % mat2.getNumRows();
        for (int colIdx = 0; colIdx < mat1.getNumCols(); colIdx++)
        {
            // int64_t pos = getPosId<orderOutput>(rowIdx, colIdx, numRows,  numCols);
            matOut(rowIdx, colIdx) =  mat1(rowIdx1, colIdx);
        }
        for (int colIdx = mat1.getNumCols(); colIdx < numCols; colIdx++)
        {
            // int64_t pos = getPosId<orderOutput>(rowIdx, colIdx, numRows, numCols);
            matOut(rowIdx, colIdx) =  mat2(rowIdx2, colIdx - mat1.getNumCols());
        }
    }
}



template<typename T, MajorOrder majorOrder>
void computeHeadsAndTails(T* d_mat, int numRows, int numCols, int colIdx) {
    T dataHeads;
    int headRowIdx = 0;
    int64_t posIdx;

    if (colIdx < numCols)
    {
        dataHeads = d_mat[getPosId<majorOrder>(headRowIdx, colIdx, numRows, numCols)];
    }
    for (int rowIdx = headRowIdx + 1; rowIdx < numRows; rowIdx++)
    {
        T i = rowIdx - headRowIdx + 1;
        if (colIdx < numCols)
        {
            T prevRowSum;
            T tailVal;
            prevRowSum = dataHeads;
            T matVal = d_mat[getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols)];
            dataHeads += matVal;
            tailVal = (matVal * (i - 1) - prevRowSum) / sqrt(i * (i - 1));
            d_mat[getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols)] = tailVal;
        }
    }
    if (colIdx < numCols)
    {
        d_mat[getPosId<majorOrder>(headRowIdx, colIdx, numRows, numCols)] = dataHeads / sqrt(numRows);
    }
}

template <typename T, MajorOrder majorOrder>
void concatenateHeadsAndTails(T* d_mat, T* d_mat2Mod, T* dOutMat, int numRows1, int numCols1, int numRows2, int numCols2, int colIdx) {
    int headRowIdx = 0;
    const int numRowsOut = numRows1 + numRows2 - 1;
    const int numColsOut = numCols1 + numCols2;

    for (int rowIdx = 0; rowIdx < numRows1; rowIdx++)
    {
        if (colIdx < numCols1)
        {
            int posIdx = getPosId<majorOrder>(rowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = d_mat[getPosId<majorOrder>(rowIdx, colIdx, numRows1, numCols1)] * sqrt(numRows2);
        }
        if (colIdx < numCols2)
        {
            int posIdx2 = getPosId<majorOrder>(rowIdx, colIdx + numCols1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = d_mat2Mod[getPosId<majorOrder>(headRowIdx, colIdx, numRows2, numCols2)];
            // printf("HERE 1 %d %d %.3f %d\n", rowIdx, colIdx + numCols1, dOutMat[posIdx2], posIdx2);
        }
    }
    for (int rowIdx = numRows1; rowIdx < numRowsOut; rowIdx++)
    {
        if (colIdx < numCols1)
        {
            int posIdx = getPosId<majorOrder>(rowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = 0;
            // printf("HERE 2 %d %d %.3f %d \n", rowIdx, colIdx, dOutMat[posIdx], posIdx);
        }
        if (colIdx < numCols2)
        {
            int posIdx2 = getPosId<majorOrder>(rowIdx, colIdx + numCols1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = d_mat2Mod[getPosId<majorOrder>(rowIdx - numRows1 + 1, colIdx, numRows2, numCols2)] * sqrt(numRows1);
            // printf("HERE 2 %d %d %.3f %d\n", rowIdx, colIdx + numCols1, dOutMat[posIdx2], posIdx2);
        }
    }
}

// template <typename T>
// __global__ void setZerosUpperTriangular(T* d_A, int numRows, int numCols)
// {
// 	int colIdx = threadIdx.x;
// 	for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
// 	{
// 		if (rowIdx > colIdx)
// 		{
// 			d_A[IDX_C(rowIdx, colIdx, numRows, numCols)] = 0;
// 		}
// 	}
// }

template <typename T, MajorOrder computeOrder>
int computeFigaro(const Matrix<T, computeOrder>& mat1, const Matrix<T, computeOrder>& mat2,
    T*& h_matR, const std::string& fileName, int compute)
{
    int numRows1 = mat1.getNumRows();
    int numRows2 = mat2.getNumRows();
    int numCols1 = mat1.getNumCols();
    int numCols2 = mat2.getNumCols();
    int numRowsOut = numRows1 + numRows2 - 1;
    int numColsOut = numCols1 + numCols2;
    bool computeSVD = compute == 2;
    auto start = std::chrono::high_resolution_clock::now();
    // T * h_matOut = new T[numRowsOut * numColsOut];
    Matrix<T, computeOrder> matOut{numRowsOut, numColsOut};
    Matrix<T, computeOrder> mat1Copy{numRows1, numCols1};
    Matrix<T, computeOrder> mat2Copy{numRows2, numCols2};

    copyMatrix<T, computeOrder>(mat1.getDataC(), mat1Copy.getData(), numRows1, numCols1, numRows1, numCols1, false);
    copyMatrix<T, computeOrder>(mat2.getDataC(), mat2Copy.getData(), numRows2, numCols2, numRows2, numCols2, false);
    // printMatrix<T, computeOrder>(mat1Copy.getData(), numRows1, numCols1, numRows1, fileName + "LinScaleCOPY1", false);
    // printMatrix<T, computeOrder>(mat2Copy.getData(), numRows2, numCols2, numRows2, fileName + "LinScaleCOPY2", false);

    // Compute join offsets for both tables
    // compute join offsets
    // for loop call for each subset the
    omp_set_num_threads(omp_get_max_threads());
    std::cout << mat1.getDataC() << " " << mat2.getDataC() << std::endl;
    #pragma omp parallel for schedule(static)
    for (int colIdx = 0; colIdx < numCols2; colIdx++)
    {
        computeHeadsAndTails<T, computeOrder>(mat2Copy.getData(), numRows2, numCols2, colIdx);
    }
    #pragma omp parallel for schedule(static)
    for (int colIdx = 0; colIdx < std::max(numCols1, numCols2); colIdx++)
    {
        concatenateHeadsAndTails<T, computeOrder>(mat1Copy.getData(), mat2Copy.getData(), matOut.getData(), numRows1, numCols1, numRows2, numCols2, colIdx);
    }

    int rank = std::min(numRowsOut, numColsOut);

    // // Compute QR factorization
    if constexpr (std::is_same<T, float>::value)
    {
        // CUSOLVER_CALL(cusolverDnSgeqrf(cusolverH, numRowsOut, numColsOut, h_matOutTran, numRowsOut, d_tau, d_work, workspace_size, devInfo));
    }
    else
    {
        T *tau = new T[numColsOut];  // Stores Householder reflector coefficients
        int info;
        if (computeOrder == MajorOrder::ROW_MAJOR)
        {
            info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, numRowsOut, numColsOut, matOut.getData(), numColsOut, tau);
        }
        else
        {
            info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, numRowsOut, numColsOut, matOut.getData(), numRowsOut, tau);
        }
        if (info != 0)
        {
            std::cerr << "QR decomposition failed!" << std::endl;
            return -1;
        }
        // if (computeSVD)
        // {
        //     setZerosUpperTriangular<<<1, numColsOut>>>(d_matOutTran, numRowsOut, numColsOut);
        //     char jobu = 'N';  // No computation of U
        //     char jobvt = 'N'; // No computation of V^T
        //     // cuSOLVER handle
        //     int *d_info;
        //     double *d_work;
        //     int lwork = 0;
        //     int ldA = numRowsOut;

        //     cusolverDnHandle_t cusolverH1 = nullptr;
        //     CUSOLVER_CALL(cusolverDnCreate(&cusolverH1));
        //     CUDA_CALL(cudaMalloc((void**)&d_info, sizeof(int)));
        //     CUSOLVER_CALL(cusolverDnDgesvd_bufferSize(cusolverH, rank, numColsOut, &lwork));
        //     CUDA_CALL(cudaMalloc((void**)&d_work, sizeof(double) * lwork));
        //         CUDA_CALL(cudaMalloc((void**)&d_S, sizeof(double) * rank));
        //     cusolverDnDgesvd(cusolverH1, jobu, jobvt, numColsOut, numColsOut, d_matOutTran, ldA, d_S, nullptr, numColsOut, nullptr, numColsOut,
        //                     d_work, lwork, nullptr, d_info);
        // }
    }

    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    copyMatrix<double, computeOrder>(matOut.getData(), h_matR, numRowsOut, numColsOut, numColsOut, numColsOut, true);
    std::cout << "\n";
    if (computeSVD)
    {
	    std::cout << "SVD decomposition ";
    }
    else
    {
        printMatrix<T, computeOrder>(h_matR, numColsOut, numColsOut, numColsOut, fileName + "LinScale", false);
	    std::cout << "QR decomposition ";
    }
    std::cout << "Linscale took " << elapsed << " seconds.\n";

    return 0;
}

template <typename T, MajorOrder majorOrder>
int computeGeneral(T* h_A, T*& h_R, int numRows, int numCols, const std::string& fileName, int compute)
{
    T *tau = new T[numCols];  // Stores Householder reflector coefficients
    int info;
    bool computeSVD = compute == 2;
    // h_R = new T[numRows * numCols];
    copyMatrix<double, majorOrder>(h_A, h_R, numRows, numCols, numRows, numCols, false);

    // Perform QR decomposition: A -> R (upper part), reflectors in lower part
    auto start = std::chrono::high_resolution_clock::now();
    if (majorOrder == MajorOrder::ROW_MAJOR)
    {
        info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, numRows, numCols, h_R, numCols, tau);
    }
    else
    {
        info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, numRows, numCols, h_R, numRows, tau);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    if (info != 0)
    {
        std::cerr << "QR decomposition failed!" << std::endl;
        return -1;
    }
    // Compute elapsed time
    // printMatrix<T, majorOrder>(h_R, numRows, numCols, numCols, fileName + "MKL", true);


    // Print execution time
    std::string nameDecomp = computeSVD ? "SVD" : "QR";
    std::cout << "\n" + nameDecomp + " decomposition MKL took " << elapsed << " seconds.\n";
    delete [] tau;
    return 0;
}

template<typename T, MajorOrder majorOrder>
void computeMatrixVector(T* pMat, T* pVect, T*& pOutVect, int numRows, int numCols,
    bool transpose = false)
{
    T alpha = 1.0;
    T beta = 0.0;
    CBLAS_TRANSPOSE aTran = (transpose ?  CblasTrans : CblasNoTrans);

    int cntOut = (transpose ? numCols : numRows);
    pOutVect = new T[cntOut];

    if constexpr (MajorOrder::ROW_MAJOR == majorOrder)
    {
        cblas_dgemv(CblasRowMajor, aTran, numRows, numCols, alpha, pMat, numCols, pVect, 1, beta, pOutVect, 1);
    }
    else
    {
        cblas_dgemv(CblasColMajor, aTran, numRows, numCols, alpha, pMat, numRows, pVect, 1, beta, pOutVect, 1);
    }
}


template<typename T, MajorOrder majorOrder>
void computeMatrixMatrix(T* pMat1, T* pMat2, T*& pOutMat, int numRows1, int numCols1,
    int numCols2, bool transpose = false)
{
    double alpha = 1.0, beta = 0.0;
    CBLAS_TRANSPOSE aTran = transpose ?  CblasTrans : CblasNoTrans;
    int cntOut = transpose ? numCols1 * numCols2 : numRows1 * numCols2;

    pOutMat = new T[cntOut];
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        cblas_dgemm(CblasRowMajor, aTran, CblasNoTrans,
                    numRows1, numCols1, numCols2, alpha, pMat1, numCols1, pMat2, numCols2, beta, pOutMat, numCols2);
    }
    else
    {
        cblas_dgemm(CblasColMajor, aTran, CblasNoTrans,
                    numRows1, numCols1, numCols2, alpha, pMat1, numRows1, pMat2, numCols1, beta, pOutMat, numRows1);
    }
}

template<typename T, MajorOrder majorOrder>
void selfTransposeMatrixMultiplication(T* pMat, T*& pOutMat, int numRows, int numCols)
{
    pOutMat = new T[numCols * numCols];
    double alpha = 1.0, beta = 0.0;
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    numCols, numCols, numRows, alpha, pMat, numCols, pMat, numCols, beta,
                     pOutMat, numCols);
    }
    else
    {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            numCols, numCols, numRows, alpha, pMat, numRows, pMat, numRows, beta,
                     pOutMat, numCols);
    }
}

template<typename T, MajorOrder majorOrder>
void computeInverse(T* pMat, int numRows, int numCols)
{
    int N = 3;  // Matrix size
    int LDA = 3, info;
    int *ipiv = new int [numCols];  // Pivot indices
    if constexpr (MajorOrder::ROW_MAJOR == majorOrder)
    {
        // Step 1: Perform LU decomposition
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, numRows, numRows, pMat, numCols, ipiv);

        // Step 2: Compute inverse using LU factorization
        LAPACKE_dgetri(LAPACK_ROW_MAJOR, numRows, pMat, numCols, ipiv);
    }
    else
    {
          // Step 1: Perform LU decomposition
        LAPACKE_dgetrf(LAPACK_COL_MAJOR, numRows, numRows, pMat, numCols, ipiv);

        // Step 2: Compute inverse using LU factorization
        LAPACKE_dgetri(LAPACK_COL_MAJOR, numRows, pMat, numCols, ipiv);
    }

    delete [] ipiv;
}

// Ax = b --- pMatA * pOutVect = pVectB,
// = A^T * A * x = A^T * b
// x = (A^T * A)^ inv * A^T * b
// A^T * A = R^T * R
template<typename T, MajorOrder majorOrder, MajorOrder rMajorOrder>
void solveLLS(T* pMatA, T* pMatR, T* pVectB, T*& pOutVect, int  numRows, int numCols, const std::string& fileName)
{
    T* pOutMat;
    T* pTempVect;

    selfTransposeMatrixMultiplication<double, rMajorOrder>(pMatR, pOutMat, numCols, numCols);
    // printMatrix<double, MajorOrder::COL_MAJOR>(pOutMat, numCols, numCols, numCols, fileName + "STMM.csv", false);

    computeInverse<double, rMajorOrder>(pOutMat, numCols, numCols);
    // printMatrix<double, MajorOrder::COL_MAJOR>(pOutMat, numCols, numCols, numCols, fileName +"STMMINV.csv", false);

    // printMatrix<double, MajorOrder::COL_MAJOR>(pMatA, numRows, numCols, numRows, fileName +"matAOrig.csv", false);
    //  printMatrix<double, MajorOrder::COL_MAJOR>(pVectB, numRows, 1, numRows, "matBOrig.csv", false);
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(pMatA, pVectB, pTempVect, numRows, numCols, true);
    // printMatrix<double, MajorOrder::COL_MAJOR>(pTempVect, numCols, 1, numCols, "ATv.csv", false);

    computeMatrixVector<double, MajorOrder::COL_MAJOR>(pOutMat, pTempVect, pOutVect, numCols, numCols, false);
    // printMatrix<double, MajorOrder::COL_MAJOR>(pOutVect, numCols, 1, numCols, fileName + "x_prod_sol.csv",
        // false);
    delete [] pTempVect;
    delete [] pOutMat;
}

template <typename T>
double computeMeanSquaredError(T* pA, T* pB, int numRows)
{
    double* diff = new double[numRows];
    double* squared = new double[numRows];
      // Compute element-wise difference: diff = a - b
    vdSub(numRows, pA, pB, diff);

    // Square each element: squared = diff^2
    vdMul(numRows, diff, diff, squared);

    // Compute sum of squared differences
    double sum_sq = cblas_dasum(numRows, squared, 1);

    // Compute MSE
    double mse = sum_sq / numRows;
    delete [] diff;
    delete [] squared;

    return mse;
}


// void evaluateTrainUpdate(MatrixDRow& mat1, MatrixDRow& mat2, MatrixDRow& vectX,
//     MatrixDCol& matMKLRUpdate, MatrixDCol& matFigRUpdate,
//     const std::string& fileName,
//     double*& pVectXMKL, double*& pVectXFig, int compute)
// {
//     double *h_pCartProdTrain, *h_pCartProdTest;
//     double *h_vectXCompMKL, *h_vectXCompFig, *pOutVectBTrain, *pOutVectBTest;
//     double *h_MKLROut;
//     int numRowsUpdate = 100;
//     auto mat1Update = generateRandom<double>(numRowsUpdate, mat1.getNumCols(), 11);
//     MatrixDRow mat1Out{1, 1};
//     concatenateMatrices(mat1, mat1Update, mat1Out, true);
//     generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::COL_MAJOR>(mat1Update, mat2, h_pCartProdTrain);
//     computeMatrixVector<double, MajorOrder::COL_MAJOR>(h_pCartProdTrain, vectX.getData(), pOutVectBTrain,
//         mat1Update.getNumRows() * mat2.getNumRows(),
//         mat1Update.getNumCols() + mat2.getNumCols(), false);

//     MatrixDCol matMKLROut{mat1Update.getNumRows() * mat2.getNumRows(),
//         mat1Update.getNumCols() + mat2.getNumCols()};
//     MatrixDCol matMKLR{matMKLROut.getNumCols(), matMKLROut.getNumCols()};



//     // /*********** TRAINING ***********************/
//     computeGeneral<double, MajorOrder::COL_MAJOR>(h_pCartProdTrain, matMKLROut.getData(),
//         matMKLROut.getNumRows(), matMKLROut.getNumCols(), fileName, compute);
//     copyMatrix<double, MajorOrder::COL_MAJOR>(matMKLROut.getData(), matMKLR.getData(),
//         matMKLROut.getNumRows(), matMKLROut.getNumCols(),
//         matMKLROut.getNumCols(), matMKLROut.getNumCols(), true);
//     MatrixDCol matRMKLConc {1, 1};
//     concatenateMatrices(matMKLRUpdate, matMKLR, matRMKLConc, true);


//     matMKLROut = MatrixDCol{matMKLRUpdate.getNumRows() ,matMKLRUpdate.getNumCols() };
//     matMKLR = MatrixDCol{matMKLRUpdate.getNumCols() ,matMKLRUpdate.getNumCols() };
//     computeGeneral<double, MajorOrder::COL_MAJOR>(matRMKLConc.getData(), matMKLROut.getData(),
//         matMKLROut.getNumRows(), matMKLROut.getNumCols(), fileName, compute);
//     copyMatrix<double, MajorOrder::COL_MAJOR>(matMKLROut.getData(), matMKLR.getData(),
//         matMKLROut.getNumRows(), matMKLROut.getNumCols(),
//         matMKLROut.getNumCols(), matMKLROut.getNumCols(), true);
//     matMKLRUpdate = std::move(matMKLR);
//     // printMatrix<double, MajorOrder::COL_MAJOR>(h_MKLR, numCols1 + numCols2, numCols1 + numCols2, numCols1 + numCols2, "R.csv", false);

//     solveLLS<double, MajorOrder::COL_MAJOR, MajorOrder::COL_MAJOR>(h_pCartProdTrain, matMKLRUpdate.getData(),     pOutVectBTrain, pVectXMKL, mat1Update.getNumRows() * mat2.getNumRows(),
//         mat1Update.getNumCols() + mat2.getNumCols(), "results/MKLInc");

//     MatrixDCol matFigaroR{mat1Update.getNumRows() + mat2.getNumRows() - 1,
//         mat1Update.getNumCols() + mat2.getNumCols()};
//     computeFigaro<double, MajorOrder::ROW_MAJOR>(mat1Update, mat2,
//         matFigaroR.getData(), fileName, compute);
//     MatrixDCol matRFigConc {1, 1};
//     concatenateMatrices(matFigRUpdate, matFigaroR, matRFigConc, true);

//     MatrixDCol matFigROut{matFigRUpdate.getNumRows(),
//         matFigRUpdate.getNumCols()};
//     MatrixDCol matFigR{matFigROut.getNumCols(), matFigROut.getNumCols()};
//     computeGeneral<double, MajorOrder::ROW_MAJOR>(matRFigConc.getData(), matFigROut.getData(),
//         matFigROut.getNumRows(), matFigROut.getNumCols(), fileName, compute);
//     copyMatrix<double, MajorOrder::ROW_MAJOR>(matFigROut.getData(), matFigR.getData(),
//         matFigROut.getNumRows(), matFigROut.getNumCols(),
//         matFigROut.getNumCols(), matFigROut.getNumCols(), true);
//     matFigRUpdate = std::move(matFigR);

//     solveLLS<double, MajorOrder::COL_MAJOR, MajorOrder::ROW_MAJOR>(h_pCartProdTrain, matFigRUpdate.getData(), pOutVectBTrain, pVectXFig, mat1Update.getNumRows() * mat2.getNumRows(),
//         mat1Update.getNumCols() + mat2.getNumCols(), "results/LinScale");


// }


// void evaluateRowUpdates(int numRows1, int numCols1, int numRows2, int numCols2, std::string& fileName, int compute)
// {
//     /*********** DATA GENERATION ***********************/
//     auto mat1 = generateRandom<double>(numRows1, numCols1, 0);
//     auto mat2 = generateRandom<double>(numRows2, numCols2, 10);
//     auto vectX = generateRandom<double>(1, numCols1 + numCols2, 15);
//     const int numUpdatesTotal = 10;
//     MatrixDCol matMKLR{1, 1};
//     MatrixDCol matFigR{1, 1};
//     double* pVectXMKL;
//     double* pVectXFig;
//     for (int numUpdates = 0; numUpdates < numUpdatesTotal; numUpdates++)
//     {
//         evaluateTrainUpdate(mat1, mat2, vectX, matMKLR, matFigR, fileName, pVectXMKL, pVectXFig, compute);
//         /*********** TESTING ***********************/
//         // auto mat1Test = generateRandom<double>(numRows1, numCols1, 0);
//         // auto mat2Test = generateRandom<double>(numRows2, numCols2, 10);
//         // generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::COL_MAJOR>(mat1, mat2, h_pCartProdTrain);
//         // computeMatrixVector<double, MajorOrder::COL_MAJOR>(h_pCartProdTest, vectX.getData(), pOutVectBTest, numRows1 * numRows2, numCols1 + numCols2, false);

//         // double* pOutVectBTestCompMKL, *pOutVectBTestCompFig;

//         // computeMatrixVector<double, MajorOrder::COL_MAJOR>(h_pCartProdTest, h_vectXCompMKL, pOutVectBTestCompMKL, numRows1 * numRows2, numCols1 + numCols2, false);
//         // computeMatrixVector<double, MajorOrder::COL_MAJOR>(h_pCartProdTest, h_vectXCompFig, pOutVectBTestCompFig, numRows1 * numRows2, numCols1 + numCols2, false);

//         // double mklError = computeMeanSquaredError(pOutVectBTestCompMKL, pOutVectBTest, numRows1 * numRows2);
//         // double figError = computeMeanSquaredError(pOutVectBTestCompFig, pOutVectBTest, numRows1 * numRows2);
//         // std::cout << "MKL MSE " << mklError << std::endl;
//         // std::cout << "Figaro MSE " << figError << std::endl;
//     }
// }


void evaluateTrain(const MatrixDRow& mat1, const MatrixDRow& mat2,
    double* h_pCartProdTrain, MatrixDCol& matMKLR, MatrixDRow& matFigR,
    const std::string& fileName, int compute)
{
    MatrixDCol matMKLROut{mat1.getNumRows() * mat2.getNumRows(), mat1.getNumCols() + mat2.getNumCols()};
    matMKLR = MatrixDCol{mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols()};
    /*********** TRAINING ***********************/
    computeGeneral<double, MajorOrder::COL_MAJOR>(h_pCartProdTrain, matMKLROut.getData(),
        mat1.getNumRows() * mat2.getNumRows(), mat1.getNumCols() + mat2.getNumCols(), fileName, compute);
    copyMatrix<double, MajorOrder::COL_MAJOR>(matMKLROut.getData(), matMKLR.getData(),
        mat1.getNumRows() * mat2.getNumRows(), mat1.getNumCols() + mat2.getNumCols(),
        mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), true);
    // printMatrix<double, MajorOrder::COL_MAJOR>(matMKLR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), "RRR.csv", false);

    // Column orientation because of the current implementation of Figaro for faster processing
    matFigR = MatrixDRow{mat1.getNumRows() + mat2.getNumRows() - 1, mat1.getNumCols() + mat2.getNumCols()};
    computeFigaro<double, MajorOrder::ROW_MAJOR>(mat1, mat2, matFigR.getData(), fileName, compute);
}


void evaluate(int numRows1, int numCols1, int numRows2, int numCols2, std::string& fileName, int compute)
{
    double *h_vectXCompMKL, *h_vectXCompFig, *pOutVectBTrain, *pOutVectBTest;
    double *h_MKLROut, *h_MKLR, *h_FigR;

    /*********** DATA GENERATION ***********************/
    auto mat1 = generateRandom<double>(numRows1, numCols1, 0);
    auto mat2 = generateRandom<double>(numRows2, numCols2, 10);
    auto vectX = generateRandom<double>(1, numCols1 + numCols2, 15);

    // printMatrix<double, MajorOrder::ROW_MAJOR>(mat1.getData(), numRows1, numCols1, numRows1, "A.csv", false);
    // printMatrix<double, MajorOrder::ROW_MAJOR>(mat2.getData(), numRows2, numCols2, numRows2, "B.csv", false);
    // printMatrix<double, MajorOrder::COL_MAJOR>(vectX.getData(), numCols1 + numCols2, 1, numCols1 + numCols2, "x_vect.csv", false);
    MatrixDCol matCartProd{1, 1};
    generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::COL_MAJOR>(mat1, mat2, matCartProd);
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProd.getData(), vectX.getData(), pOutVectBTrain, numRows1 * numRows2, numCols1 + numCols2, false);

    // printMatrix<double, MajorOrder::COL_MAJOR>(pOutVectBTrain, numRows1 * numRows2, 1, numRows1 * numRows2, "matProd.csv", false);
    // printMatrix<double, MajorOrder::COL_MAJOR>(h_pCartProdTrain, numRows1 * numRows2, numCols1 + numCols2, numRows1 * numRows2, "mat.csv", false);
    MatrixDCol matMKLR{1, 1};
    MatrixDRow matFigR{1, 1};
    evaluateTrain(mat1, mat2, matCartProd.getData(), matMKLR, matFigR, fileName, compute);
    // printMatrix<double, MajorOrder::COL_MAJOR>(matMKLR.getData(), numCols1 + numCols2, numCols1 + numCols2, numCols1 + numCols2, "MKLR.csv", true);
    printMatrix<double, MajorOrder::ROW_MAJOR>(matFigR.getData(), numCols1 + numCols2, numCols1 + numCols2, numCols1 + numCols2, "FIGR.csv", false);
    solveLLS<double, MajorOrder::COL_MAJOR, MajorOrder::COL_MAJOR>(matCartProd.getData(), matMKLR.getData(), pOutVectBTrain, h_vectXCompMKL, numRows1 * numRows2, numCols1 + numCols2, "results/MKL");
    solveLLS<double, MajorOrder::COL_MAJOR, MajorOrder::ROW_MAJOR>(matCartProd.getData(), matFigR.getData(), pOutVectBTrain, h_vectXCompFig, numRows1 * numRows2, numCols1 + numCols2, "results/LinScale");

    /*********** TESTING ***********************/
    auto mat1Test = generateRandom<double>(numRows1, numCols1, 0);
    auto mat2Test = generateRandom<double>(numRows2, numCols2, 10);
    MatrixDCol matCartProdTest{1, 1};
    generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::COL_MAJOR>(
            mat1Test, mat2Test,  matCartProdTest);
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProdTest.getData(), vectX.getData(), pOutVectBTest, numRows1 * numRows2, numCols1 + numCols2, false);

    double* pOutVectBTestCompMKL, *pOutVectBTestCompFig;
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProdTest.getData(), h_vectXCompMKL, pOutVectBTestCompMKL, numRows1 * numRows2, numCols1 + numCols2, false);
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProdTest.getData(), h_vectXCompFig, pOutVectBTestCompFig, numRows1 * numRows2, numCols1 + numCols2, false);
    double mklError = computeMeanSquaredError(pOutVectBTestCompMKL, pOutVectBTest, numRows1 * numRows2);
    double figError = computeMeanSquaredError(pOutVectBTestCompFig, pOutVectBTest, numRows1 * numRows2);
    std::cout << "MKL MSE " << mklError << std::endl;
    std::cout << "Figaro MSE " << figError << std::endl;
}

int main(int argc, char* argv[])
{
    int numRows1 = 1000, numCols1 = 24;
    int numRows2 = 1000, numCols2 = 24;
    int compute = 1;
    try {
        // Define the command-line options
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Show help message")
            ("input,i", po::value<std::string>(), "Input file")
            ("m1", po::value<int>(), "Number of rows 1")
            ("m2", po::value<int>(), "Number of rows 2")
            ("n1", po::value<int>(), "Number of columns 1")
            ("n2", po::value<int>(), "Number of columns 2")
            ("compute", po::value<int>(), "Compute mode")
            ("verbose,v", "Enable verbose mode");

        // Parse the command-line arguments
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        // Handle the help flag
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        if (vm.count("m1"))
        {
            numRows1 = vm["m1"].as<int>();
        }
        if (vm.count("m2"))
        {
            numRows2 = vm["m2"].as<int>();
        }
        if (vm.count("n1"))
        {
            numCols1 = vm["n1"].as<int>();
        }
        if (vm.count("n2"))
        {
            numCols2 = vm["n2"].as<int>();
        }
	if (vm.count("compute"))
	{
		compute = vm["compute"].as<int>();
	}
        std::string fileName = "results/" + std::to_string(numRows1) + "x" + std::to_string(numCols1) + "," + std::to_string(numRows2) + "x" + std::to_string(numCols2);
        evaluate(numRows1, numCols1, numRows2, numCols2, fileName, compute);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}

