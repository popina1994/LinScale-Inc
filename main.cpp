#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <boost/program_options.hpp>
#include <mkl_lapacke.h>
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

// column major version
template <typename T>
void generateRandom(T*& pArr, int numRows, int numCols, int offset)
{
    std::mt19937 gen(offset); // Fixed seed
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    pArr = new T [numRows * numCols];
    // col_major
    for (int colIdx = 0; colIdx < numCols; colIdx++)
    {
        for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
        {
            int pos = IDX_R(rowIdx, colIdx, numRows, numCols);
            pArr[pos] = dist(gen);
        }
    }
}

template<typename T, MajorOrder orderOutput>
void generateCartesianProduct(T* pArr1, T* pArr2, int numRows1, int numCols1, int numRows2, int numCols2, T*& pArr)
{
    int numRows = numRows1 * numRows2;
    int numCols =  numCols1 + numCols2;
    pArr = new T[numRows * numCols];
    for (int rowIdx = 0; rowIdx < numRows1 * numRows2; rowIdx++)
    {
        int rowIdx1 = rowIdx / numRows2;
        int rowIdx2 = rowIdx % numRows2;
        for (int colIdx = 0; colIdx < numCols1; colIdx++)
        {
            int pos;
            if constexpr (orderOutput == MajorOrder::ROW_MAJOR)
            {
                pos = IDX_R(rowIdx, colIdx, numRows, numCols);
            }
            else
            {
                pos = IDX_C(rowIdx, colIdx, numRows, numCols);
            }
            pArr[pos] =  pArr1[IDX_R(rowIdx1, colIdx, numRows1, numCols1)];
        }
        for (int colIdx = numCols1; colIdx < numCols; colIdx++)
        {
            int pos;
            if constexpr (orderOutput == MajorOrder::ROW_MAJOR)
            {
                pos = IDX_R(rowIdx, colIdx, numRows, numCols);
            }
            else
            {
                pos = IDX_C(rowIdx, colIdx, numRows, numCols);
            }
            pArr[pos] =  pArr2[IDX_R(rowIdx2, colIdx - numCols1, numRows2, numCols2)];
        }
    }
}

template<typename T>
void computeHeadsAndTails(T* d_mat, int numRows, int numCols, int colIdx) {
    T dataHeads[1024];
    int headRowIdx = 0;

    if (colIdx < numCols)
    {
        dataHeads[colIdx] = d_mat[IDX_R(headRowIdx, colIdx, numRows, numCols)];
    }
    // __syncthreads();
    for (int rowIdx = headRowIdx + 1; rowIdx < numRows; rowIdx++)
    {
        T i = rowIdx - headRowIdx + 1;
        if (colIdx < numCols)
        {
            T prevRowSum;
            T tailVal;
            prevRowSum = dataHeads[colIdx];
            T matVal = d_mat[IDX_R(rowIdx, colIdx, numRows, numCols)];
            dataHeads[colIdx] += matVal;
            tailVal = (matVal * (i - 1) - prevRowSum) / sqrtf(i * (i - 1));
            d_mat[IDX_R(rowIdx, colIdx, numRows, numCols)] = tailVal;
            // printf("TAIL VAL %d %d %.3f %.3f\n", rowIdx, colIdx, i, tailVal);
        }
        // __syncthreads();
    }
    if (colIdx < numCols)
    {
        d_mat[IDX_R(headRowIdx, colIdx, numRows, numCols)] = dataHeads[colIdx] / sqrtf(numRows);
        // printf("HT: %.3f\n", dataHeads[colIdx] / sqrtf(numRows));
    }
}

template <typename T>
void concatenateHeadsAndTails(T* d_mat, T* d_mat2Mod, T* dOutMat, int numRows1, int numCols1, int numRows2, int numCols2, int colIdx) {
    int headRowIdx = 0;
    const int numRowsOut = numRows1 + numRows2 - 1;
    const int numColsOut = numCols1 + numCols2;

    for (int rowIdx = 0; rowIdx < numRows1; rowIdx++)
    {
        if (colIdx < numCols1)
        {
            int posIdx = IDX_R(rowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = d_mat[IDX_R(rowIdx, colIdx, numRows1, numCols1)] * sqrtf(numRows2);
            // printf("HERE 1 %d %d %.3f %d\n", rowIdx, colIdx, dOutMat[posIdx], posIdx);
        }
        if (colIdx < numCols2)
        {
            int posIdx2 = IDX_R(rowIdx, colIdx + numCols1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = d_mat2Mod[IDX_R(headRowIdx, colIdx, numRows2, numCols2)];
            // printf("HERE 1 %d %d %.3f %d\n", rowIdx, colIdx + numCols1, dOutMat[posIdx2], posIdx2);
        }
    }
    for (int rowIdx = numRows1; rowIdx < numRowsOut; rowIdx++)
    {
        if (colIdx < numCols1)
        {
            int posIdx = IDX_R(rowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = 0;
            // printf("HERE 2 %d %d %.3f %d \n", rowIdx, colIdx, dOutMat[posIdx], posIdx);
        }
        if (colIdx < numCols2)
        {
            int posIdx2 = IDX_R(rowIdx, colIdx + numCols1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = d_mat2Mod[IDX_R(rowIdx - numRows1 + 1, colIdx, numRows2, numCols2)] * sqrtf(numRows1);
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

template <typename T>
int computeFigaro(T* h_mat1, T* h_mat2, int numRows1, int numCols1, int numRows2, int numCols2,
    std::string& fileName, int compute)
{
    int numRowsOut = numRows1 + numRows2 - 1;
    int numColsOut = numCols1 + numCols2;
    bool computeSVD = compute == 2;

    auto start = std::chrono::high_resolution_clock::now();
    T * h_matOut = new T[numRowsOut * numColsOut];

    // Compute join offsets for both tables
    // compute join offsets
    // for loop call for each subset the
    for (int colIdx = 0; colIdx < numCols2; colIdx++)
    {
        computeHeadsAndTails(h_mat2, numRows2, numCols2, colIdx);
    }
    for (int colIdx = 0; colIdx < std::max(numCols1, numCols2); colIdx++)
    {
        concatenateHeadsAndTails(h_mat1, h_mat2, h_matOut, numRows1, numCols1, numRows2, numCols2, colIdx);
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

        info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, numRowsOut, numColsOut, h_matOut, numColsOut, tau);
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

    std::cout << "\n";
    if (computeSVD)
    {
	    std::cout << "SVD decomposition ";
    }
    else
    {
        printMatrix<T, MajorOrder::ROW_MAJOR>(h_matOut, numRowsOut, numColsOut, numColsOut, fileName + "LinScale", true);
	    std::cout << "QR decomposition ";
    }
    std::cout << "Linscale took " << elapsed << " seconds.\n";

    return 0;
}

template <typename T, MajorOrder majorOrder>
int computeGeneral(T* h_A, int numRows, int numCols, const std::string& fileName, int compute)
{

    T *tau = new T[numCols];  // Stores Householder reflector coefficients
    int info;
    bool computeSVD = compute == 2;

    // Perform QR decomposition: A -> R (upper part), reflectors in lower part
    auto start = std::chrono::high_resolution_clock::now();
    info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, numRows, numCols, h_A, numCols, tau);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    if (info != 0)
    {
        std::cerr << "QR decomposition failed!" << std::endl;
        return -1;
    }
    // Compute elapsed time
    printMatrix<T, MajorOrder::ROW_MAJOR>(h_A, numRows, numCols, numCols, fileName + "MKL", true);


    // Print execution time
    std::string nameDecomp = computeSVD ? "SVD" : "QR";
    std::cout << "\n" + nameDecomp + " decomposition MKL took " << elapsed << " seconds.\n";

    return 0;
}

void evaluate(int numRows1, int numCols1, int numRows2, int numCols2, std::string& fileName, int compute)
{
    double *h_mat1, *h_mat2, *pArr;
    generateRandom(h_mat1, numRows1, numCols1, 0);
    generateRandom(h_mat2, numRows2, numCols2, 10);
    printMatrix<double, MajorOrder::ROW_MAJOR>(h_mat1, numRows1, numCols1, numRows1, "A.csv", false);
    printMatrix<double, MajorOrder::ROW_MAJOR>(h_mat2, numRows2, numCols2, numRows2, "B.csv", false);

    generateCartesianProduct<double, MajorOrder::ROW_MAJOR>(h_mat1, h_mat2, numRows1, numCols1, numRows2, numCols2, pArr);
    //printMatrix<double, MajorOrder::ROW_MAJOR>(pArr, numRows1 * numRows2, numCols1 + numCols2, numRows1 * numRows2, "mat.csv", false);

    computeGeneral<double, MajorOrder::ROW_MAJOR>(pArr, numRows1 * numRows2, numCols1 + numCols2, fileName, compute);
    computeFigaro<double>(h_mat1, h_mat2, numRows1, numCols1, numRows2, numCols2, fileName, compute);
}

int main(int argc, char* argv[])
{
    int numRows1 = 1000, numCols1 = 4;
    int numRows2 = 2, numCols2 = 4;
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

