// matmul_compare.cu
#include "test.h"
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <vector>
#include <cstdlib>
#include <cudnn.h>

// nvcc -o test main.cu -lcudnn -lcublas

// 100轮测试
int num_runs = 100;

const int num_threads_large = 1024;

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if(status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(call)                                               \
{                                                                       \
    cudnnStatus_t status = (call);                                      \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
        std::cerr << "cuDNN error in " << __FILE__ << ":" << __LINE__   \
                  << " - " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

// 创建 cuBLAS 句柄
cublasHandle_t handle;
cudnnHandle_t cudnn;

void test_matmul();
void test_add();
void test_multiply();
void test_softmax();
void test_silu();
void test_rope();
void test_embedding();
void test_rmsnorm();
void test_attention();


int main() {
    // cudaFree(0) 被广泛用作一种简便的方法来强制初始化 CUDA 运行时。
    // 这种自动初始化是延迟的，直到第一次需要设备资源时才发生。
    // CHECK_CUDA(cudaFree(0));

    // CHECK_CUBLAS(cublasCreate(&handle));
    // CHECK_CUDNN(cudnnCreate(&cudnn));

    // test_matmul();
    // test_add();
    // test_multiply();
    // test_softmax();
    // test_silu();
    // test_rope();
    // test_embedding();
    // test_rmsnorm();
    test_attention();

    // CHECK_CUDNN(cudnnDestroy(cudnn));
    // CHECK_CUBLAS(cublasDestroy(handle));
}


// 自定义 CUDA 核函数
__global__ void matmul_kernel(float *xout, const float *x, const float *w, int W_in, int W_out, int num) {
    int num_idx = blockIdx.y;  // 批处理索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 输出向量索引

    if (i >= W_out || num_idx >= num)
        return;

    float sum = 0.0f;
    for (int j = 0; j < W_in; j++) {
        sum += w[i * W_in + j] * x[num_idx * W_in + j];
    }
    // 使用列主序索引
    xout[i + num_idx * W_out] = sum;
}

// 错误检查宏

// 自定义核函数的主机调用函数
void matmul_cuda(float *y, const float *x, const float *w, int W_in, int W_out, int num) {
    // 计算线程块和网格大小
    int blockSize = 256;
    int gridSizeX = (W_out + blockSize - 1) / blockSize;
    int gridSizeY = num;
    dim3 gridSize(gridSizeX, gridSizeY);
    dim3 blockSizeDim(blockSize);

    // 启动核函数
    matmul_kernel<<<gridSize, blockSizeDim>>>(y, x, w, W_in, W_out, num);
}

void matmul_blas(float *y, const float *x, const float *w, int W_in, int W_out, int num) {
    // 参数设置
    float alpha = 1.0f;
    float beta = 0.0f;

    // 调用 cuBLAS 的 SGEMM
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             W_out, num, W_in,     // M, N, K
                             &alpha,
                             w, W_in,          // A lda
                             x, W_in,           // B ldb
                             &beta,
                             y, W_out)); // C ldc
}

void test_matmul() {
    std::cout << "\n***********************************************************\n" <<
                 "************************* test matmul *********************\n" <<
                 "***********************************************************\n\n";
    // 矩阵维度
    int W_in = 8192;     // 输入维度
    int W_out = 8192;    // 输出维度
    int num = 200;       // 批处理大小

    Test testtool;
    testtool.setDevice("cuda");

    // 设备内存分配
    float *d_W, *d_X, *d_C_kernel, *d_C_cublas;
    d_W = testtool.getArr(W_out * W_in, true);
    d_X = testtool.getArr(num * W_in, true);
    d_C_kernel = testtool.getArr(num * W_out);
    d_C_cublas = testtool.getArr(num * W_out);

    // 同步设备
    CHECK_CUDA(cudaDeviceSynchronize());


    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    for(int run = 0; run < num_runs; run++) {
        testtool.start_timing();
        matmul_cuda(d_C_kernel, d_X, d_W, W_in, W_out, num);
        testtool.end_timing();
        total_time_kernel += testtool.duration();

        // 记录 cuBLAS 执行时间
        testtool.start_timing();
        matmul_blas(d_C_cublas, d_X, d_W, W_in, W_out, num);
        testtool.end_timing();
        total_time_cublas += testtool.duration();
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;

    testtool.check(d_C_kernel, d_C_cublas, num * W_out, "Matmul");

    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average cuBLAS SGEMM Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;

}


/**********************************************************
 ***********************************************************/

__global__ void add_cuda_kernel(float* y, const float* x1, const float* x2, int n, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size;
    if (idx < total_elements) {
        y[idx] = x1[idx] + x2[idx];
    }
}

void add_cuda(float* y, const float* x1, const float* x2, const int n, const int batch_size) {
    int total_elements = n * batch_size;
    int threads = num_threads_large;
    int blocks = (total_elements + threads - 1) / threads;
    
    add_cuda_kernel<<<blocks, threads>>>(y, x1, x2, n, batch_size);
}

void add_cudnn(float* y, const float* x1, const float* x2, const int n, const int num) {
    cudnnTensorDescriptor_t descA;

    int dimA[4] = {1, 1, num, n};
    int strideA[4] = {num * n, num * n, n, 1};

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&descA));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(descA,
                                           CUDNN_DATA_FLOAT,
                                           4,
                                           dimA,
                                           strideA));

    cudnnOpTensorDescriptor_t opTensorDesc;
    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&opTensorDesc));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(
        opTensorDesc,
        CUDNN_OP_TENSOR_ADD,
        CUDNN_DATA_FLOAT,
        CUDNN_PROPAGATE_NAN));

    float alpha1 = 1.0f; // Coefficient for A
    float alpha2 = 1.0f; // Coefficient for B
    float beta = 0.0f;    // Coefficient for C

    // Perform the operation: C = alpha1 * A + alpha2 * B + beta * C
    CHECK_CUDNN(cudnnOpTensor(cudnn,
                              opTensorDesc,
                              &alpha1,
                              descA,
                              x1,
                              &alpha2,
                              descA,
                              x2,
                              &beta,
                              descA,
                              y));
    
    CHECK_CUDNN(cudnnDestroyOpTensorDescriptor(opTensorDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(descA));
}

void test_add() {
    std::cout << "\n***********************************************************\n" <<
                 "*************************  test add  *********************\n" <<
                 "***********************************************************\n\n";
    size_t size = 5e8;

    Test testtool;
    testtool.setDevice("cuda");

    float *x1, *x2, *y_cudnn;
    x1 = testtool.getArr(size, true);
    x2 = testtool.getArr(size, true);
    y_cudnn = testtool.getArr(size);

    // 同步设备
    CHECK_CUDA(cudaDeviceSynchronize());


    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    for(int run = 0; run < num_runs; run++) {
        // 记录 cudnn 执行时间
        testtool.start_timing();
        add_cudnn(y_cudnn, x1, x2, size, 1);
        testtool.end_timing();
        total_time_cublas += testtool.duration();

        testtool.start_timing();
        add_cuda(x1, x1, x2, size, 1);
        testtool.end_timing();
        total_time_kernel += testtool.duration();
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;

    testtool.check(x1, y_cudnn, size, "ADD");

    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average cudnn SGEMM Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;
}


/**********************************************************
 ***********************************************************/

__global__ void elem_multiply_cuda_kernel(float* y, const float* x1, const float* x2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x1[idx] * x2[idx];
    }
}

void multiply_cuda(float* y, const float* x1, const float* x2, const int size) {
    int threads = num_threads_large;
    int blocks = (size + threads - 1) / threads;
    
    elem_multiply_cuda_kernel<<<blocks, threads>>>(y, x1, x2, size);
}

void multiply_cudnn(float* y, const float* x1, const float* x2, int size) {
    cudnnTensorDescriptor_t descA;

    int dimA[4] = {1, 1, 1, size};
    int strideA[4] = {size, size, size, 1};

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&descA));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(descA,
                                           CUDNN_DATA_FLOAT,
                                           4,
                                           dimA,
                                           strideA));

    cudnnOpTensorDescriptor_t opTensorDesc;
    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&opTensorDesc));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(
        opTensorDesc,
        CUDNN_OP_TENSOR_MUL,
        CUDNN_DATA_FLOAT,
        CUDNN_PROPAGATE_NAN));

    float alpha1 = 1.0f; // Coefficient for A
    float alpha2 = 1.0f; // Coefficient for B
    float beta = 0.0f;    // Coefficient for C

    // Perform the operation: C = alpha1 * A + alpha2 * B + beta * C
    CHECK_CUDNN(cudnnOpTensor(cudnn,
                              opTensorDesc,
                              &alpha1,
                              descA,
                              x1,
                              &alpha2,
                              descA,
                              x2,
                              &beta,
                              descA,
                              y));
    
    CHECK_CUDNN(cudnnDestroyOpTensorDescriptor(opTensorDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(descA));
}

void test_multiply() {
    std::cout << "\n***********************************************************\n" <<
             "*************************  test Multiply  *********************\n" <<
             "***********************************************************\n\n";
    size_t size = 1e8;

    Test testtool;
    testtool.setDevice("cuda");

    float *x1, *x2, *y_cuda, *y_cudnn;
    x1 = testtool.getArr(size, true);
    x2 = testtool.getArr(size, true);
    y_cudnn = testtool.getArr(size);
    y_cuda = testtool.getArr(size);

    // 同步设备
    CHECK_CUDA(cudaDeviceSynchronize());


    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    for(int run = 0; run < num_runs; run++) {
        testtool.start_timing();
        multiply_cuda(y_cuda, x1, x2, size);
        testtool.end_timing();
        total_time_kernel += testtool.duration();

        // 记录 cudnn 执行时间
        testtool.start_timing();
        multiply_cudnn(y_cudnn, x1, x2, size);
        testtool.end_timing();
        total_time_cublas += testtool.duration();
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;

    testtool.check(x1, y_cudnn, size, "Multiply");

    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average cudnn SGEMM Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;
}

/**********************************************************
 ***********************************************************/
#define NUM_THREADS 256
__global__ void softmax_gpu(float *__restrict__ x, int size) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int batch_idx = blockIdx.y;
    int idx = batch_idx * size;

    x += idx;

    // Step 1: Find the maximum value for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < size; i += block_size) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    using BlockReduceMax = cub::BlockReduce<float, NUM_THREADS>;
    __shared__ typename BlockReduceMax::TempStorage temp_storage_max;
    __shared__ float shared_max;

    float max_result = BlockReduceMax(temp_storage_max).Reduce(max_val, cub::Max());
    if (tid == 0) {
        shared_max = max_result;
    }
    __syncthreads();
    max_val = shared_max;

    // Step 2: Compute exponentials and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += block_size) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    using BlockReduceSum = cub::BlockReduce<float, NUM_THREADS>;
    __shared__ typename BlockReduceSum::TempStorage temp_storage_sum;
    __shared__ float shared_sum;

    float sum_result = BlockReduceSum(temp_storage_sum).Sum(sum);
    if (tid == 0) {
        shared_sum = sum_result;
    }
    __syncthreads();
    sum = shared_sum;

    // Step 3: Normalize to get softmax probabilities
    for (int i = tid; i < size; i += block_size) {
        x[i] /= sum;
    }
}

void softmax_cuda(float *x, int n, int num) {
    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(1, num);
    
    softmax_gpu<<<gridDim, blockDim>>>(x, n);
}

void softmax_cudnn(float *y, float *x, int n, int num) {
    cudnnTensorDescriptor_t desc_x, desc_y;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc_x));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc_y));

    int dimA[4] = {n, 1, num, 1};
    int strideA[4] = {num, num, 1, 1};

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc_x,
                                           CUDNN_DATA_FLOAT,
                                           4,
                                           dimA,
                                           strideA));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc_y,
                                           CUDNN_DATA_FLOAT,
                                           4,
                                           dimA,
                                           strideA));
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE; // 对每个实例（每列）单独应用 Softmax
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_FAST;
    float alpha = 1.0f;
    float beta = 0.0f;

    // 执行 Softmax 前向传播
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn,
                                    algo,
                                    mode,
                                    &alpha,
                                    desc_x,
                                    x,
                                    &beta,
                                    desc_y,
                                    y));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc_x));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc_y));
}

void test_softmax() {
    std::cout << "\n***********************************************************\n" <<
             "*************************  test Softmax  *********************\n" <<
             "***********************************************************\n\n";
    size_t n = 8192;
    size_t num = 5000;

    Test testtool;
    testtool.setDevice("cuda");

    float *x, *x1, *y_cudnn;
    x = testtool.getArr(n * num, true);
    x1 = testtool.getArr(n * num, true);
    y_cudnn = testtool.getArr(n * num);

    // 同步设备
    CHECK_CUDA(cudaDeviceSynchronize());


    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    for(int run = 0; run < num_runs; run++) {

        testtool.start_timing();
        softmax_cuda(x1, n, num);
        testtool.end_timing();
        total_time_kernel += testtool.duration();

        // 记录 cudnn 执行时间
        testtool.start_timing();
        softmax_cudnn(y_cudnn, x, n, num);
        testtool.end_timing();
        total_time_cublas += testtool.duration();
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;

    softmax_cuda(x, n, num);
    testtool.check(x, y_cudnn, n * num, "Softmax");

    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average cudnn SGEMM Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;
}


/**********************************************************
 ***********************************************************/

unsigned int nextPowerOfTwo(unsigned int x) {
    if (x == 0)
        return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

// Kernel to compute attention scores
__global__ void compute_scores_kernel(float* score, const float* q, const float* k,
                                      int dim, int q_head, int kv_head, int pos,
                                      int rep, int kv_dim, float scale) {
    int hq = blockIdx.y * blockDim.y + threadIdx.y; // Query head index
    int p = blockIdx.x * blockDim.x + threadIdx.x;  // Position index

    if (hq < q_head && p < pos) {
        const float* _q = q + hq * dim;
        const float* _k = k + p * kv_dim + (hq / rep) * dim;
        float dot = 0.0f;
        for(int d = 0; d < dim; d++) {
            dot += _q[d] * _k[d];
        }
        int s_index = hq * pos + p;
        score[s_index] = dot * scale;
    }
}

// Kernel to apply softmax to the scores
__global__ void _softmax_kernel(float* score, int pos) {
    extern __shared__ float shared_data[];
    int hq = blockIdx.x; // Each block processes one query head
    int p = threadIdx.x; // Thread processes one position

    float val = -INFINITY;
    if (p < pos) {
        val = score[hq * pos + p];
    }
    shared_data[p] = val;
    __syncthreads();

    // Compute max value for numerical stability
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (p < stride && (p + stride) < pos) {
            shared_data[p] = fmaxf(shared_data[p], shared_data[p + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_data[0];
    __syncthreads();

    // Compute exponentials and sum
    if (p < pos) {
        val = expf(val - max_val);
        score[hq * pos + p] = val;
        shared_data[p] = val;
    } else {
        shared_data[p] = 0.0f;
    }
    __syncthreads();

    // Compute sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (p < stride && (p + stride) < pos) {
            shared_data[p] += shared_data[p + stride];
        }
        __syncthreads();
    }
    float sum = shared_data[0];
    __syncthreads();

    // Normalize the scores
    if (p < pos) {
        score[hq * pos + p] /= sum;
    }
}

// Kernel to compute the output y
__global__ void compute_output_kernel(float* y, const float* score, const float* v,
                                      int dim, int q_head, int kv_head, int pos,
                                      int rep, int kv_dim) {
    int hq = blockIdx.y * blockDim.y + threadIdx.y; // Query head index
    int d = blockIdx.x * blockDim.x + threadIdx.x;  // Dimension index

    if (hq < q_head && d < dim) {
        float sum = 0.0f;
        int s_index = hq * pos;
        int v_offset = (hq / rep) * dim + d;
        for(int p = 0; p < pos; p++) {
            float s = score[s_index + p];
            float v_val = v[p * kv_dim + v_offset];
            sum += s * v_val;
        }
        y[hq * dim + d] = sum;
    }
}

// Main function to perform masked attention using CUDA
void maksed_attention_cuda(float* y, const float* q, const float* k, const float* v,
                           const int dim, const int q_head, const int kv_head, const int _pos) {
    
    int pos = _pos + 1;
    int rep = q_head / kv_head;
    int kv_dim = kv_head * dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    // Allocate device memory
    float *d_score;
    cudaError_t err = cudaMalloc((void**)&d_score, q_head * pos * sizeof(float));
    cudaMemset(d_score, 0, q_head * pos * sizeof(float));

    // Launch kernel to compute scores
    dim3 blockDimScore(16, 16);
    dim3 gridDimScore((pos + blockDimScore.x - 1) / blockDimScore.x,
                      (q_head + blockDimScore.y - 1) / blockDimScore.y);
    compute_scores_kernel<<<gridDimScore, blockDimScore>>>(d_score, q, k, dim,
                                                           q_head, kv_head, pos, rep, kv_dim, scale);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel compute_scores_kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    // 同步并再次检查
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel compute_scores_kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }

    float* score_cpu = new float[q_head * pos];
    cudaMemcpy(score_cpu, d_score, q_head * pos * sizeof(int), cudaMemcpyDeviceToHost);

    // Launch kernel to apply softmax
    int softmaxBlockSize = nextPowerOfTwo(pos);
    size_t sharedMemSize = softmaxBlockSize * sizeof(float);
    _softmax_kernel<<<q_head, softmaxBlockSize, sharedMemSize>>>(d_score, pos);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel _softmax_kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    // 同步并再次检查
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel _softmax_kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }

    // float* score_cpu = new float[q_head * pos];
    // cudaMemcpy(score_cpu, d_score, q_head * pos * sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "q = " << pos-1 << std::endl;
    // for(int i = 0; i < pos; i++) {
    //     std::cout << "\t";
    //     for(int j = 0; j < kv_head; j++) {
    //         std::cout << score_cpu[i*kv_head + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    
    // Launch kernel to compute the output y
    dim3 blockDimOutput(32, 32);
    dim3 gridDimOutput((dim + blockDimOutput.x - 1) / blockDimOutput.x,
                       (q_head + blockDimOutput.y - 1) / blockDimOutput.y);
    compute_output_kernel<<<gridDimOutput, blockDimOutput>>>(y, d_score, v, dim, q_head, kv_head, pos, rep, kv_dim);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel compute_output_kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    // 同步并再次检查
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel compute_output_kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }

    // Free device memory
    cudaFree(d_score);
}



/**********************************************************
 ***********************************************************/

__global__ void silu_cuda_kernel(float *x, int n, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size;
    if (i < total_elements) {
        float val = x[i];
        x[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}

void silu_cuda(float *x, const int n, const int num) {
    int total_elements = n * num;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    silu_cuda_kernel<<<blocks, threads>>>(x, n, num);
}

void test_silu() {
    std::cout << "\n***********************************************************\n" <<
         "*************************  test Silu  *********************\n" <<
         "***********************************************************\n\n";

    size_t n = 8192;
    size_t num = 5000;

    Test testtool;
    testtool.setDevice("cuda");

    float *x;
    x = testtool.getArr(n * num, true);

    CHECK_CUDA(cudaDeviceSynchronize());


    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    for(int run = 0; run < num_runs; run++) {

        testtool.start_timing();
        silu_cuda(x, n, num);
        testtool.end_timing();
        total_time_kernel += testtool.duration();
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;


    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average cudnn SGEMM Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;
}

/**********************************************************
 ***********************************************************/

__global__ void apply_rope_kernel_optimized(
    float *x, const float *pos, const float *cos, const float *sin,
    int n, int dim, int num
) {
    // 计算p和i的索引
    int p = blockIdx.y;
    int i = blockIdx.x;

    if (p >= num || i >= n / (dim * 2))
        return;

    // 线程索引对应j
    int j = threadIdx.x;

    if (j >= dim)
        return;

    // 获取当前p对应的cos和sin的指针
    int pos_p = static_cast<int>(pos[p]);
    const float* cos_ptr = cos + pos_p * dim;
    const float* sin_ptr = sin + pos_p * dim;

    // 使用共享内存
    extern __shared__ float shared_mem[];
    float* shared_cos = shared_mem;
    float* shared_sin = shared_mem + dim;

    // 将cos和sin加载到共享内存
    shared_cos[j] = cos_ptr[j];
    shared_sin[j] = sin_ptr[j];
    __syncthreads();

    // 计算x的起始位置
    float* x_ptr = x + p * n + i * dim * 2;

    // 读取当前值
    float x1 = x_ptr[j];
    float x2 = x_ptr[dim + j];

    // 从共享内存中读取cos和sin
    float c = shared_cos[j];
    float s = shared_sin[j];

    // 应用旋转
    x_ptr[j]       = x1 * c - x2 * s;
    x_ptr[dim + j] = x2 * c + x1 * s;
}


// 封装的函数，支持批处理
void apply_rope_cuda(float *x, const float *pos, const float *cos, const float *sin, const int n, const int dim, const int num) {
    // 计算网格和线程块的尺寸
    dim3 blockDim(dim);
    dim3 gridDim(n / (dim * 2), num);

    // 计算共享内存的大小
    size_t sharedMemSize = 2 * dim * sizeof(float);
    // 启动内核
    apply_rope_kernel_optimized<<<gridDim, blockDim, sharedMemSize>>>( x, pos, cos, sin, n, dim, num);
}

void test_rope() {
    const int num = 5000;       // 对应函数参数
    const int n   = 8192;       // 对应函数参数
    const int dim = 32;         // 对应函数参数
    
    // 为简化，我们统一称 max_length=10000，用来存储 cos/sin 的预计算
    const int max_length = 5001;

    Test testtool;
    testtool.setDevice("cuda");

    float *x = testtool.getArr(n * num, true);
    float *pos = testtool.getArr(num, true);
    float *cos_val = testtool.getArr(max_length * dim, true);
    float *sin_val = testtool.getArr(max_length * dim, true);

    CHECK_CUDA(cudaDeviceSynchronize());

    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    for(int run = 0; run < num_runs; run++) {

        testtool.start_timing();
        apply_rope_cuda(x, pos, cos_val, sin_val, n, dim, num);
        testtool.end_timing();
        total_time_kernel += testtool.duration();
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;


    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average cudnn SGEMM Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;
}

/**********************************************************
 ***********************************************************/

__global__ void embedding_cuda_kernel(float* y, const int* x, const float* W, const int d, const int x_size) {
    // 计算当前线程处理的 token 索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < x_size) {
        // 获取当前 token 的索引 (输入 token)
        int token_idx = x[idx];

        // 每个 token 对应的 embedding 向量
        const float* W_row = W + token_idx * d;

        // 将 embedding 写入输出
        float* y_row = y + idx * d;
        for (int i = 0; i < d; ++i) {
            y_row[i] = W_row[i];
        }
    }
}

void embedding_cuda(float* y, const int* x, const float* W, const int d, const int x_size) {
    // 定义线程块和网格的维度
    int block_size = 256;
    int grid_size = (x_size + block_size - 1) / block_size;

    // 启动 CUDA 核函数
    embedding_cuda_kernel<<<grid_size, block_size>>>(y, x, W, d, x_size);
}


// 使用只读缓存优化嵌入矩阵访问
__global__ void embeddingLookupOptimized(
    const float* __restrict__ embeddingMatrix, // 列主序的嵌入矩阵，大小为 hiddensize * vocabsize
    const int* __restrict__ indices,          // 输入的索引数组，大小为 batch_size
    float* __restrict__ output,               // 输出的嵌入向量，大小为 batch_size * hiddensize
    int hiddensize,
    int vocabsize
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= hiddensize * gridDim.y) return;

    // 计算对应的词汇索引
    int word_idx = indices[blockIdx.y * blockDim.x + threadIdx.x];
    if (word_idx < 0 || word_idx >= vocabsize) return;

    // 每个线程加载多个元素以提高内存带宽利用率
    for (int i = threadIdx.y; i < hiddensize; i += blockDim.y) {
        // 使用__ldg指令加载只读数据，提高缓存利用率
        output[blockIdx.y * hiddensize + i] = __ldg(&embeddingMatrix[word_idx * hiddensize + i]);
    }
}

// 主机端函数：配置并调用核函数
void embeddingLookupOptimizedHost(
    const float* d_embeddingMatrix, // 设备端嵌入矩阵
    const int* d_indices,           // 设备端索引数组
    float* d_output,                // 设备端输出数组
    int batch_size,
    int hiddensize,
    int vocabsize
) {
    // 选择合适的线程块大小和网格布局
    // 假设每个线程块处理一个batch元素
    dim3 threadsPerBlock(256, 4); // 256线程，每个线程处理4个元素
    dim3 numBlocks((batch_size + 255) / 256, 1); // 根据batch_size调整

    // 启动核函数
    embeddingLookupOptimized<<<numBlocks, threadsPerBlock>>>(
        d_embeddingMatrix,
        d_indices,
        d_output,
        hiddensize,
        vocabsize
    );

    // 检查是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel failed: %s\n", cudaGetErrorString(err));
        // 处理错误（例如，退出程序）
    }
}

void test_embedding() {
    size_t vocab_size = 128500;
    size_t hidden_size = 8192;
    size_t batch_size = 2000;

    float* h_embeddingMatrix = (float*)malloc(hidden_size * vocab_size * sizeof(float));
    int* h_indices = (int*)malloc(batch_size * sizeof(int));

    // 初始化嵌入矩阵和索引（此处为示例，实际应用中应有具体数据）
    for (int i = 0; i < hidden_size * vocab_size; ++i) {
        h_embeddingMatrix[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (int i = 0; i < batch_size; ++i) {
        h_indices[i] = i % vocab_size;
    }

    // 分配设备端内存
    float* d_embeddingMatrix;
    int* d_indices;
    float* d_output1;
    float* d_output2;
    cudaMalloc(&d_embeddingMatrix, hidden_size * vocab_size * sizeof(float));
    cudaMalloc(&d_indices, batch_size * sizeof(int));
    cudaMalloc(&d_output1, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_output2, batch_size * hidden_size * sizeof(float));

    // 复制数据到设备端
    cudaMemcpy(d_embeddingMatrix, h_embeddingMatrix, hidden_size * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, batch_size * sizeof(int), cudaMemcpyHostToDevice);


    CHECK_CUDA(cudaDeviceSynchronize());

    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    Test testtool;
    testtool.setDevice("cuda");

    for(int run = 0; run < num_runs; run++) {

        testtool.start_timing();
        embedding_cuda(d_output1, d_indices, d_embeddingMatrix, hidden_size, batch_size);
        testtool.end_timing();
        total_time_kernel += testtool.duration();

        testtool.start_timing();
        embeddingLookupOptimizedHost(d_embeddingMatrix , d_indices, d_output2, batch_size, hidden_size, vocab_size);
        testtool.end_timing();
        total_time_cublas += testtool.duration();
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;


    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average new Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;

    free(h_embeddingMatrix);
    free(h_indices);
    cudaFree(d_embeddingMatrix);
    cudaFree(d_indices);
    cudaFree(d_output1);
    cudaFree(d_output2);
}


/**********************************************************
 ***********************************************************/

// RMSNorm CUDA 内核
__global__ void rmsnorm_kernel(float *x, const float *w, int n, int batch_size, const float epsilon, int elementsPerThread) {
    int batch_idx = blockIdx.y;  // 批次索引
    // 计算输入和输出的偏移量
    float *x_batch = x + batch_idx * n;

    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < n)
            ss += x_batch[j] * x_batch[j];
    }

    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    ss = BlockReduce(temp_storage).Sum(ss);

    // 计算归一化因子
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= n;
        ss += epsilon;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    
    float ss_normalized = shared_ss;

    // 归一化并缩放
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < n) {
            x_batch[j] = w[j] * (ss_normalized * x_batch[j]);
        }
    }
}

inline int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

// 封装的 rmsnorm 函数
void rmsnorm_cuda(float* x, const float* w, int n, int batch_size, const float epsilon) {
    int elementsPerThread = divUp(n, num_threads_large);

    // 计算线程块和网格大小
    dim3 blockSize(num_threads_large);
    dim3 gridSize(1, batch_size);  // 每个批次一个线程块

    // 调用 CUDA 内核
    rmsnorm_kernel<<<gridSize, blockSize>>>(x, w, n, batch_size, epsilon, elementsPerThread);
}

void test_rmsnorm() {
    std::cout << "\n***********************************************************\n" <<
         "*************************  test rsmnorm  *********************\n" <<
         "***********************************************************\n\n";

    size_t n = 8192;
    size_t num = 5000;

    Test testtool;
    testtool.setDevice("cuda");

    float *x = testtool.getArr(n * num, true);
    float *w = testtool.getArr(n, true);

    CHECK_CUDA(cudaDeviceSynchronize());


    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    for(int run = 0; run < num_runs; run++) {

        testtool.start_timing();
        rmsnorm_cuda(x, w, n, num, 6e-5);
        testtool.end_timing();
        total_time_kernel += testtool.duration();
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;


    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average cudnn SGEMM Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;
}

/**********************************************************
 ***********************************************************/

// q [seq_q,  head_num, dim]
// k [seq_kv, head_num, dim]
// kernel<<<(seq_kv, head_num), (seq_q)>>>
// scores [seq_q, head_num, seq_kv]
__global__ void compute_masked_scores_kernel(
    float* scores,
    float* __restrict__ q,
    float* __restrict__ k_cache,
    int* q_pos,
    int dim,
    float  scale
) {
    int kv_id = blockIdx.x;      // gridDim.x = seq_kv
    int head_id = blockIdx.y;    // gridDim.y = head_num
    int q_id = threadIdx.x;      // blockDim.x = seq_q

    int kv_num = gridDim.x;      // seq_kv
    int head_num = gridDim.y;    // head_num

    int pos = q_pos[q_id];

    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < dim; i++) {
        sum += q[q_id * head_num * dim + head_id*dim + i] * k_cache[kv_id*head_num*dim + head_id*dim + i];
    }

    if(kv_id <= pos) {
        scores[q_id*head_num*kv_num + head_id*kv_num + kv_id] = sum * scale;
    } else {
        scores[q_id*head_num*kv_num + head_id*kv_num + kv_id] = -INFINITY;
    }
}

// o      [seq_q, head_num, dim]
// scores [seq_q, head_num, seq_kv]
// kernel<<<(seq_q), (head_num)>>>
__global__ void compute_masked_output_kernel(
    float* o,
    float* v_cache,
    float* scores,
    int kv_num,
    int dim
) {
    int head_num = blockDim.x;

    int h_id = threadIdx.x;
    int q_id = blockIdx.x;
    

    for(int i = 0; i < kv_num; i++) {
        float s = scores[q_id*head_num*kv_num + h_id*kv_num + i];
        #pragma unroll
        for(int d = 0; d < dim; d++) {
            o[q_id*head_num*dim + h_id*dim + d] += s * v_cache[i*head_num*dim + h_id*dim + d];
        }
    }

}

void masked_attention(
    float* y, 
    float* q, 
    float* k, 
    float* v, 
    float* scores, 
    int* pos, 
    int dim, 
    int head_num,
    int seq_q,
    int seq_kv
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    compute_masked_scores_kernel<<<dim3(seq_kv, head_num), dim3(seq_q)>>>(scores, q, k, pos, dim, scale);

    softmax_gpu<<<dim3(1, seq_q * head_num), dim3(NUM_THREADS)>>>(scores, seq_kv);

    compute_masked_output_kernel<<<dim3(seq_q), dim3(head_num)>>>(y, v, scores, seq_kv, dim);
}


void test_attention() {

    std::cout << "\n***********************************************************\n" <<
     "*************************  test multihead-attention  *********************\n" <<
     "***********************************************************\n\n";

    // 示例参数
    const int seq_q = 2000;
    const int head_num = 128;
    const int dim = 64;
    const int seq_kv = seq_q;

    // 主机端分配
    size_t q_size = seq_q * head_num * dim;
    size_t k_size = seq_kv * head_num * dim;
    size_t scores_size = seq_q * head_num * seq_kv;

    Test testtool;
    testtool.setDevice("cuda");

    float *q = testtool.getArr(q_size, true);
    float* k = testtool.getArr(k_size, true);
    float* v = testtool.getArr(k_size, true);
    float* scores = testtool.getArr(scores_size);
    float* o1 = testtool.getArr(q_size);

    float* o2 = testtool.getArr(q_size);



    int* pos_cpu = new int[seq_q * sizeof(int)];

    for(int i = 0; i < seq_q; i++) pos_cpu[i] = i;

    int *pos;
    cudaMalloc(&pos, seq_q * sizeof(int));
    cudaMemcpy(pos, pos_cpu, seq_q * sizeof(int), cudaMemcpyHostToDevice);

    CHECK_CUDA(cudaDeviceSynchronize());


    // 变量用于累积时间
    float total_time_kernel = 0.0f;
    float total_time_cublas = 0.0f;

    for(int run = 0; run < num_runs; run++) {

        // testtool.start_timing();
        // for(int i = 0; i < seq_q; i++) {
        //     maksed_attention_cuda(o2+i*dim*head_num, q+i*dim*head_num, k, v, dim, head_num, head_num, i);
        // }
        // testtool.end_timing();
        // total_time_kernel += testtool.duration();

        testtool.start_timing();
        masked_attention(o1, q, k, v, scores, pos, dim, head_num, seq_q, seq_kv);
        testtool.end_timing();
        total_time_cublas += testtool.duration();
        cudaMemset(o2, 0, q_size*sizeof(float));
    }

    // 计算平均时间
    float average_time_kernel = total_time_kernel / num_runs;
    float average_time_cublas = total_time_cublas / num_runs;


    // 输出性能比较
    std::cout << "Average Custom Kernel Time over " << num_runs << " runs: " 
              << average_time_kernel << " ms" << std::endl;
    std::cout << "Average New Time over " << num_runs << " runs: " 
              << average_time_cublas << " ms" << std::endl;



    testtool.check(o1, o2, q_size, "multihead-attention");
    

    delete[] pos_cpu;
}
