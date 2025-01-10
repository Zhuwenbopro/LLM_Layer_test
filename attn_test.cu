// matmul_compare.cu
#include "test.h"
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cblas.h>

// nvcc -o test main.cu -lopenblas

void origin(float* y, const float* q, const float* k, const float* v, const int dim, const int q_head, const int kv_head, const int _pos);
void masked_attention_cpu(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv);
void masked_attention_cuda(
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
);

int main() {
    size_t pos = 5;
    size_t hidden_size = 2048;
    size_t head_dim = 64;
    size_t head_num = 32;

    size_t scores_size = 5 * head_num * 5;

    assert(hidden_size == head_dim*head_num);

    Test testtool;
    testtool.setDevice("cpu");

    float* q = testtool.getArr(hidden_size * pos, true);
    float* k = testtool.getArr(hidden_size * pos, true);
    float* v = testtool.getArr(hidden_size * pos, true);
    float* scores = testtool.getArr(scores_size);
    float* o1 = testtool.getArr(hidden_size*pos);
    float* o2 = testtool.getArr(hidden_size*pos);

    int offset = 0;
    for(int p = 0; p < pos; p++) {
        origin(o1 + offset*hidden_size, q + offset*hidden_size, k, v, head_dim, head_num, head_num, p);
        offset += 1;
    }

    int position[pos];
    position[0] = 0;position[1] = 1;position[2] = 2;position[3] = 3;position[4] = 4;
    masked_attention_cpu(o2, q, k, v, nullptr, position, head_dim, head_num, pos, pos);


    testtool.check(o1, o2, hidden_size * pos, "CPU Attention");

    testtool.setDevice("cuda");

    float* o3 = testtool.get_from_cpu(o2, hidden_size*pos);
    float* o_cuda = testtool.getArr(hidden_size*pos);
    float* q_cuda = testtool.get_from_cpu(q, hidden_size * pos);
    float* k_cuda = testtool.get_from_cpu(k, hidden_size * pos);
    float* v_cuda = testtool.get_from_cpu(v, hidden_size * pos);

    int* p_cuda = (int*)testtool.get_from_cpu((float*)position, pos * sizeof(float)/sizeof(int));

    masked_attention_cuda(o_cuda, q_cuda, k_cuda, v_cuda, nullptr, p_cuda, head_dim, head_num, pos, pos);

    testtool.check(o3, o_cuda, hidden_size * pos, "CUDA Attention");
}

void softmax_cpu(float *x, int n, int batch_size) {
    for(int b = 0; b < batch_size; b++) {
        // 找到输入数组中的最大值，以提高数值稳定性
        float* input = x + b * n;
        float max_val = input[0];
        for(int i = 1; i < n; ++i){
            if(input[i] > max_val){
                max_val = input[i];
            }
        }

        // 计算每个元素的指数值，并累加
        float sum = 0.0f;
        for(int i = 0; i < n; ++i){
            input[i] = std::exp(input[i] - max_val);
            sum += input[i];
        }

        // 将每个指数值除以总和，得到概率分布
        for(int i = 0; i < n; ++i){
            input[i] /= sum;
        }
    }
}

void origin(float* y, const float* q, const float* k, const float* v, const int dim, const int q_head, const int kv_head, const int _pos) {
    int pos = _pos + 1;
    float* score = new float[q_head * pos](); // 置初始值为0，列优先，pos行，q_head列

    int rep = q_head / kv_head;
    int kv_dim = kv_head * dim;

    float scale = 1.0 / std::sqrt(static_cast<float>(dim));
    for(int p = 0; p < pos; p++) {
        for(int hq = 0; hq < q_head; hq++) {
            const float* _q = q + hq * dim;
            const float* _k = k + p * kv_dim + (hq / rep) * dim;
            const int s_index = hq*pos + p;
            for(int d = 0; d < dim; d++) {
                score[s_index] += _q[d] * _k[d];
            }
            score[s_index] *= scale;
        }
    }

    softmax_cpu(score, pos, q_head);

    std::memset(y, 0, dim * q_head * sizeof(float));

    for(int hq = 0; hq < q_head; hq++) {
        float* _s = score + hq * pos;
        float* _y = y + hq * dim;
        for(int p = 0; p < pos; p++) {
            const float* _v = v + p * kv_dim + (hq / rep) * dim;
            for(int d = 0; d < dim; d++) {
                _y[d] += _s[p] * _v[d];
            }
        }
    }

    delete score;
}

inline float dot(float* a, float* b, size_t size) {
    return cblas_sdot(size, a, 1, b, 1);
}

inline void scale(float* a, float alpha, size_t size) {
    cblas_sscal(size, alpha, a, 1);
}

// Y = alpha * X + Y
inline void add(float* y, float* x, size_t size, float alpha = 1) {
    cblas_saxpy(size, alpha, x, 1, y, 1);
}

void masked_attention_cpu(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv) {
    bool hasvalue = true;
    if(scores == nullptr) {
        scores = new float[seq_kv * head_num];
        hasvalue = false;
    }

    std::memset(y, 0, dim * head_num * seq_q * sizeof(float));

    float scale_ = 1.0 / std::sqrt(static_cast<float>(dim));

    int kv_num_ = seq_kv - seq_q;
    for(int i_q = 0; i_q < seq_q; i_q++) {
        kv_num_++;
        float* q_ = q + i_q * dim * head_num;
        float* y_ = y + i_q * dim * head_num;
        for(int i_kv = 0; i_kv < kv_num_; i_kv++) {
            float* k_ = k + i_kv * dim * head_num;
            for(int h = 0; h < head_num; h++) {
                scores[i_kv + h*kv_num_] = dot(q_ + h*dim, k_ + h*dim, dim);
            }
        }
        scale(scores, scale_, kv_num_*head_num);
        softmax_cpu(scores, kv_num_, head_num);

        for(int i_kv = 0; i_kv < kv_num_; i_kv++) {
            float* v_ = v + i_kv * dim * head_num;
            for(int h = 0; h < head_num; h++) {
                add(y_ + h*dim, v_ + h*dim, dim, scores[i_kv + h*kv_num_]);
            }
        }
    }

    if(!hasvalue) delete scores;
}



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

__global__ void softmax_gpu(float *__restrict__ x, int size) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int batch_idx = blockIdx.y;
    int idx = batch_idx * size;

    x += idx;

    // 找到最大值（用于数值稳定性）
    float max_val = -FLT_MAX;
    for (int i = tid; i < size; i += block_size) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_max;

    float max_result = BlockReduce(temp_storage).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_max = max_result;
    }
    __syncthreads();
    max_val = shared_max;

    // 计算指数和总和
    float sum = 0.0f;
    for (int i = tid; i < size; i += block_size) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        shared_max = sum;
    }
    __syncthreads();
    sum = shared_max;

    // 归一化
    for (int i = tid; i < size; i += block_size) {
        x[i] /= sum;
    }
}


void masked_attention_cuda(
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
    bool hasvalue = true;
    if(scores == nullptr) {
        hasvalue = false;
        cudaError_t err = cudaMalloc((void**)&scores, seq_kv*head_num*dim*sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }

    compute_masked_scores_kernel<<<dim3(seq_kv, head_num), dim3(seq_q)>>>(scores, q, k, pos, dim, scale);

    softmax_gpu<<<dim3(1, seq_q * head_num), dim3(1024)>>>(scores, seq_kv);

    compute_masked_output_kernel<<<dim3(seq_q), dim3(head_num)>>>(y, v, scores, seq_kv, dim);

    if(!hasvalue) cudaFree(scores);
}


