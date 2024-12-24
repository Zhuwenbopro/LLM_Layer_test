#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <random>
#include <omp.h>
#include <cblas.h>

// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

auto start = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;

size_t iter_num = 100;

// g++ -O3 -fopenmp -march=native main.cc -o test -lopenblas

// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green

inline void check_pass(const std::string&  message){
    std::cout << GREEN << message << RESET << std::endl;
}

inline void check_error(const std::string&  message){
    std::cout << RED << message << RESET << std::endl;
}

void check(float*a , float* b, size_t size, const std::string& str) {
    // 检查两个版本的计算结果是否一致
    bool results_match = true;
    for (int i = 0; i < size; ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-3f) {
            results_match = false;
            std::cout << "different at " << i << " :  " << a[i] << " vs " << b[i] << std::endl;
            break;
        }
    }

    // 输出是否一致
    if (results_match) {
        check_pass("["+str+"] The results are consistent.\n");
    } else {
        check_error("["+str+"] The results are NOT consistent!!!\n");
    }
}

void test_add(size_t size);
void test_multiply(size_t size);
void test_rmsnorm(int n, int batch_size, const float epsilon);
void test_attn();
void test_matmul();
void test_softmax(int n, int batch_size);
void test_silu(int n, int batch_size);
void test_rope();
void test_embedding(size_t vocal_size, size_t hidden_size, size_t select);

int main() {

    int batch_size = 5000;
    int n = 8192;
    float epsilon = 1e-6f;  // 防止除以零的小常数

    // 执行测试
    // test_add(5e8);
    // test_rmsnorm(8192, 5000, epsilon);
    test_attn();
    // test_matmul();
    test_softmax(5000, 8192);
    test_silu(5000, 8192);
    test_rope();
    test_multiply(1e8);
    test_embedding(128500, 8192, 2000);

    return 0;
}

/**********************************************************
 ***********************************************************/

void add_cpu(float* y, float* x1, float* x2, size_t size) {
    for(int i = 0; i < size; i++)
        y[i] = x1[i] + x2[i];
}

// Y = alpha * X + Y
inline void add(float* y, float* x, size_t size, float alpha = 1) {
    cblas_saxpy(size, alpha, x, 1, y, 1);
}

void test_add(size_t size) {

    // 分配对齐内存
    float* a = new float[size * sizeof(float)];
    float* b = new float[size * sizeof(float)];
    float* result = new float[size * sizeof(float)];


    for (size_t i = 0; i < size; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
        result[i] = a[i];
    }

    // 调用 SIMD 加法函数
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        add_cpu(result, b, result, size);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "ADD Time taken: " << 1000 * duration.count() / iter_num << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        add(a, b, size);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CBLAS Time taken: " << 1000 * duration.count() / iter_num << " ms" << std::endl;

    check(a, result, size, "ADD");

    // 释放内存
    std::free(a);
    std::free(b);
    std::free(result);
}

/**********************************************************
 ***********************************************************/

void rmsnorm_cpu(float* x, const float* w, int n, int batch_size, const float epsilon) {
    for(int b = 0; b < batch_size; b++) {
        // 求平方和
        float sum_of_squares = 0.0f;
        for (int i = 0; i < n; ++i) {
            int index = i + b * n;
            sum_of_squares += x[index] * x[index];
        }

        // 计算均方根归一化系数
        float mean_square = sum_of_squares / n;
        float rms = 1.0f / std::sqrt(mean_square + epsilon); // 防止除以零

        // 归一化并乘以权重
        for (int i = 0; i < n; ++i) {
            int index = i + b * n;
            x[index] = w[i] * x[index] * rms;
        }
    }
}

void rmsnorm_cblas(float* x, float* w, int n, int batch_size, const float epsilon) {
    float sn = std::sqrt(n);
    for(int b = 0; b < batch_size; b++) {
        float snrm = cblas_snrm2(n, x + b*n, 1);

        float rms = sn / (snrm + epsilon);
        
        int index = b * n;
        for (int i = 0; i < n; ++i, index++) {
            x[index] = w[i] * x[index] * rms;
        }
    }
    
}

void test_rmsnorm(int n, int batch_size, const float epsilon) {
    // 使用 C 风格的数组分配内存
    float* x = new float[n * batch_size];
    float* w = new float[n * batch_size];

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 初始化 x 和 w 数据
    for (int i = 0; i < n * batch_size; ++i) {
        x[i] = dist(gen);
        w[i] = dist(gen);
    }

    // 测试 CPU 版本
    float* x_cpu = new float[n * batch_size];
    std::copy(x, x + n * batch_size, x_cpu);  // 复制 x 到 x_cpu

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        rmsnorm_cpu(x_cpu, w, n, batch_size, epsilon);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "RMSNorm CPU version time: " << 1000 * duration.count() / iter_num << " ms\n";

    // 测试 cblas 版本
    float* x_cblas = new float[n * batch_size];
    std::copy(x, x + n * batch_size, x_cblas);  // 复制 x 到 x_cblas

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        rmsnorm_cblas(x_cblas, w, n, batch_size, epsilon);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "RMSNorm CBLAS version time: " << 1000 * duration.count() / iter_num << " ms\n";


    check(x_cblas, x_cpu, n * batch_size, "RMSNorm");

    // 清理内存
    delete[] x;
    delete[] w;
    delete[] x_cpu;
    delete[] x_cblas;
}



inline float dot(float* a, float* b, size_t size) {
    return cblas_sdot(size, a, 1, b, 1);
}

inline void scale(float* a, float alpha, size_t size) {
    cblas_sscal(size, alpha, a, 1);
}

/**********************************************************
 ***********************************************************/

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

void softmax_openblas(float *x, int n, int batch_size) {
    // Step 1: Subtract max value from each column (vector) for numerical stability
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        // Find the maximum element in the column
        float max_val = -MAXFLOAT;
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j; // Column-major index calculation
            max_val = std::max(max_val, x[idx]);
        }
        
        // Subtract the max from each element in the column
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j; // Column-major index calculation
            x[idx] -= max_val;
        }
    }

    // Step 2: Compute the exponential of each element
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            x[idx] = std::exp(x[idx]); // Element-wise exp
        }
    }

    // Step 3: Compute the sum of exponentials for each column
    std::vector<float> row_sums(batch_size, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            sum += x[idx];
        }
        row_sums[i] = sum;
    }

    // Step 4: Normalize each element by the column sum
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        float row_sum = row_sums[i];
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            x[idx] /= row_sum;
        }
    }
}

void test_softmax(int n, int batch_size) {
    float* x_cpu = new float[n * batch_size];
    float* x = new float[n * batch_size];

    // 初始化 x 和 w 数据
    for (int i = 0; i < n * batch_size; ++i) {
        x[i] = dist(gen);
        x_cpu[i] = x[i];
    }


    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        softmax_cpu(x_cpu, n, batch_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Softmax CPU version time: " << 1000 * duration.count() / iter_num << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        softmax_openblas(x, n, batch_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Softmax OpenBLAS version time: " << 1000 * duration.count() / iter_num << " ms\n";

    check(x_cpu, x, n * batch_size, "Softmax");

    delete x;
    delete x_cpu;
}

/**********************************************************
 ***********************************************************/

// q    (dim , q_head  * q_num)
// kv   (dim , kv_head * kv_num)
// mask (kv_num , q_num)
void masked_attention_cpu(float* y, float* q, float* k, float* v, float* mask, int dim, int head_num, int kv_num, int q_num) {
    float* score = new float[kv_num * head_num];
    std::memset(y, 0, dim * head_num * q_num * sizeof(float));

    float scale_ = 1.0 / std::sqrt(static_cast<float>(dim));

    int kv_num_ = kv_num - q_num;
    for(int i_q = 0; i_q < q_num; i_q++) {
        kv_num_++;
        float* q_ = q + i_q * dim * head_num;
        float* y_ = y + i_q * dim * head_num;
        for(int i_kv = 0; i_kv < kv_num_; i_kv++) {
            float* k_ = k + i_kv * dim * head_num;
            for(int h = 0; h < head_num; h++) {
                score[i_kv + h*kv_num_] = dot(q_ + h*dim, k_ + h*dim, dim);
            }
        }
        scale(score, scale_, kv_num_*head_num);
        softmax_openblas(score, kv_num_, head_num);

        for(int i_kv = 0; i_kv < kv_num_; i_kv++) {
            float* v_ = v + i_kv * dim * head_num;
            for(int h = 0; h < head_num; h++) {
                add(y_ + h*dim, v_ + h*dim, dim, score[i_kv + h*kv_num_]);
            }
        }


    }

    delete score;
}

void masked_attention_origin(float* y, const float* q, const float* k, const float* v, const int dim, const int q_head, const int kv_head, const int _pos) {
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

void test_attn() {
    size_t seq_len = 200;
    size_t num_heads = 128;
    size_t embed_dim = 8192;
    size_t head_dim = embed_dim / num_heads;

    float* q = new float[seq_len * embed_dim * sizeof(float)];
    float* k = new float[seq_len * embed_dim * sizeof(float)];
    float* v = new float[seq_len * embed_dim * sizeof(float)];

    for(int i = 0; i < seq_len * embed_dim; i++) {
        q[i] = dist(gen);
        k[i] = dist(gen);
        v[i] = dist(gen);
    }

    float* o = new float[seq_len * embed_dim * sizeof(float)];
    float* o1= new float[seq_len * embed_dim * sizeof(float)];

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        masked_attention_cpu(o, q, k, v, nullptr, head_dim, num_heads, seq_len, seq_len);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Attention New Time taken: " << 1000 * duration.count() / iter_num << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        for(int p = 0; p < seq_len; p++) {
            masked_attention_origin(o1 + p*embed_dim, q + p*embed_dim, k , v, head_dim, num_heads, num_heads, p);
        }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Attention Origin Time taken: " << 1000 * duration.count() / iter_num << " ms" << std::endl;

    check(o, o1, seq_len * embed_dim, "Multihead-Attention");

    delete q;
    delete k;
    delete v;
    delete o;
    delete o1;
}

/**********************************************************
 ***********************************************************/

void matmul_cpu(float *y, const float *x, const float *w, int n, int d, int num) {
    for(int b = 0; b < num; b++){
        for (int i = 0; i < d; ++i) {
            double sum = 0.0f;
            for (int j = 0; j < n; ++j) {    
                sum += w[i * n + j] * x[b * n + j];
            }
            y[b*d + i] = sum;
        }
    }
}
// y = WX     W(W_in*W_out), X(W_in*num), C(W_out*num)  
void matmul(float *y, const float *X, const float *W, int W_in, int W_out, int num) {
    // 缩放因子
    float alpha = 1.0;
    float beta = 0.0;  // C 的初始权重

    // 调用 OpenBLAS 的 SGEMM 函数
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                W_out, num, W_in,         // 矩阵维度
                alpha,           // alpha
                W, W_in,            // 矩阵 W 和列主布局步长
                X, W_in,            // 矩阵 X 和列主布局步长
                beta,            // beta
                y, W_out);           // 结果矩阵 C 和列主布局步长
}

void test_matmul() {
    // 矩阵维度
    int W_in = 8192;   // 输入维度
    int W_out = 8192;  // 输出维度
    int num = 200;    // 样本数量

    // 分配和初始化矩阵
    std::vector<float> W(W_in * W_out);
    std::vector<float> X(W_in * num);
    std::vector<float> C_cpu(W_out * num, 0.0f);
    std::vector<float> C_openblas(W_out * num, 0.0f);

    // 初始化矩阵W和X
    for(auto &val : W) val = dist(gen);
    for(auto &val : X) val = dist(gen);

    // 预热，确保数据已加载到缓存
    matmul_cpu(C_cpu.data(), X.data(), W.data(), W_in, W_out, num);
    matmul(C_openblas.data(), X.data(), W.data(), W_in, W_out, num);
    std::cout << "warm up matmul finished." << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++)
        matmul_cpu(C_cpu.data(), X.data(), W.data(), W_in, W_out, num);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Matrix Origin Time taken: " << 1000 * duration.count() / 10 << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++)
        matmul(C_openblas.data(), X.data(), W.data(), W_in, W_out, num);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Matrix OPENBLAS Time taken: " << 1000 * duration.count() / 10 << " ms" << std::endl;

    check(C_openblas.data(), C_cpu.data(), W_out * num, "Matrix Multiply");
}

/**********************************************************
 ***********************************************************/

void silu_cpu(float *x, const int n, int batch_size){
    for(int i = 0; i < batch_size * n; i++) {
        x[i] = x[i] / (1 + std::exp(-x[i]));
    }
}

void silu(float *x, const int n, int batch_size) {
    #pragma omp parallel for
    for(int i = 0; i < batch_size * n; i++) {
        x[i] = x[i] / (1 + std::exp(-x[i]));
    }
}

void test_silu(int n, int batch_size) {
    float* x_cpu = new float[n * batch_size];
    float* x = new float[n * batch_size];

    // 初始化 x 和 w 数据
    for (int i = 0; i < n * batch_size; ++i) {
        x[i] = dist(gen);
        x_cpu[i] = x[i];
    }


    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        silu_cpu(x_cpu, n, batch_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "SiLU CPU version time: " << 1000 * duration.count() / iter_num << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        silu(x, n, batch_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "SiLU New version time: " << 1000 * duration.count() / iter_num << " ms\n";

    check(x_cpu, x, n * batch_size, "SiLU");

    delete x;
    delete x_cpu;
}

/**********************************************************
 ***********************************************************/
// 有 num 个 x 和 pos
// dim = 32, num = 6
void apply_rope_cpu(float *_x, const float *_pos, const float *_cos, const float *_sin, const int n, const int dim, const int num) {
    for(int p = 0; p < num; p++){   // 6
        const float* cos = _cos + (int)_pos[p] * dim;
        const float* sin = _sin + (int)_pos[p] * dim;
        for(int i = 0; i < n/(dim*2); i++) {
            float* x = _x + p*n + i*dim*2;
            for(int j = 0; j < dim; j++) {
                float x1 = x[j];
                float x2 = x[dim + j];
                x[j]       = x1 * cos[j] - x2 * sin[j];
                x[dim + j] = x2 * cos[j] + x1 * sin[j];
            }
        }
    }
}

void apply_rope_cpu_omp(
    float *__restrict x, 
    const float *__restrict pos, 
    const float *__restrict cosArr, 
    const float *__restrict sinArr, 
    const int n, 
    const int dim, 
    const int num)
{
    const int loop_count = n / (dim*2);

    #pragma omp parallel for
    for(int p = 0; p < num; p++){
        int pos_idx = static_cast<int>(pos[p]);
        const float* cptr = cosArr + pos_idx * dim;
        const float* sptr = sinArr + pos_idx * dim;

        float* xBase = x + p*n;

        for(int i = 0; i < loop_count; i++) {
            float* xptr = xBase + i * dim * 2;
            // 尝试让编译器自动向量化
            #pragma omp simd
            for(int j = 0; j < dim; j++) {
                float x1 = xptr[j];
                float x2 = xptr[j + dim];
                float c  = cptr[j];
                float s  = sptr[j];

                xptr[j]       = x1 * c - x2 * s;
                xptr[j + dim] = x2 * c + x1 * s;
            }
        }
    }
}



void test_rope() {
    // -----------------------------
    // 1. 定义测试规模
    // -----------------------------
    const int batch_size = 1;   // 虽然函数里没用到，但我们按需生成数据
    const int num = 5000;       // 对应函数参数
    const int n   = 8192;       // 对应函数参数
    const int dim = 32;         // 对应函数参数
    
    // 为简化，我们统一称 max_length=10000，用来存储 cos/sin 的预计算
    const int max_length = 5001;

    // -----------------------------
    // 2. 分配内存并生成测试数据
    // -----------------------------
    // x 大小: num * n (因为函数中使用 p*n 作为偏移)
    // 这里 batch_size=1，只需 num*n 即可。
    float* x = new float[num * n];
    float* x_omp = new float[num * n];
    // pos 大小: num (一个序列一个 pos)
    float* pos = new float[num];
    float* pos_omp = new float[num];
    // cos/sin 大小: max_length * dim
    float* cos_val = new float[max_length * dim];
    float* sin_val = new float[max_length * dim];
    float* cos_val_omp = new float[max_length * dim];
    float* sin_val_omp = new float[max_length * dim];

    // 填充 x
    for (size_t i = 0; i < num * n; i++) {
        x[i] = dist(gen);
        x_omp[i] = x[i];
    }
    // 填充 pos (int -> float 存储)
    for (int p = 0; p < num; p++) {
        pos[p] = p;
        pos_omp[p] = p;
    }

    // 填充 cos/sin
    // 这里给一个简单公式: cos_val[l*dim + j] = cos(l + j), sin_val[l*dim + j] = sin(l + j)
    // 当然你也可以用更符合RoPE需求的公式
    for(int l = 0; l < max_length; l++) {
        for(int j = 0; j < dim; j++) {
            cos_val[l*dim + j] = cosf(l * x[j]);
            sin_val[l*dim + j] = sinf(l * x[j]);

            cos_val_omp[l*dim + j] = cos_val[l*dim + j];
            sin_val_omp[l*dim + j] = sin_val[l*dim + j];
        }
    }

    // -----------------------------
    // 3. 调用函数并测试性能
    // -----------------------------
    // 先做一次干跑(预热)，让 cache 等机制就绪
    apply_rope_cpu(x, pos, cos_val, sin_val, n, dim, num);
    apply_rope_cpu_omp(x_omp, pos_omp, cos_val_omp, sin_val_omp, n, dim, num);

    // 正式计时
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++)
        apply_rope_cpu(x, pos, cos_val, sin_val, n, dim, num);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "RoPE CPU version time: " << 1000 * duration.count() / 10 << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++)
        apply_rope_cpu_omp(x_omp, pos_omp, cos_val_omp, sin_val_omp, n, dim, num);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "RoPE OMP CPU version time: " << 1000 * duration.count() / 10 << " ms\n";

    check(x_omp, x, num * n, "RoPE");

    delete[] x;
    delete[] x_omp;
    delete[] cos_val;
    delete[] sin_val;

    return ;
}

/**********************************************************
 ***********************************************************/

void elem_multiply_cpu(float* y, const float* x1, const float* x2, const int size) {
    for(int i = 0; i < size; i++) {
        y[i] = x1[i] * x2[i];
    }
}

void elem_multiply_omp(float* y, const float* x1, const float* x2, const int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        y[i] = x1[i] * x2[i];
    }
}


void test_multiply(size_t size) {
    
    // 分配对齐内存
    float* a = new float[size];
    float* b = new float[size];
    float* result1 = new float[size];
    float* result2 = new float[size];


    for (size_t i = 0; i < size; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    // 调用 SIMD 加法函数
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        elem_multiply_cpu(result1, b, a, size);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "MULTIPLY Time taken: " << 1000 * duration.count() / iter_num << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        elem_multiply_omp(result2, b, a, size);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "MULTIPLY OMP Time taken: " << 1000 * duration.count() / iter_num << " ms" << std::endl;

    check(result2, result1, size, "MULTIPLY");

    // 释放内存
    std::free(a);
    std::free(b);
    std::free(result1);
    std::free(result2);
}

/**********************************************************
 ***********************************************************/

void embedding_cpu(float* y, const float* x, const float* W, const int d, const int x_size) {
    for(int i = 0; i < x_size; i++) {
        int id = (int)x[i];
        memcpy(y + i * d, W + id * d, sizeof(float) * d);
    }
}

void embedding_omp(float* y, const float* x, const float* W, const int d, const int x_size) {
    #pragma omp parallel for
    for(int i = 0; i < x_size; i++) {
        int id = (int)x[i];
        memcpy(y + i * d, W + id * d, sizeof(float) * d);
    }
}

void test_embedding(size_t vocal_size, size_t hidden_size, size_t select) {
    float* embedding = new float[vocal_size * hidden_size];
    float* a = new float[select * hidden_size];
    float* b = new float[select * hidden_size];
    float* x = new float[select];

    for (size_t i = 0; i < vocal_size * hidden_size; ++i) {
        embedding[i] = dist(gen);
    }

    for (size_t i = 0; i < select; ++i) {
        x[i] = (int(dist(gen)*10000))%vocal_size;
    }

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        embedding_cpu(a, x, embedding, hidden_size, select);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "EMBEDDING Time taken: " << 1000 * duration.count() / iter_num << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iter_num; i++)
        embedding_omp(b, x, embedding, hidden_size, select);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "EMBEDDING OMP Time taken: " << 1000 * duration.count() / iter_num << " ms" << std::endl;

    check(a, b, select * hidden_size, "MULTIPLY");

    delete[] embedding;
    delete[] a;
    delete[] b;
    delete[] x;
}
