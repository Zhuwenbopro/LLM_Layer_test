#include <chrono>
#include <cstring>
#include <random>
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <stdexcept>

// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> start = std::chrono::high_resolution_clock::now();
std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green

inline void check_pass(const std::string&  message){
    std::cout << GREEN << message << RESET << std::endl;
}

inline void check_error(const std::string&  message){
    std::cout << RED << message << RESET << std::endl;
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

class Device {
public:
    std::string dev;
    float _duration;

    virtual float* malloc(size_t size, bool autofill = false) = 0;
    virtual void free(float* ptr) = 0;
    virtual void copy(float* dst, float* src, size_t size) = 0;
    virtual void time_tick() = 0;
    virtual void time_stop() = 0;

    virtual ~Device() {}

    double duration() {
        return _duration;
    }

    void rand(float* a, size_t size) {
        for(int i = 0; i < size; i++) {
            a[i] = dist(gen);
        }
    }
};

class CPU : public Device {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> start;
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> end;
public:
    CPU() {
        dev = "cpu";
    }

    float* malloc(size_t size, bool autofill = false) override {
        float* ret = new float[size];
        if(autofill) {
            rand(ret, size);
        }
        return ret;
    }

    void free(float* ptr) override {
        delete[] ptr;
    }

    void copy(float* dst, float* src, size_t size) override {
        std::memcpy(dst, src, size*sizeof(float));
    }

    void time_tick() override {
        start = std::chrono::high_resolution_clock::now();
    }

    void time_stop() override {
        end = std::chrono::high_resolution_clock::now();
        auto _d = end - start;
        _duration = _d.count();
    }
 
};

class CUDA : public Device {
    cudaEvent_t start_kernel, stop_kernel;
public:
    CUDA() {
        dev = "cuda";
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);
    }

    ~CUDA() {
        CHECK_CUDA(cudaEventDestroy(start_kernel));
        CHECK_CUDA(cudaEventDestroy(stop_kernel));
    }

    float* malloc(size_t size, bool autofill = false) override {
        float* ret;
        CHECK_CUDA(cudaMalloc(&ret, size * sizeof(float)));
        if(autofill) {
            float* temp = new float[size];
            rand(temp, size);
            CHECK_CUDA(cudaMemcpy(ret, temp, size * sizeof(float), cudaMemcpyHostToDevice));
            delete[] temp;
        }
        return ret;
    }

    void free(float* ptr) override {
        CHECK_CUDA(cudaFree(ptr));
    }

    void copy(float* dst, float* src, size_t size) override {
        CHECK_CUDA(cudaMemcpy(dst, src, size*sizeof(float), cudaMemcpyDeviceToDevice));
    }

    void time_tick() override {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start_kernel));
    }

    void time_stop() override {
        CHECK_CUDA(cudaEventRecord(stop_kernel));
        CHECK_CUDA(cudaEventSynchronize(stop_kernel));
        CHECK_CUDA(cudaEventElapsedTime(&_duration, start_kernel, stop_kernel));
    }
};

class Test {
private:
    std::vector<float*> _list;
    std::string _device = "cpu";
    Device* dev;

public:
    Test() {
        dev = new CPU();
    }

    ~Test() {
        while(!_list.empty()) {
            float* removed = _list.back();
            _list.pop_back();
            dev->free(removed);
        }
    }

    void setDevice(const std::string& device) {
        delete dev;
        if(device == "cpu") {
            dev = new CPU();
        } else if(device == "cuda") {
            dev = new CUDA();
        } else {
            throw std::logic_error("Do not have device : " + _device);
        }
        _device = device;
    }

    float* getArr(size_t size, bool autofill = false) {
        float* ret = dev->malloc(size, autofill);
        _list.push_back(ret);
        return ret;
    }

    inline void copy(float* __restrict__ dst, float* __restrict__ src, size_t size) {
        dev->copy(dst, src, size);
    }

    void print(float* a, size_t col, size_t row = 1, const std::string& msg = "") {
        float *a1;
        size_t size = col*row;
        if(_device == "cpu") {
            a1 = a;
        } else if(_device == "cuda") {
            a1 = new float[size];
            CHECK_CUDA(cudaMemcpy(a1, a, size*sizeof(float), cudaMemcpyDeviceToHost));
        }

        std::cout << msg << std::endl;
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                std::cout << a1[i*col+j] << " ";
            }
            std::cout << std::endl;
        }

        if(_device == "cuda") {
            delete[] a1;
        }
    }

    void check(float*a , float* b, size_t size, const std::string& msg, float epsilion = 5e-2) {
        float *a1, *a2;
        if(_device == "cpu") {
            a1 = a;
            a2 = b;
        } else if(_device == "cuda") {
            a1 = new float[size];
            a2 = new float[size];
            CHECK_CUDA(cudaMemcpy(a1, a, size*sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(a2, b, size*sizeof(float), cudaMemcpyDeviceToHost));
        }

        // 检查两个版本的计算结果是否一致
        bool results_match = true;
        for (int i = 0; i < size; ++i) {
            if (std::fabs(a1[i] - a2[i]) > 1e-3f) {
                results_match = false;
                std::cout << "different at " << i << " :  " << a1[i] << " vs " << a2[i] << std::endl;
                break;
            }
        }

        // 输出是否一致
        if (results_match) {
            check_pass("["+msg+"] The results are consistent.\n");
        } else {
            check_error("["+msg+"] The results are NOT consistent!!!\n");
        }

        if(_device == "cuda") {
            delete[] a1;
            delete[] a2;
        }
    }

    void start_timing() {
        dev->time_tick();
    }

    void end_timing() {
        dev->time_stop();
    }

    float duration() {
        return dev->duration();
    }
};