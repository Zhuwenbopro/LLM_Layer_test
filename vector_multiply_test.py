import torch
import time

def measure_cpu_vs_gpu_vector_multiply(size=100_000_000, warmup_times=5, repeat_times=100):
    """
    测试在 CPU 和 GPU 上的向量逐位乘法性能。
    
    参数：
        size         : 向量的长度
        warmup_times : GPU 预热次数
        repeat_times : 正式测试次数
    """
    # 如果机器上没有 CUDA，则直接退出
    if not torch.cuda.is_available():
        print("No GPU found! Only CPU is available.")
        return
    
    # ====== 在 CPU 上生成随机张量 ======
    # 注意：也可以直接在 GPU 上生成，然后再拷回 CPU 进行对比
    x_cpu = torch.randn(size, dtype=torch.float32)
    y_cpu = torch.randn(size, dtype=torch.float32)
    
    # ====== CPU 测试 ======
    # 仅测算一次简单测试，如果需要更精确数据可以做循环取平均值
    start_time = time.time()
    z_cpu = x_cpu * y_cpu  # CPU 上元素级相乘
    cpu_time = time.time() - start_time
    
    # ====== 结果简单验证 ======
    # （这里仅是示例，实际使用中可根据需要进行更严格验证）
    print(f"CPU Result Check: {z_cpu[:5]} ...")

    # ====== 数据拷贝到 GPU ======
    # 也可以选择直接在 GPU 上生成 x_gpu, y_gpu，省去拷贝时间
    x_gpu = x_cpu.to("cuda")
    y_gpu = y_cpu.to("cuda")
    
    # ====== GPU 预热 ======
    # 在正式测试前多次运行以消除 GPU 首次调用开销
    for _ in range(warmup_times):
        _ = x_gpu * y_gpu  # 只做运算不计时
    # 同步等待预热计算完成
    torch.cuda.synchronize()

    # ====== GPU 测试 ======
    gpu_times = []
    for _ in range(repeat_times):
        start_time = time.time()
        z_gpu = x_gpu * y_gpu  # GPU 上元素级相乘
        # 强制同步，确保计算结束再计时
        torch.cuda.synchronize()
        gpu_times.append(time.time() - start_time)
    
    avg_gpu_time = sum(gpu_times) / len(gpu_times)
    
    # ====== GPU 结果拷贝回 CPU 并验证 ======
    z_gpu_cpu = z_gpu.to("cpu")
    print(f"GPU  Result Check: {z_gpu_cpu[:5]} ...")

    # ====== 打印结果 ======
    print(f"CPU Time (once)    : {cpu_time:.6f} seconds")
    print(f"GPU Time (average over {repeat_times} runs): {avg_gpu_time:.6f} seconds")


if __name__ == "__main__":
    measure_cpu_vs_gpu_vector_multiply()
