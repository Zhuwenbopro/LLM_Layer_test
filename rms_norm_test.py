import torch
import torch.nn as nn
import time

def test_rmsnorm_performance(iter_num=100, dim=8192, num_vectors=5000):
    """
    测试 PyTorch RMSNorm 在 CPU 和 CUDA 上的性能。

    参数:
        iter_num (int): 测试迭代次数。
        dim (int): 向量的维度。
        num_vectors (int): 要处理的向量数量（batch_size）。
    """
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        print(f"\nTesting RMSNorm on {device.upper()}")
        
        # 设置设备
        dev = torch.device(device)
        
        # 初始化 RMSNorm
        rms_norm = nn.RMSNorm(dim).to(dev)
        
        # 生成随机输入
        # 形状为 (num_vectors, dim)
        input_tensor = torch.randn(num_vectors, dim).to(dev)
        
        # 设置为评估模式
        rms_norm.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                output = rms_norm(input_tensor)
        
        # 同步 CUDA
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 测试时间
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iter_num):
                output = rms_norm(input_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iter_num
        print(f"Total time for {iter_num} runs: {total_time:.6f} seconds")
        print(f"Average time per run: {avg_time:.6f} seconds")

if __name__ == "__main__":
    # 设置多线程数量（针对 CPU）
    torch.set_num_threads(8)  # 根据 CPU 核心数调整，如 8
    
    # 测试 RMSNorm 性能
    test_rmsnorm_performance(iter_num=100, dim=8192, num_vectors=5000)
