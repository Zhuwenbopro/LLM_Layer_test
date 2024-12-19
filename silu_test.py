import torch
import time

def test_silu_performance(device, data, num_iterations=100):
    if device.type == 'cuda':
        # 使用 CUDA 事件进行高精度计时
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        
        for _ in range(num_iterations):
            result = torch.nn.functional.silu(data)
        
        ender.record()
        torch.cuda.synchronize()
        elapsed_time_ms = starter.elapsed_time(ender)  # 毫秒
        return elapsed_time_ms
    else:
        # 对于 CPU，使用 time.time()
        start_time = time.time()
        for _ in range(num_iterations):
            result = torch.nn.functional.silu(data)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000  # 转换为毫秒
        return elapsed_time_ms

def main():
    # 配置参数
    num_vectors = 5000
    dimensions = 8192
    num_iterations = 100

    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    devices = [torch.device('cpu')]
    if cuda_available:
        devices.append(torch.device('cuda'))

    # 生成测试数据
    # 使用 float32 类型
    data_cpu = torch.randn(num_vectors, dimensions, dtype=torch.float32)
    
    results = {}
    for device in devices:
        if device.type == 'cuda':
            # 将数据移动到 GPU
            data = data_cpu.to(device)
        else:
            data = data_cpu

        # 预热（避免首次运行时的延迟影响结果）
        torch.nn.functional.silu(data)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 运行测试
        elapsed_time_ms = test_silu_performance(device, data, num_iterations)
        avg_time_ms = elapsed_time_ms / num_iterations
        results[device.type] = {
            'total_time_ms': elapsed_time_ms,
            'avg_time_ms_per_iteration': avg_time_ms
        }

    # 打印结果
    for device_type, timing in results.items():
        print(f"Device: {device_type.upper()}")
        print(f"  Total Time for {num_iterations} iterations: {timing['total_time_ms']:.3f} ms")
        print(f"  Average Time per iteration: {timing['avg_time_ms_per_iteration']:.6f} ms\n")

if __name__ == "__main__":
    main()
