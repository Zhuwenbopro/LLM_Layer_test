import torch
import time

# 设置参数
batch_size = 5000  # 向量个数
embed_dim = 8192  # 向量维度
num_repeats = 100  # 重复次数

# 创建随机输入数据
input_cpu = torch.randn(batch_size, embed_dim)  # 在 CPU 上创建输入数据
input_cuda = input_cpu.cuda()  # 在 CUDA 上创建输入数据

# 测试在 CPU 上的 softmax 性能
start_time_cpu = time.time()
for _ in range(num_repeats):
    softmax_cpu = torch.softmax(input_cpu, dim=-1)
end_time_cpu = time.time()
cpu_time = (end_time_cpu - start_time_cpu) / num_repeats
print(f"CPU Softmax 每次平均耗时: {cpu_time:.6f}秒")

# 测试在 CUDA 上的 softmax 性能
start_time_cuda = time.time()
for _ in range(num_repeats):
    softmax_cuda = torch.softmax(input_cuda, dim=-1)
end_time_cuda = time.time()
cuda_time = (end_time_cuda - start_time_cuda) / num_repeats
print(f"CUDA Softmax 每次平均耗时: {cuda_time:.6f}秒")
