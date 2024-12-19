import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 定义线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # 缩放因子
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, embed_dim = query.size()
        
        # 线性变换
        Q = self.q_linear(query)  # (batch_size, seq_len, embed_dim)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # 拆分多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            # mask shape: (batch_size, 1, 1, seq_len) 或者 (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # 应用注意力权重
        context = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # 最后的线性变换
        output = self.out_linear(context)  # (batch_size, seq_len, embed_dim)
        
        return output

def generate_subsequent_mask(seq_len):
    """
    生成一个下三角矩阵作为掩码，确保每个位置只能关注当前位置及之前的位置。
    """
    mask = torch.tril(torch.ones((seq_len, seq_len)))  # (seq_len, seq_len)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return mask  # 可根据需要扩展到(batch_size, 1, seq_len, seq_len)

def test_matrix_multiplication(device, iter_num=100, dim_a=(8192, 8192), dim_b=(8192, 200)):
    """
    测试矩阵乘法的性能。

    参数:
        device (torch.device): 运行设备（CPU 或 CUDA）。
        iter_num (int): 测试迭代次数。
        dim_a (tuple): 第一个矩阵的维度。
        dim_b (tuple): 第二个矩阵的维度。
    """
    print(f"\nTesting Matrix Multiplication on {device.type.upper()}")

    # 初始化矩阵
    A = torch.randn(dim_a, device=device)
    B = torch.randn(dim_b, device=device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            C = torch.matmul(A, B)

    # 同步 CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 测试时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iter_num):
            C = torch.matmul(A, B)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iter_num
    print(f"Matrix Multiplication: Total time for {iter_num} runs: {total_time:.4f} seconds")
    print(f"Matrix Multiplication: Average time per run: {avg_time:.6f} seconds")


def test_vector_addition(device, iter_num=100, vector_size=int(5e8)):
    """
    测试向量加法的性能。

    参数:
        device (torch.device): 运行设备（CPU 或 CUDA）。
        iter_num (int): 测试迭代次数。
        vector_size (int): 向量的大小。
    """
    print(f"\nTesting Vector Addition on {device.type.upper()}")

    # 初始化向量
    x = torch.randn(vector_size, device=device)
    y = torch.randn(vector_size, device=device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            x + y

    # 同步 CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 测试时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iter_num):
            x + y
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iter_num
    print(f"Vector Addition: Total time for {iter_num} runs: {total_time:.4f} seconds")
    print(f"Vector Addition: Average time per run: {avg_time:.6f} seconds")


def test_performance(iter_num=100, dim_a=(8192, 8192), dim_b=(8192, 200), vector_size=5 * 10**8, num_heads=128, num_vectors=5000):
    """
    测试带掩码多头注意力机制在 CPU 和 CUDA 上的性能。

    参数:
        iter_num (int): 测试迭代次数。
        dim_a (tuple): 第一个矩阵的维度。
        dim_b (tuple): 第二个矩阵的维度。
        vector_size (int): 向量的大小。
        num_heads (int): 多头数量。
        num_vectors (int): 要处理的向量数量（batch_size）。
    """
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n=== Testing on {device.type.upper()} ===")

        # 测试矩阵乘法
        test_matrix_multiplication(device, iter_num, dim_a, dim_b)
        
        # 测试向量加法
        test_vector_addition(device, iter_num, vector_size)


def main():
    iter_num = 100
    dim_a = (8192, 8192)
    dim_b = (8192, 200)
    vector_size = int(5e8)  # 500,000,000

    # 测试矩阵乘法和向量加法
    test_performance(iter_num=iter_num, dim_a=dim_a, dim_b=dim_b, vector_size=vector_size)


if __name__ == "__main__":
    # 开始测试
    print(5 * 10**8)
    main()
