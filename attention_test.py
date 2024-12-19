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
    # 生成一个下三角矩阵作为掩码
    mask = torch.tril(torch.ones((seq_len, seq_len)))  # (seq_len, seq_len)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return mask  # 可根据需要扩展到(batch_size, 1, seq_len, seq_len)

def test_performance_cuda():
    # 设置参数
    seq_len = 200
    num_heads = 128
    embed_dim = 8192
    batch_size = 1  # 根据需要调整批量大小
    
    # 初始化模型
    model = MaskedMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads).cuda()
    
    # 生成随机输入
    query = torch.randn(batch_size, seq_len, embed_dim).cuda()
    key = torch.randn(batch_size, seq_len, embed_dim).cuda()
    value = torch.randn(batch_size, seq_len, embed_dim).cuda()
    
    # 生成掩码
    mask = generate_subsequent_mask(seq_len).cuda()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(query, key, value, mask)
    
    # 测试时间
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(query, key, value, mask)
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / 100
    print(f"Total time for 100 runs: {total_time:.4f} seconds")
    print(f"Average time per run: {avg_time:.6f} seconds")

def test_performance_cpu():
    # 设置参数
    seq_len = 200
    num_heads = 128
    embed_dim = 8192
    batch_size = 1  # 根据需要调整批量大小
    
    # 检查设备
    device = torch.device('cpu')
    
    # 初始化模型
    model = MaskedMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)
    
    # 生成随机输入
    query = torch.randn(batch_size, seq_len, embed_dim).to(device)
    key = torch.randn(batch_size, seq_len, embed_dim).to(device)
    value = torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    # 生成掩码
    mask = generate_subsequent_mask(seq_len).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(query, key, value, mask)
    
    # 测试时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(query, key, value, mask)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / 100
    print(f"Total time for 100 runs: {total_time:.4f} seconds")
    print(f"Average time per run: {avg_time:.6f} seconds")

if __name__ == "__main__":
    test_performance_cpu()
    test_performance_cuda()
