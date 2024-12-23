import time
import torch

def build_sin_cos_pos_emb(seq_len, head_dim, base=10000.0):
    """
    生成形状为 [seq_len, head_dim//2] 的 sin, cos 位置编码。
    假设 head_dim 是偶数。
    """
    half_dim = head_dim // 2  # 例如 head_dim=64 => half_dim=32

    # [0, 1, 2, ..., half_dim-1]
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    # 这里给出一种常见频率分布计算方式，也可根据需要调整
    freq_seq = base ** (-2 * (freq_seq // 2) / head_dim)

    # 位置 [0, 1, ..., seq_len-1]
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]

    # 广播到 [seq_len, half_dim]
    theta = pos * freq_seq  # [seq_len, half_dim]
    sin_emb = torch.sin(theta)
    cos_emb = torch.cos(theta)
    return sin_emb, cos_emb


def apply_rotary_pos_emb(q, k, sin_emb, cos_emb):
    """
    将 RoPE 位置编码应用于 Q, K （拆分后分别乘 sin、cos 再拼接）。
    q, k 的 shape: [batch_size, seq_len, num_heads, head_dim]
    sin_emb, cos_emb 的 shape: [seq_len, head_dim//2]
    """
    bsz, seq_len, n_heads, head_dim = q.shape
    half_dim = head_dim // 2  # 例如 64//2=32

    # 扩展 sin, cos 以便与 (bsz, seq_len, n_heads, half_dim) 广播
    # 结果：[1, seq_len, 1, half_dim]
    sin_emb = sin_emb.unsqueeze(0).unsqueeze(2)
    cos_emb = cos_emb.unsqueeze(0).unsqueeze(2)

    # 拆分 Q、K 的最后一维
    q1, q2 = q.chunk(2, dim=-1)  # q1,q2 各 [bsz, seq_len, n_heads, 32]
    k1, k2 = k.chunk(2, dim=-1)

    # RoPE 公式:
    #   q1_rot = q1*cos - q2*sin
    #   q2_rot = q1*sin + q2*cos
    #   (k 同理)
    q1_rot = q1 * cos_emb - q2 * sin_emb
    q2_rot = q1 * sin_emb + q2 * cos_emb
    k1_rot = k1 * cos_emb - k2 * sin_emb
    k2_rot = k1 * sin_emb + k2 * cos_emb

    # 合并回 [bsz, seq_len, n_heads, head_dim]
    q_rot = torch.cat([q1_rot, q2_rot], dim=-1)
    k_rot = torch.cat([k1_rot, k2_rot], dim=-1)

    return q_rot, k_rot


def rope_once(q, k, sin_emb, cos_emb):
    """
    封装：对给定 Q/K 做一次 RoPE 变换。
    仅用于快速计时，实际项目中你可能直接调用 apply_rotary_pos_emb。
    """
    return apply_rotary_pos_emb(q, k, sin_emb, cos_emb)


def measure_rope_performance_cpu_and_gpu(
    batch_size=1,
    seq_len=512,
    num_heads=8,
    head_dim=64,
    warmup_times=5,
    repeat_times=10
):
    """
    测试 RoPE 在 CPU 和 GPU 上的执行性能。
    
    参数：
      batch_size  : 批量大小 (默认=1)
      seq_len     : 序列长度
      num_heads   : 注意力头数
      head_dim    : 每个注意力头的维度（假设偶数）
      warmup_times: GPU预热次数
      repeat_times: 正式测试次数
    """

    if not torch.cuda.is_available():
        print("No GPU found! Only CPU is available.")
        return

    print("===== RoPE Performance Test =====")
    print(f"batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")

    # 1) 在 CPU 上生成数据
    # Q, K shape: [batch_size, seq_len, num_heads, head_dim]
    q_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    k_cpu = torch.randn_like(q_cpu)
    sin_emb_cpu, cos_emb_cpu = build_sin_cos_pos_emb(seq_len, head_dim)

    # 2) CPU 测试 (一次)
    start_time = time.time()
    for _ in range(repeat_times):
        rope_once(q_cpu, k_cpu, sin_emb_cpu, cos_emb_cpu)
    cpu_time = time.time() - start_time
    cpu_time = cpu_time / repeat_times

    # 3) 拷贝到 GPU
    q_gpu = q_cpu.to('cuda')
    k_gpu = k_cpu.to('cuda')
    sin_emb_gpu = sin_emb_cpu.to('cuda')
    cos_emb_gpu = cos_emb_cpu.to('cuda')

    # 4) GPU 预热
    for _ in range(warmup_times):
        rope_once(q_gpu, k_gpu, sin_emb_gpu, cos_emb_gpu)
    torch.cuda.synchronize()

    # 5) GPU 正式测试 (多次测量取平均)
    gpu_times = []
    for _ in range(repeat_times):
        start_time = time.time()
        rope_once(q_gpu, k_gpu, sin_emb_gpu, cos_emb_gpu)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - start_time)

    avg_gpu_time = sum(gpu_times) / len(gpu_times)

    # 6) 打印结果
    print(f"CPU Time (once): {cpu_time:.6f} s")
    print(f"GPU Time (avg over {repeat_times} runs): {avg_gpu_time:.6f} s")


if __name__ == "__main__":
    measure_rope_performance_cpu_and_gpu(
        batch_size=1,     # 你需要的 batch_size=1
        seq_len=5000,     # 可调
        num_heads=32,      # 可调
        head_dim=128,      # 可调(请保持偶数)
        warmup_times=5,   # GPU预热次数，可按需调整
        repeat_times=100   # 正式测试次数
    )
