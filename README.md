# LLM_Layer_test
使用llama 3.2 1B 进行250轮迭代，GPU为笔记本4060（8GB），CPU为i7-13650HX。详细信息见下方。
## CPU (ms, 100 iter)
||origin|openblas/openomp|pytorch|
|:---:|:---:|:---:|:---:|
|矩阵乘 (8192, 8192) * (8192, 200)|7050|210|72.6|
|向量加 (5x10^8, )|85|104|140|
|RMSNorm (5000, 8192)|154|91|35|
|Multihead-Attention (1, 200, 8192)|70|72|312|
|Softmax (5000, 8192)|126|38|14|
|SiLU (5000, 8192)|81|23|13|
|**Embedding (128500, 8192)→(2000,8192)**|5.3|3.2|4.3|
|RoPE (5000, 8192)|10|6.2|54.7|
|逐位乘 (10^8)|43|25|35|

实际上仅使用 embedding 层的优化，效果从577s变为了169s。其余优化加入进去不会提升性能，反而会降低性能。pytorch (bfloat16) 用时154.8s，(float32)用时 29.4s。

## 设备信息
```
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Byte Order:                           Little Endian
Address sizes:                        39 bits physical, 48 bits virtual
CPU(s):                               8
On-line CPU(s) list:                  0-7
Thread(s) per core:                   2
Core(s) per socket:                   4
Socket(s):                            1
Vendor ID:                            GenuineIntel
CPU family:                           6
Model:                                183
Model name:                           13th Gen Intel(R) Core(TM) i7-13650HX
Stepping:                             1
CPU MHz:                              2803.199
BogoMIPS:                             5606.39
Virtualization:                       VT-x
Hypervisor vendor:                    Microsoft
Virtualization type:                  full
L1d cache:                            192 KiB
L1i cache:                            128 KiB
L2 cache:                             5 MiB
L3 cache:                             24 MiB

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 531.79       CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 L...    On | 00000000:01:00.0  On |                  N/A |
| N/A   46C    P8                2W /  N/A|    236MiB /  8188MiB |      2%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```
