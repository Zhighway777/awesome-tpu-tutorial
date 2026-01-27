mainly reference: [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
# 再谈脉动阵列

> We did a very fast chip design. It was really quite remarkable. We started shipping the first silicon with no bug fixes or mask changes. Considering we were hiring the team as we were building the chip, then hiring RTL (circuitry design) people and rushing to hire design verification people, it was hectic.---from [First in-depth look at Google's TPU architecture](https://www.nextplatform.com/2017/04/05/first-depth-look-googles-tpu-architecture/), The Next Platform

Google的TPU架构

## 脉动阵列的设计理念
脉动阵列（Systolic Array）是一种特殊的并行计算架构，最早由H.T. Kung在1970年代提出[1](#ref-1)。其核心思想是通过大量简单处理单元的协同工作，实现高效的数据流处理。脉动阵列的设计理念包括以下几个关键点：
1. Simple and regular （简单且规则）
2. High concurrency（高并发）
3. Simple communication（简单的通信）
4. Balance of computation with I/O（计算与I/O的平衡）

在传统的CPU或者GPU架构中，I/O带宽往往成为性能瓶颈。例如，10 MB/s的I/O带宽和2字节的操作数，最多只能支持每秒500万次操作。
脉动架构（Systolic approach）提出了一种革命性的数据流动方式，其核心理念是内存像心脏一样“泵送”数据到处理单元（Processing Elements, PEs），而数据和部分结果则像血液一样在相邻PE之间流动。

![cpu_gpu_io_bottleneck](/docs/pics/arch_deepdive/cpu_gpu_io_bottleneck.png)

具体而言，现代多层次的存储层级下，脉动阵列往往和Register级别的存储层级进行交互：
![micro dataflow](/docs/pics/arch_deepdive/micro_dataflow_CPU_GPUvsTPU.png)

**脉动架构的结构与机制** \
一个脉动架构由多个处理单元（cells）组成。只有位于系统边界的单元才能作为I/O端口。中间单元之间进行局部通信，交换部分结果和输入。虽然目前常见的脉动阵列是二维结构，但实际上可以是树形结构或者不规则阵列的结构等。
![systolic_array_structure](/docs/pics/arch_deepdive/systolic_array_structure.png)

虽然脉动阵列现在多用于计算矩阵乘法任务，但通过不同的设计，脉动阵列也可以适用于任何“规则的计算密集型问题”，即对大型数据集执行重复计算的问题。如信号与图像处理： FIR、IIR 滤波、一维卷积、二维卷积与相关、离散傅里叶变换（DFT）、插值、中值滤波、几何变形等。

矩阵算术： 矩阵-向量乘法、矩阵-矩阵乘法、矩阵三角化（线性系统求解、矩阵求逆）、QR 分解（特征值、最小二乘计算）以及三角线性系统求解等。

非数值应用： 数据结构（栈、队列、搜索、优先级队列、排序）、图算法（传递闭包、最小生成树、连通分量）、语言识别（字符串匹配、正则表达式）、动态规划、编码器和关系数据库操作等。


    

> As compared to CPUs and GPUs, the single-threaded TPU has none of the sophisticated microarchitectural features that consume transistors and energy to improve the average case but not the 99th-percentile case: no caches, branch prediction, out-of-order execution, multiprocessing, speculative prefetching, address coalescing, multithreading, context switching and so forth. Minimalism is a virtue of domain-specific processors." (p.8)




## Reference
<a id="ref-1"></a>[1] [Why Systolic Architectures?](https://safari.ethz.ch/architecture_seminar/fall2018/lib/exe/fetch.php?media=why_systolic_architectures.pdf)

