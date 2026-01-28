mainly reference: [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
# 再谈脉动阵列
> As compared to CPUs and GPUs, the single-threaded TPU has none of the sophisticated microarchitectural features that consume transistors and energy to improve the average case but not the 99th-percentile case: no caches, branch prediction, out-of-order execution, multiprocessing, speculative prefetching, address coalescing, multithreading, context switching and so forth. Minimalism is a virtue of domain-specific processors." (p.8)


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

这里以卷积计算为例，说明脉动阵列的工作机制：\

    
**为什么是二维阵列？**
上面提到的脉动阵列可以是多种结构形式，那么为什么Google TPU选择了二维阵列结构？\
我们可以结合Systolic论文和Google TPU论文中的描述来分析：\
在Systolic[1](#ref-1) 论文中提到：
>"When the memory speed is more than cell speed, two-dimensional systolic arrays such as those depicted in Figure 11 should be used. At each cell cycle, all the I/O ports on the array boundaries can input or output data items to or from the memory; as a result, the available memory bandwidth can be fully utilized. Thus, the choice of a one- or two-dimensional scheme is very depenc ent on how cells and memories will be imple-mented."

这里指出了这样的设计逻辑：
如果内存（Memory）读取数据的速度远快于单个处理单元（Cell）处理数据的速度，那么一维阵列（1D array）就无法消耗掉所有的带宽。通过使用二维阵列（2D array），可以增加阵列边界上 I/O 端口的数量。设二维阵列的大小为n×n，其周长（即边界 I/O 点）为 O(n)，这样就可以并行地从内存中泵入/泵出更多数据，从而实现对内存带宽的“充分利用”。

在Google TPU论文[2](#ref-2)中也提到：
> As reading a large SRAM uses much more power than arithmetic, the matrix unit uses systolic execution to save energy
by reducing reads and writes of the Unified Buffer. It relies on data from different directions
arriving at cells in an array at regular intervals where they are combined.

但是随着芯片领域的发展，以及AI计算任务引入的量化算法，内存带宽的增长速度相较于计算性能发生了明显的滞后。

|GPU 架构|计算能力 (TFLOPS)|内存带宽（GB/s）|计算/带宽增长比|
| ---|---|---|---|
|NVIDIA Kepler| 4.5	|288|	16x
|NVIDIA Pascal|	11	|484|	22x
|NVIDIA Ampere|	30	|768|	39x

因此，之前的2D阵列的原因不再适用。尽管如此，Google TPU仍然选择了二维阵列结构因为在AI场景下其仍有很大的优势：
在TPU中，读取SRAM的能耗远高于算术运算的能耗。在AI计算中大部分操作时矩阵乘法，通过使用脉动执行（Systolic execution），可以减少对统一缓冲区（Unified Buffer）的读写次数，从而节省能量消耗。二维阵列通过让数据从不同方向以规则的时间间隔到达阵列中的单元，并在单元中进行组合计算，实现了高效的数据流处理。


接下来介绍脉动阵列的多种运行方式：
1. 高吞吐量

![High-Throughput-array](/docs/pics/arch_deepdive/highThroughput.png)

权重驻留（Weight-Stay）：权重（Weights）预先加载并固定在每个处理单元（PE）中，不再移动。
数据流动：输入数据（Input）在 PE 阵列中像波浪一样线性穿过。
满载工作：所有处理单元在计算过程中始终处于忙碌状态，没有空闲（Idle）周期。
本地累加：中间计算结果在 PE 内部的寄存器中直接累加，直到完成最终计算。\
**优点：**
极高的利用率：硬件资源利用率接近 100%，单位时间内处理的数据量达到最大化。
节省功耗（高能效）：由于中间结果（通常位宽较长，如 32位或 64位）不需要在 PE 之间频繁搬运，大幅减少了数据传输的能量消耗。
无限扩展性：可以像搭积木一样增加 PE 数量来处理更大规模的计算，而不需要改变系统基础结构。
适应复杂计算：非常适合处理卷积核很大或权重数量极多的任务。

2. 广播式

![Broadcast-array](/docs/pics/arch_deepdive/broadcast-array.png)

全局广播：输入数据 x 通过一条公共总线，同时发送给阵列中的所有处理单元。
权重固定（Weight Stationary）：每个处理单元内部存储一个固定的权重值（如w1,w2,w3​），在计算过程中不移动。
结果移位：中间结果 y在处理单元之间逐级传递。每个时钟周期，单元接收左侧传来的值，加上自身的乘法结果 (w⋅x)，再传给右侧。

3. 低延迟

![Low-latency-array](/docs/pics/arch_deepdive/low-latency-array.png)

上方的图展示了一个基本的计算节点和由多个计算节点组成的阵列。对于一个计算节点，其具有双向的数据通道：x 序列从左向右流动，y 序列从右向左流动。
w：代表单元内部存储的固定值（通常是卷积核的权重）。
计算逻辑：每个单元在每个时钟周期执行一次乘累加（MAC）操作

在多个处理单元串联成阵列的配置中，x 序列（输入信号）和 y 序列（输出结果/中间值）反向流动。
由于数据在单元间同步推进（像心脏泵血一样，故名“Systolic”），它避免了全局广播（Broadcast）带来的长导线延迟问题，非常适合高频率的大规模集成电路（VLSI）。

但是由于 x 和 y 反向流动，为了使正确的 x_i 遇到正确的 y_j，通常需要在序列中插入间隔，这导致在任何给定时刻只有一半的单元在进行有效工作，从而降低了硬件利用率。


Google采用的是高吞吐量（High Throughput） + 权重驻留（Weight-Stay）的脉动阵列运行方式


> We did a very fast chip design. It was really quite remarkable. We started shipping the first silicon with no bug fixes or mask changes. Considering we were hiring the team as we were building the chip, then hiring RTL (circuitry design) people and rushing to hire design verification people, it was hectic.---from [First in-depth look at Google's TPU architecture](https://www.nextplatform.com/2017/04/05/first-depth-look-googles-tpu-architecture/), The Next Platform



## Reference
<a id="ref-1"></a>[1] [Why Systolic Architectures?](https://safari.ethz.ch/architecture_seminar/fall2018/lib/exe/fetch.php?media=why_systolic_architectures.pdf)\
<a id="ref-2"></a>[2] In-Datacenter Performance Analysis of a Tensor Processing Unit
