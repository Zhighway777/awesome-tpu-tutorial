# Background TPUv1~v3 revealed
这部分聚焦于TPUv4之前的架构演变进展，目的是快速recap一下TPU架构的设计思路和设计路线，为TPUv4以及之后的架构理解做铺垫。

(这部分主要参考资料 [Google’s Training Chips Revealed:TPUv2 and TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)以及[The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf))

TPU的架构转变是根据需求侧的转变发展而来。从2014年至2017年间Google的AI需求从数据占主导逐步变为推理和训练同时占主导。如下图所示：
![alt text](../pics/arch_deepdive/2014.png)
![alt text](../pics/arch_deepdive/2015.png)
![alt text](../pics/arch_deepdive/2017.png)

由于训练需求的增加，引入了一些列挑战：相较于推理负载训练需要更多类型的计算算子，包括backprop，transpose，derivatives等操作；此外由于要存储反向转播的data因而需要更多的Memory；同时，更广泛的operands的精度支持也是必要的；为了在训练过程中的不断实验和optimiztors的调整，灵活性也变得更加重要。为了训练更大的模型，需要进一步提升scale up而非scale out。因为后者的提升已经很难有更好地收益。

为了满足以上要求，Google在最原始的TPUv1架构上进行了多方面的改进，最终形成了TPUv2和TPUv3架构。

**TPUv1** 
![TPUv1-PCB](../pics/arch_tracking/TPUv1-PCB.png)
TPUv1的组件如下所示：

![TPUv1-1](../pics/arch_deepdive/TPUv1-1.png)
下面的三张图显示了**TPUv2**的Vector处理相关部分的改动：\
将累加器部分和Activation使用Vector Memory代替，同时将Activation Pipeline改为Vector Unit。从而支持更广泛的Vector操作和更快速灵活的访问模式。

1. TPUv2将Activation Storage和Accumulators放在了一起，引入了“Single vector memory instead of buffers between fixed function units”（单一向量内存，取代了固定功能单元之间的缓冲区）。这意味着这个向量内存被设计来为向量单元提供更灵活和高效的数据存储和访问。

![alt text](../pics/arch_deepdive/TPUv2-1.png)

![alt text](../pics/arch_deepdive/TPUv2-2.png)
2. TPUv2引入了Vector Unit来代替Activation Pipeline。首先这是因为训练阶段需要广泛的使用Vector operand，如激活函数，归一化，数据预处理以及计算图中可能出现的向量操作等。其次为了增加可编程性，不再使用固定的Activation pipeline管线，使得开发者可以支持更多种类的向量操作。同时，这也是Vector Memory系统的设计原因。为了配合这个更通用的向量单元，需要一个更灵活的内存子系统。向量内存Vector Memory就是为此而设计的。

![alt text](../pics/arch_deepdive/TPUv2-3.png)
针对Matrix Multiply Unit在架构层面的改动，原本MXU直接从存储单元中获取数据，数据通路被修改为直接从Vector Unit offload数据。\
在TPUv1中MXU主要负责矩阵运算，其输出后的结果会传递给后续的Accumulators和Activation Pipeline，最后存储到Activation Storage中，以便后续的使用，这样的链接是一种P-C生产者消费者关系，主要用于数据的存储和检索。但是在支持的算子类型灵活度上以及运行效率上并不是最佳。\
TPUv2中MXU的数据通路被修改为直接从Vector Unit offload数据，**作为协处理器连接到向量单元**。这样的设计有以下几个好处：
1. 与Vector Unit协同工作：MXU的计算结果可以直接交给Vector Unit进行后处理，Vector Unit的计算结果也可以输入到MXU中进行计算，这种紧密的协同允许在单指令周期中完成他们的工作
2. 通用的Vector Unit可以支持更加通用的激活函数和矩阵向量乘法操作
3. 在Vector Unit和MXU中减少了不必要的内存读写，提高了数据吞吐量
4. 编译器友好，TPUv2采用了 VLIW（Very Long Instruction Word）架构。在这种架构下，编译器负责将多个独立指令打包成一个 VLIW 指令，从而并行执行包内的指令，编译器可以排布好Vector Unit和MXU的计算顺序将他们的工作打包在一个VLIW指令中，从而提高整体的执行效率。(TODO: 这一点需要确认)


![alt text](../pics/arch_deepdive/TPUv2-4.png)
在推理中将只读参数加载到 MXU 的方式并不适用于训练。 训练需要写入这些参数，并且需要大量的缓冲空间来存储每步的临时变量。 因此，DDR3 被移至向量存储器之后，使两者形成存储层级结构，并在后续的版本使用HBM来代替DDR3。HBM的带宽增加了20倍这直观地解决了ML训练的内存容量和带宽问题。同时这样的设计也使得存储系统更加灵活，能够支持复杂的算子所带来的各种数据传输问题。

![alt text](../pics/arch_deepdive/TPUv2-5.png)

![alt text](../pics/arch_deepdive/TPUv2-6.png)

TPUv2引入了定制的片间互连(InterChip Interconnect, ICI)模块，使得芯片从Scale Up变为了Scale Out。Interconnect直接和HBM以及Vector Memory相连，使得片间通讯有着更高的带宽和更低的延迟。

![alt text](../pics/arch_deepdive/TPUv2-7.png)
从这里可以看到Interconnect模块既可以作为片上的多个核心之间的互联，通过这些互连，可以形成庞大的分布式计算网络，使得谷歌能够训练极其庞大和复杂的机器学习模型。
这里TPUv2采用双核配置。因为构建一个更大的，单一的核心的连线延迟会大大增加，而双核配置在合理的流水线延迟与额外的单芯片计算能力之间达到了理想的平衡。不使用更多的核心是处于便于编程的考虑。
>  " We took advantage of training fundamentally being a big-data problem. "

![alt text](../pics/arch_deepdive/TPUv2.png)


TPUv2到TPUv3的改动主要通过“边际效应”来提升。

计算能力翻倍：MXU 数量加倍，使最大 FLOPs/秒翻倍。
![alt text](../pics/arch_deepdive/TPUv3-arch-1.png)
时钟频率提升：从 700 MHz 提高到 940 MHz，性能提升30% 
![alt text](../pics/arch_deepdive/TPUv3-arch-2.png)
HBM 性能与容量提升：HBM 带宽提升30%，容量翻倍，支持更大的模型和批次大小 
![alt text](../pics/arch_deepdive/TPUv3-arch-3.png)
互联带宽提升：链接带宽提升30%至 650 Gbps/链接

![alt text](../pics/arch_deepdive/TPUv3-arch-4.png)
系统规模扩展：最大系统规模从 TPUv2 的 256 芯片扩展到 1024 芯片
![alt text](../pics/arch_deepdive/TPUv3-arch-5.png)
下图是使用Roofline模型对比TPUv2和TPUv3的性能提升以及与当时Nvidia最先及的显卡V100进行对比。
- TPUv3有着更小的面积和较落后的nm工艺在 MLPerf 0.6 基准测试中，TPUv3 的单芯片性能几何平均值与 NVIDIA V100 相当。
- TPUv3 相较于 TPUv2 的峰值计算性能提升了 2.7 倍，尽管内存带宽、ICI 带宽和时钟频率的提升仅为 1.3 倍，这表明额外的 MXU 利用率高且面积贡献小
- TPUv3 超级计算机在运行 Google 应用时，GigaFLOPs/Watt 效率比运行 Linpack 基准测试的通用超级计算机高出50倍
![alt text](../pics/arch_deepdive/Roofline_Models.png)

## Reference
<a id="ref-1"></a>[1] [Google’s Training Chips Revealed:TPUv2 and TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)\
<a id="ref-2"></a>[2] [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)