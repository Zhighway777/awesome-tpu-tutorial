# From TPUv3 to TPUv4i

## Background
在开始介绍TPUv4i之前，我们先做一个warm up: 

在上一节[TPUv1~v3 revealed](TPUv1~v3_revealed.md)中我们介绍了TPUv1到TPUv3的架构演变过程。TPUv1到TPUv3的设计主要是为了应对从推理为主到训练和推理并重的需求转变。TPUv1主要解决了推理任务(当年推理任务还被称为serving任务)TPUv1使用Activation Pipeline的固定单元和64K个8bit的MAC组成的MXU作为计算模块，使用DDR3作为存储模块，Activation Storage和Accumulators作为中间数据存储模块。CPU通过PCIe总线和TPU交换数据名且发送指令。TPUv1的设计目标是高效能的推理计算，同时考虑到大规模部署的成本和功耗问题。

TPUv2的设计目标转向为更难的训练任务，这在并行化训练(训练需要并行协调资源，而推理不需要)，计算能力(涉及到反向传播的求导计算)和更高的内存需求（权重更新需要访问前向和反向传播的中间值），以及更高的可编程性（训练算法和模型在不断变化）和最后需要同时支持INT型和浮点型计算。因此将Activation Storage和Accumulators合并为Vector Memory，使用更通用的Vector Unit替代Activation Pipeline，增强了可编程性。需要补充的是Bf16更适合DNN，MXU的bf16也是从TPUv2开始支持的。由于训练的目的就是设定权重值，而且需要大量的缓冲空间来存储中间变量。使用HBM DRAM来作为Vector Memory的后备，二者也构成了编译器可控的存储层次结构。在指令获取方面TPUv2从本地的存储器获取自己的322bit的VLIW指令，摆脱了对CPU的依赖。由于训练需要Scale Out，TPUv2引入了定制的片间互连(InterChip Interconnect, ICI)模块，可以构建256个芯片组成的pod。

TPUv3是TPUv2的“年上版本”，通过提升特定的参数来大幅提升芯片性能，这里不再赘述，感兴趣的读者可以参考上一节内容。
TPUv1~TPUv3的变化可以用下图来总结[3](#ref-3)：
![TPU-arch-evolution](../pics/arch_deepdive/TPU-arch-evolution.png)



## 指导TPUv4i设计的十个经验教训
Google 从TPUv3到TPUv4i的设计范式有了巨大的转变，论文“Ten Lessons From Three Generations Shaped Google’s TPUv4i Industrial Product”[3](#ref-3)中总结了十个设计经验，这些经验指导了TPUv4i的设计方向。在本文中将集中介绍TPUv4i的架构设计，并且和TPUv3进行对比，帮助读者理解TPUv4i的设计理念。关于TPUv1~TPUv4i的Feature对比可以参考下表[3](#ref-3)：

![Key-of-DSA](../pics/arch_deepdive/Key_characteristics_of_DSAs.png)

### 架构设计理念
TPUv4i版本的设计是Google对前三代TPU的总结，并且用“Ten Lessons”作为指导原则来设计TPUv4i。以下是这十个设计经验的简要总结：
> 1. 逻辑、导线、SRAM 和 DRAM 的改进速度不均衡 
> 2. 芯片和编译器应该协同设计，并利用现有的编译器优化
> 3. 以总拥有成本（TCO）而非初始资本支出（CapEx）来设计
> 4. 支持向后机器学习（ML）的编译器兼容性，而非二进制兼容性
> 5. 推理 DSA 需要气冷以实现全球规模部署
> 6. 某些推理应用需要浮点算术
> 7. 生产推理通常需要多租户支持
> 8. DNN 每年增长约 1.5 倍（内存和计算量）
> 9. DNN 工作负载随 DNN架构的突破而不断演变
> 10. 推理 SLO 应该是是 P99 延迟，而非批量大小

在上面10条Lessons中 1,6,7,8,9很好理解，是比较直观的事实或者应用需求，这里重点介绍剩余的5个Lessons：\
**Lesson 2: 芯片和编译器应该协同设计，并利用现有的编译器优化**\
DSA芯片的设计需要和编译器以及软件紧密配合，因为编译器能够将高层的计算图映射到硬件上，并且进行各种优化，最大化发挥硬件的潜力。新的硬件特性需要编译器来有效地利用，而开发一套全新的、高度优化的编译器是一个巨大的工程。\
因此， 新的DSA设计应该尽可能地利用现有的编译器优化技术。这样便无需从头构建软件栈，可以快速获得性能优势。

**Lesson 3: 以总拥有成本（TCO）而非初始资本支出（CapEx）来设计**\
初始资本支出（CapEx）指的是购买硬件设备的初始成本。\
拥有成本(TCO)：包括初始资本支出（CapEx）和运营支出（OpEx）。
运营支出（OpEx）：包括维护、能源消耗、冷却和空间等持续成本。

通常而言，CPU和GPU通常用其发布时的基准测试性能与购买价格之比来作为其价值的衡量标准，然而随着数据中心规模的扩大，运营支出（OpEx）在总拥有成本（TCO）中占据了更大的比例。比如，下图展示了TCO的组成部分[5](#ref-5)：\
![TCO-composition](../pics/arch_deepdive/TCO-composition.png)

在计算时，会将CapEX根据设备的使用寿命进行摊销(一般是3-5年)，对与3年的使用寿命来讲\
**TCO = CapEx + 3 ✕ OpEx**\
因此，Google 和大多数公司更关心生产环境应用程序的性能/总拥有成本（performance/TCO，简称 perf/TCO），而非基准测试的原始性能或性能/资本支出（performance/CapEx，简称 perf/CapEx）。\
在设计DSA的架构时，架构师不应该只考虑设计之初的TCO，而是需要为后续的DNN负载的变化留有余量(headroom),同时考虑长期的TCO。
但是TCO的信息一般是保密的，因为其设计到一些商业定价的信息。尽管如此架构师仍可以根据TDP来对TCO进行估算。下图展示了TPU中TDP和TCO的相关关系：
![TDP-vs-TCO](../pics/arch_deepdive/TDP-vs-TCO.png)

**Lesson 4: 支持向后机器学习（ML）的编译器兼容性，而非二进制兼容性**\
首先先说明什么是向后的ML兼容：

向后 ML 兼容性是为了让机器学习模型能够无缝地从一个 TPU 版本迁移到另一个版本。这中间需要机器学习编译器的参与，编译器会分析模型结构并得出对应的计算图，将其映射到硬件上执行，在执行时进行的计算是有次序的。\
如果下一代TPU不支持相同的数值精度，编译器就没有足够的硬件能力来直接、精确地执行模型所需的计算。它无法保证计算结果与旧 TPU 保持一致，也无法提供可预测的性能。\
因此，TPUv4i继续支持TPUv3的数值格式，包括bfloat16和int8。这种设计选择确保了向后 ML 兼容性，使得现有模型能够无缝迁移到新的硬件平台上。

接下来介绍为什么需要编译器兼容而非二进制兼容：\
二进制兼容，意味着旧软件可以直接在新硬件上运行，而无需重新编译。这在传统 CPU 架构中很常见。\
编译器兼容性则意味着硬件的指令集（ISA）可能发生变化，但只要编译器能够适应这些变化（例如，通过重新编译），软件就能在新硬件上运行。这对于 VLIW 这样的架构尤为重要。\
编译器兼容性使得硬件设计者和软件开发者可以持续地协同工作。硬件设计者可以专注于提升核心计算能力和架构效率，而编译器开发者则负责将这些优势转化为软件性能。

**Lesson 5: 推理 DSA 需要风冷以实现全球规模部署**\
TPUv3使用液冷系统来管理其高功耗和热量。然而，液冷需要占用更大的空间，增加了数据中心的成本。下图展示了液冷和风冷所需空间大小的区别：
![liquid-vs-air-cooling](../pics/arch_deepdive/liquid-vs-air-cooling.png)

对于模型的训练来讲，液冷是可以接收的，因为训练任务不需要广泛部署。然而面向用户的推理场景则不同，因为低用户延迟需要全球范围的覆盖，一些数据中心的建设已经拥挤不堪，难以容纳液冷系统。为了简化部署并降低成本，TPUv4i设计为风冷系统。
> 注释：这篇文章发布于2021年，目前液冷技术也在不断发展，目前先进的液冷技术可能已经极大的缩减了占地空间，使得液冷系统也可以适用于推理场景。这需要进一步的调研。    

**Lesson 10: 推理 SLO 应该是 P99 延迟，而非批量大小**\
服务水平目标(SLO)是指服务提供者承诺的服务质量标准。对于推理任务来说，SLO通常以延迟而非batch size为衡量标准。因为在推理场景下，对于用户体验而言，延迟才是是关键指标。
1. 硬件设计和优化应该着重于降低单个请求的响应时间（延迟），而不是仅仅追求一次处理大量请求的能力（批处理大小）。
2. P99 延迟，即系统中99%的请求的响应时间都必须低于某个阈值。
3. 尽管增加批处理大小可以提高整体吞吐量（单位时间内处理的请求数），但它可能增加单个请求的延迟。
4. batch size的定义是在一次前向传播中处理的输入样本数量。较大的batch size可以提高硬件利用率和吞吐量，但会增加每个请求的等待时间，从而影响延迟表现。


下面这张图可以直观地看出Google的10个Lessons和TPUv4i设计之间的关系[4](#ref-4)：
![TPUv4i-lessons-design](../pics/arch_deepdive/TPUv4i-lessons-design.png)

## TPUv4i架构
这里i指的是inference，没错，TPUv4i是专门为推理任务设计的芯片。在文章[3](#ref-3)中用“2 birds with 1 stone” 来形容TPUv4i的设计目标：一石二鸟。使用同样的芯片核心和非核心的缩放部分，来构成单核的推理芯片和双核的训练芯片。单核TPUv4i作为推理芯片，双核TPUv4用作训练。\
下图是TPUv4i的架构图和layout分布图 [3](#ref-3)：

![TPUv4i-architecture](../pics/arch_deepdive/TPUv4i-architecture.png)
![TPUv4i-layout](../pics/arch_deepdive/TPUv4i-layout.png)

### CMEM
对应绿色部分的CMEM

在TPUv4i中变化最大的是存储系统部分，引入了128MB的CMEM(Common Memory)作为中间缓存部分。这一引入是考虑到perf/TCO和perf/CapEx的trade-off。虽然SRAM的单位成本高于DRAM，但是SRAM的访问延迟和带宽远优于DRAM。通过在芯片上集成更大的SRAM，可以减少对DRAM的访问频率，尽管这增加了CapEx，但是却降低整体的TCO。\

### 4D tensor DMA
对应于红色部分的MGR

在TPUv4i中引入了4D tensor的DMA引擎，相较于TPUv2和TPUv3的2D tensor DMA引擎，支持更多的张量维度和复杂操作。在详细介绍之前，先回顾一下CPU和GPU的DMA引擎以及TPUv2和TPUv3的DMA引擎。\
CPU和GPU的DMA引擎主要用于在主存储器和设备之间传输数据。对CPU来讲，设备主要是网卡，存储控制器，侧重于通过块传输增加吞吐量，CPU DMA的通用性强，但是缺少对张量特定操作的内置支持。\
GPU的DMA引擎主要用于在主存储器和GPU显存之间传输数据（常被称为Transfer Engine 或 Copy Engine），侧重于高吞吐量和低延迟。GPU DMA引擎通常支持二维的内存块复制，可以与GPU计算核心协同工作，然而GPU DMA主要围绕块传输和一维内存地址的优化，缺少对多维张量操作的直接支持。\
TPUv2和TPUv3的DMA引擎支持2D tensor的传输，能够处理二维张量的数据布局和传输需求。TPUv4i的4D tensor DMA引擎进一步扩展了这一能力，支持四维张量的传输和操作。

首先4D DMA具有强大的灵活性和可编程性：
- ①支持每一步的任意步长（steps-per-stride）以及每个维度上的正负步长（positive/negative stride distances）
- ②源端和目标端的步长参数可以独立编程

在设置带宽时，与内存系统和向量单元协同：
- ③512B 字保证高效的 HBM 访问和互连设计
- ④原生 512B 字大小，与 TPUv2/v3 中继承下来的 128 通道 32 位矢量单元（vector unit）相匹配

在DMA的数据传输操作和内存访问策略上做了取舍：
- ⑤保留松散的DMA排序模型，这意味这无论是一个DMA内部还是不同DMA请求之间的操作在执行时可以是无序的，这依赖于编译器对其进行显示同步
- ⑥统一的DMA架构，芯片内部，不同芯片之间以及芯片与主机之间的DMA架构是一致的
- ⑦设置特定的内存访问机制，CMEM可以通过Load/Store指令和DMA进行访问，而HBM只能通过DMA进行访问（HBM的特点时带宽高打是延迟低，CMEM相反，因此HBM适合使用DMA进行大批量数据存取）
- ⑧支持DMA和TensorCore同步工作，可以隐藏DMA的启动(ramp-up)和停止延迟(ramp-down)

这样一来便可以支持以下特点：
- 支持复杂的数据layout转换，如gather，scatter， reshape等reshape的操作 ①②
- 在内存之直接执行，提供高效的数据准备和预取，减少TensorCore的数据传输负担 ①②
- 优化和CMEM以及HBM之间的数据交互，充分利用 CMEM 的高带宽和 HBM 的大容量 ③④⑦
- 降低了延迟并且提升吞吐，从而提升SLO ⑤⑥⑦⑧


### 片上互联OCI
对应于橙色的框图互联部分

OCI(On Chip Interconnect)是TPUv4i中新增硬件。以往TPU设计中的每个组件都是点对点连接的。例如，在TPUv3中，一个 TensorCore 只能访问HBM的一半作为本地内存：必须通过ICI才能访问 HBM 的另一半。这种分割限制了软件未来使用芯片的方式，OCI代替原本点到点的链接，它连接了芯片上的所有组件，可以根据现有的组件来扩展其拓扑结构。CMEM的引入使得OCI变得更加重要，后续仍然需要不断优化HBM，CMEM和VMEM之间的分配和数据传输。

为了支持更大的DNN负载，在访存层面通过**物理分区和创建独立的访存通道**来提高对于HBM的访问。TPUv4i设计的数据通路为512B，借助NUMA的思想，将HBM、CMEM和VMEM这样的高带宽内存划分为多个独立的128B宽的组，为每个组提供**独立的OCI连接**


### MXU计算单元
这部分的改进来自于三个方面：1. 为了ML的先后兼容性，保留TPUv3对fp16，bf16的支持和TPUv1对int8量化的支持。  2. 由于技术节点工艺的提升，将MXU的数量相较于TPUv3提升了一倍。 3. 优化脉动阵列的计算方式，将关键路径的延迟缩短到基准方法的1/4。


最后回顾一下10条经验是如何塑造TPUv4i 的设计：

① 逻辑电路的改进速度快于布线和 SRAM ⇒ TPUv4i 每个核心拥有 4 个 MXU，而 TPUv3 为 2 个，TPUv1/v2 为 1 个。 \
② 利用现有的编译器优化 ⇒ TPUv4i 是从 TPUv3 演进而来，而非全新的指令集架构 (ISA)。\
③ 为性能/总拥有成本 (perf/TCO) 而设计，而非性能/资本支出 (perf/CapEx) ⇒ 热设计功耗 (TDP) 较低，CMEM/HBM 速度快，且芯片面积不大。 \
④ 机器学习向后兼容性支持已训练 DNN 的快速部署 ⇒ TPUv4i 支持 bf16，并且从 XLA 编译器的角度来看其行为类似于 TPUv3，从而避免了算术问题。 \
⑤ 推理 DSA 需要风冷以实现全球规模部署 ⇒ 其设计和 1.0GHz 的时钟频率将其 TDP 降低至175W。 \
⑥ 某些推理应用需要浮点运算 ⇒ 它支持 bf16 和 int8，因此量化是可选的。 \
⑦ 生产环境下的推理通常需要多租户支持 ⇒ TPUv4i 的 HBM 容量可以支持多个租户。 \
⑧ DNN 的内存和计算需求每年增长约 1.5 倍 ⇒ 为了支持 DNN 的增长，TPUv4i 配备了 4 个 MXU、快速的片上和片外内存，以及用于连接 4 个相邻 TPU 的 ICI。 \
⑨ DNN 工作负载随 DNN 技术的突破而演进 ⇒ 其可编程性和软件栈有助于紧跟 DNN 的发展步伐。 \
⑩ 推理的服务水平目标 (SLO) 是 P99 延迟，而非批处理大小 (batch size) ⇒ 具备机器学习向后兼容性的训练使 DNN 能够适配 TPUv4i，产生 8~128 的批处理大小，从而在满足 SLO 的同时提高吞吐量。



## Reference
<a id="ref-1"></a>[1] [Google’s Training Chips Revealed:TPUv2 and TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)\
<a id="ref-2"></a>[2] [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)\
<a id="ref-3"></a>[3] [Ten Lessons From Three Generations Shaped Google’s TPUv4i Industrial Product](https://ieeexplore.ieee.org/document/9499913)\
<a id="ref-4"></a>[4] [(EPFL CS723)Ten Lessons From Three Generations Shaped Google’s TPUv4i](https://parsa.epfl.ch/course-info/cs723/lectures/hw_accel.pdf) \
<a id="ref-5"></a>[5] [Using a Total Cost of Ownership (TCO) Model for Your Data Center](https://www.datacenterknowledge.com/business/using-a-total-cost-of-ownership-tco-model-for-your-data-center)\
<a id="ref-6"></a>[6] [A Machine Learning Supercomputer With An Optically Reconfigurable Interconnect and Embeddings Support](https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf)