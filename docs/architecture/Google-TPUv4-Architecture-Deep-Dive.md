# From TPUv3 to TPUv4i

## Background
在开始介绍TPUv4i之前，我们先做一个warm up：\

在上一节[TPUv1~v3 revealed](TPUv1~v3_revealed.md)中我们介绍了TPUv1到TPUv3的架构演变过程。TPUv1到TPUv3的设计主要是为了应对从推理为主到训练和推理并重的需求转变。TPUv1主要解决了推理任务(当年推理任务还被称为serving任务)TPUv1使用Activation Pipeline的固定单元和64K个8bit的MAC组成的MXU作为计算模块，使用DDR3作为存储模块，Activation Storage和Accumulators作为中间数据存储模块。CPU通过PCIe总线和TPU交换数据名且发送指令。TPUv1的设计目标是高效能的推理计算，同时考虑到大规模部署的成本和功耗问题。\

TPUv2的设计目标转向为更难的训练任务，这在并行化训练(训练需要并行协调资源，而推理不需要)，计算能力(涉及到反向传播的求导计算)和更高的内存需求（权重更新需要访问前向和反向传播的中间值），以及更高的可编程性（训练算法和模型在不断变化）和最后需要同时支持INT型和浮点型计算。因此将Activation Storage和Accumulators合并为Vector Memory，使用更通用的Vector Unit替代Activation Pipeline，增强了可编程性。需要补充的是Bf16更适合DNN，MXU的bf16也是从TPUv2开始支持的。由于训练的目的就是设定权重值，而且需要大量的缓冲空间来存储中间变量。使用HBM DRAM来作为Vector Memory的后备，二者也构成了编译器可控的存储层次结构。在指令获取方面TPUv2从本地的存储器获取自己的322bit的VLIW指令，摆脱了对CPU的依赖。由于训练需要Scale Out，TPUv2引入了定制的片间互连(InterChip Interconnect, ICI)模块，可以构建256个芯片组成的pod。\

TPUv3是TPUv2的“年上版本”，通过提升特定的参数来吉大提升性能，这里不再赘述，感兴趣的读者可以参考上一节内容。

Google 从TPUv3到TPUv4i的设计范式有了巨大的转变，
## Reference
<a id="ref-1"></a>[1] [Google’s Training Chips Revealed:TPUv2 and TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)\
<a id="ref-2"></a>[2] [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)