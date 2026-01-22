# Latest Google TPU architecture tracking
> Supplementing information on Google TPU models and characteristics from 2021 to present.

## Historical architecture
Google TPU started with TPU-v1 in 2016. The architectural evolution to the widely known TPU-v4 series in 2021 can be found in ZOMI’s article [1](#ref-1), and Professor Patterson also wrote a dedicated overview of TPU evolution from 2016 to 2021 [2](#ref-2). If you are interested in the proposal and subsequent development of TPU architecture, you can refer to my other article [From CPU, GPU to TPU](From-CPU-GPU-to-TPU.md).

### TPUv1
The TPUv1 paper mentions “significantly improving TCO [4](#ref-4) to reduce the cost of neural network inference in the data center.” Therefore, TPU was designed from the start for large-scale deployment and efficient compute. To simplify hardware design and rapid deployment, TPU chose a PCIe/IO bus architecture, plugged into servers like a GPU accelerator card [3](#ref-3). Simply put, it is like a “plug-in single-chip controller.”

Below is the PCB design and a photo of actual deployment in server racks [5](#ref-5):
<div style="text-align: center;">
<img src="../pics/arch_tracking/TPUv1-PCB.png" alt="TPUv1-PCB" width="200">
<img src="../pics/arch_tracking/TPUv1-rack.png" alt="Server racks with TPUs" width="200">
</div>  

### TPUv2

TPUv1 was designed for NN inference, and soon exposed bottlenecks: it could not support ML training tasks such as AlexNet, CNN0, ImageNet, and NLP tasks like BERT.\
This required more compute capacity, memory capacity, and bandwidth. Also, because training algorithms and models evolve, greater programmability is needed. Thus in TPUv2 [6](#ref-6), Google adopted a dual-core configuration and used HBM instead of DDR3 as memory. For programmability, TPUv2 replaced TPUv1’s hardcoded functional units with a more general programmable vector unit, and adopted a VLIW architecture closely integrated with the XLA compiler (XLA also grew gradually from TPUv2). TPUv2 also integrated a custom InterChip Interconnect (ICI/片间互联) [6](#ref-6) to support large-scale distributed training and proposed the pod (supercomputer) concept that continues to TPUv7.\
Overall, TPUv1 focused on efficient inference, while TPUv2 successfully transformed into a powerful ML training chip through specialized compute units, dramatically higher memory bandwidth, a rebuilt memory hierarchy, more flexible architecture, and interconnect designed for large-scale parallelism. These changes completed in 2017, the same year Google proposed the landmark paper “Attention is All You Need [8]”. Once again, it highlights Google’s engineering capability and forward-looking design.


The figure below [7](#ref-7) shows TPUv2 PCB and pod architecture:

TPUv2 boards = 4 chips  

![TPUv2-PCB](../pics/arch_tracking/TPUv2-PCB.png)

TPUv2 pod = 256 chips  

![TPUv2-pod](../pics/arch_tracking/TPUv2-pod.png)
### TPUv3
TPUv3 is an incremental improvement based on TPUv2, using “marginal effects” to significantly boost performance. Although there were no major architectural changes, these incremental improvements still required careful analysis and tuning.
Key changes include:
1) Doubled Matrix Multiply Unit (MXU - Matrix Multiply Unit) count, directly doubling max FLOPS
2) Clock frequency increased from 700 MHz (TPUv2) to 940 MHz (TPUv3), improving performance
3) Increased HBM bandwidth and doubled capacity, enabling larger models and batch sizes
4) Enhanced interconnect link bandwidth
5) Maximum system scale expanded from TPUv2’s 256 chips to TPUv3’s 1024 chips
6) Liquid cooling adopted starting from TPUv3

By stacking these optimizations on a mature design, TPUv3 achieved a large performance leap in a relatively short time, supporting larger ML workloads.

The figure below [7](#ref-7) shows TPUv3 PCB and pod architecture:\
TPUv3 boards = 4 chips  

![TPUv3-PCB](../pics/arch_tracking/TPUv3-PCB.png)

TPUv3 pod = 1024 chips   

![TPUv3-pod](../pics/arch_tracking/TPUv3-pod.png)

### TPUv4i

### TPUv4
TPUv4 was finally released in 2022, four years after TPUv3. Despite many important AI achievements in this period, Google seemed to have delayed TPU updates, which led to weaker chip performance compared to competitors like NVIDIA and AMD in 2023 and 2024. This further widened the gap between Google’s AI models and top AI companies like OpenAI and Meta. Nevertheless, Google’s Gemini models caught up within the following two years (2024-2025), which is inseparable from its full-stack self-reliant capability and formidable engineering ability.

Between 2018 and 2022, AI changed significantly. The rise of transformer architectures and the popularity of LLMs and diffusion models, and PyTorch replacing TensorFlow as the mainstream framework (Google also shifted internally from TensorFlow to JAX). I will later write a dedicated article about framework evolution [Training Framework Rise and Fall](../framework/Training-Framework-History.md). Besides model architectures and frameworks, model parameter counts grew by three to four orders of magnitude. ![Trend of FLOPs required for model training](../pics/arch_tracking/模型训练所需的FLOPs增长趋势.png) Therefore, TPUv4 was designed to address these changes.\
From TPUv4 onward, I will analyze its chip architecture in detail and compare it with TPUv3 ([In-depth look at TPUv1~v3 architectural changes](TPUv1~v3_revealed.md)), because v4 plays a pivotal role in the overall lineage. Therefore, a detailed analysis is necessary. Here I only cover the overall architecture to keep the article coherent. If you want deeper details of TPUv4 hardware architecture and programming model, please refer to the dedicated article [Google TPUv4 Architecture Deep Dive](Google-TPUv4-Architecture-Deep-Dive.md).

The figure below [9](#ref-9) shows TPUv4 PCB and pod architecture:\
TPUv4 boards = 4 chips \
![TPUv4-PCB](../pics/arch_tracking/TPUv4-PCB.png)

TPUv4 pod = 4096 chips \
![TPUv4-pod](../pics/arch_tracking/TPUv4-pod.png)

TPUv4


### TPUv5
> **Note:** Most online TPU architecture summaries stop at TPUv4. There is less information about TPUv5 and later versions. The following content is compiled from limited information; corrections are welcome.


**TPUv5e**


**TPUv5p**


### TPUv6



### TPUv7

> **Concepts Explained:**
> - TCO (Total Cost of Ownership)\
Means the total of all direct and indirect costs related to an asset or device over its lifecycle. In data centers, power consumption is highly correlated with TCO, so TPU’s main design metric is performance per watt.

## Reference
<a id="ref-1"></a>[1] [谷歌TPU历史发展|ZOMI](https://infrasys-ai.github.io/aisystem-docs/02Hardware05Abroad/04TPUIntrol.html)\
<a id="ref-2"></a>[2] [TPU演进十年：Google的十大经验教训](https://www.cs.ucla.edu/wp-content/uploads/cs/PATTERSON-10-Lessons-4-TPU-gens-CO2e-45-minutes.pdf)\
<a id="ref-3"></a>[3] [An in-depth look at Google's first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)\
<a id="ref-4"></a>[4] [Total Cost of Ownership - Wikipedia](https://en.wikipedia.org/wiki/Total_cost_of_ownership)\
<a id="ref-5"></a>[5] [Google supercharges machine learning tasks with custom chip](https://cloud.google.com/blog/products/ai-machine-learning/google-supercharges-machine-learning-tasks-with-custom-chip)\
<a id="ref-6"></a>[6] [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9351692)\
<a id="ref-7"></a>[7] [Google’s Training Chips Revealed: TPUv2 and TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)\
<a id="ref-8"></a>[8] [Attention is All You Need](https://arxiv.org/abs/1706.03762) \
<a id="ref-8"></a>[9] [TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings](https://arxiv.org/pdf/2304.01433)
