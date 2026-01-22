# awesome tpu tutorial

![GitHub stars](https://img.shields.io/github/stars/Zhighway777/TPU-Programming-Tutorials?style=social)
![GitHub forks](https://img.shields.io/github/forks/Zhighway777/TPU-Programming-Tutorials?style=social)
![GitHub issues](https://img.shields.io/github/issues/Zhighway777/TPU-Programming-Tutorials)
![GitHub license](https://img.shields.io/github/license/Zhighway777/TPU-Programming-Tutorials)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

[English](#english) | [中文](#chinese)

---

## <a name="english"></a>English

### 📚 About This Repository

> 🎓 **Focus**: Architecture-aware TPU programming  
> This repository emphasizes **how TPU hardware architecture, XLA/HLO compilation,
> and kernel-level design interact**, rather than only API usage.


A comprehensive tutorial repository covering **TPU (Tensor Processing Unit)** programming and architecture. This repository provides educational materials, hands-on tutorials, code examples, and resources for developers, researchers, and students interested in learning TPU programming.

### 🔗 Resources

#### Blogs
- [TPU Deep Dive](https://henryhmko.github.io/posts/tpu/tpu.html)
- [Google TPU Architecture: Complete Guide to 7 Generations](https://www.introl.io/blog/google-tpu-architecture-complete-guide-7-generations)

#### Slides
- [TPU Datacenter Performance Analysis](https://andrewt0301.github.io/hse-acos-course/part1ca/16_TPU/NAETPUslides5Apr17v2.pdf)
- [Ten Lessons: 4 TPU Generations](https://www.cs.ucla.edu/wp-content/uploads/cs/PATTERSON-10-Lessons-4-TPU-gens-CO2e-45-minutes.pdf)

#### Papers
- [Ten Lessons From Three Generations Shaped Google’s TPUv4i](https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf)
- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/pdf/1704.04760)
- [Tensor Processing Units (TPU): A Technical Analysis and Their Impact on Artificial Intelligence](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf)

#### Docs
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [TensorFlow TPU Guide](https://www.tensorflow.org/guide/tpu)
- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [JAX on TPU](https://jax.readthedocs.io/)
- [Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html)
- [How to Scale Your Model | Google DeepMind](https://jax-ml.github.io/scaling-book/)

#### GitHub
- [TPU Starter](https://github.com/ayaka14732/tpu-starter/tree/main)

### 📊 Reference Summary

> References cited in the architecture articles under `docs/architecture/`. English section includes **English-language sources only**.

#### Blogs
- [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
- [Google supercharges machine learning tasks with custom chip](https://cloud.google.com/blog/products/ai-machine-learning/google-supercharges-machine-learning-tasks-with-custom-chip)

#### Slides
- [Google’s Training Chips Revealed: TPUv2 and TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)
- [Ten Lessons: 4 TPU Generations](https://www.cs.ucla.edu/wp-content/uploads/cs/PATTERSON-10-Lessons-4-TPU-gens-CO2e-45-minutes.pdf)
- [TPU Datacenter Performance Analysis](https://andrewt0301.github.io/hse-acos-course/part1ca/16_TPU/NAETPUslides5Apr17v2.pdf)
- [Ten Lessons From Three Generations Shaped Google’s TPUv4i (EPFL CS723)](https://parsa.epfl.ch/course-info/cs723/lectures/hw_accel.pdf)
- [A Machine Learning Supercomputer With An Optically Reconfigurable Interconnect and Embeddings Support](https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf)

#### Papers
- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/pdf/1704.04760)
- [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)
- [The Design Process for Google’s Training Chips: TPUv2 and TPUv3 (IEEE)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9351692)
- [Ten Lessons From Three Generations Shaped Google’s TPUv4i Industrial Product](https://ieeexplore.ieee.org/document/9499913)
- [TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings](https://arxiv.org/pdf/2304.01433)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Tensor Processing Units (TPU): A Technical Analysis and Their Impact on Artificial Intelligence](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf)

#### Docs
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [TensorFlow TPU Guide](https://www.tensorflow.org/guide/tpu)
- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [JAX on TPU](https://jax.readthedocs.io/)
- [Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html)
- [How to Scale Your Model | Google DeepMind](https://jax-ml.github.io/scaling-book/)
- [Total Cost of Ownership - Wikipedia](https://en.wikipedia.org/wiki/Total_cost_of_ownership)

#### GitHub
- [TPU Starter](https://github.com/ayaka14732/tpu-starter/tree/main)

### 🎯 What You'll Learn

- **TPU Architecture**: Understanding TPU hardware design, components, and performance characteristics
- **TPU Programming**: Programming models, APIs, and frameworks for TPU development
- **Optimization Techniques**: Best practices for optimizing TPU performance
- **Practical Examples**: Real-world applications and use cases
- **Cloud TPU & Edge TPU**: Working with Google Cloud TPU and Edge TPU devices

### 📖 Table of Contents

- [Resources](#resources)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tutorials](#tutorials)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Community](#community)

### 🚀 Getting Started

#### Prerequisites

- Basic understanding of machine learning and deep learning concepts
- Familiarity with Python programming
- Knowledge of TensorFlow or PyTorch (recommended)

#### Quick Start

```bash
# Clone the repository
git clone https://github.com/Zhighway777/awesome-tpu-tutorial.git

# Navigate to the repository
cd awesome-tpu-tutorial

# Explore tutorials
cd tutorials/
```

### 📁 Repository Structure

```
awesome-tpu-tutorial/
├── README.md                    # This file
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # License information
├── CODE_OF_CONDUCT.md          # Community guidelines
├── docs/                        # Documentation
│   ├── architecture/            # TPU architecture documentation
│   ├── programming-guides/      # Programming guides
│   └── api-reference/           # API references
├── tutorials/                   # Step-by-step tutorials
│   ├── beginner/               # Beginner-level tutorials
│   ├── intermediate/           # Intermediate-level tutorials
│   └── advanced/               # Advanced tutorials
├── examples/                    # Code examples
│   ├── tensorflow/             # TensorFlow examples
│   ├── pytorch/                # PyTorch examples
│   └── jax/                    # JAX examples
└── resources/                   # Additional resources
    ├── papers/                 # Research papers
    ├── presentations/          # Slides and presentations
    └── references/             # External references
```

### 📝 Tutorials

#### Beginner Level
- Introduction to TPU and its advantages
- Setting up TPU development environment
- Your first TPU program
- Basic tensor operations on TPU

#### Intermediate Level
- TPU memory management
- Data pipeline optimization
- Model parallelism on TPU
- Training neural networks on TPU

#### Advanced Level
- Custom TPU kernels
- Performance profiling and optimization
- Large-scale distributed training
- TPU research and cutting-edge techniques

### 📚 Documentation

- **[TPU Architecture Guide](docs/architecture/)**: Deep dive into TPU hardware design
- **[Programming Guides](docs/programming-guides/)**: Comprehensive programming tutorials
- **[API Reference](docs/api-reference/)**: Detailed API documentation

### 💡 Examples

Browse our collection of practical examples:
- Image classification models
- Natural language processing
- Recommendation systems
- Reinforcement learning
- Custom training loops

### 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- How to submit issues
- How to propose new tutorials
- Code style guidelines
- Pull request process

### 📄 License

This project is licensed under the Apache 2.0 License.- see the [LICENSE](LICENSE) file for details.


### 🌟 Acknowledgments

Special thanks to all contributors and the TPU community for their valuable input and support.

---

## <a name="chinese"></a>中文

### 📚 关于本仓库

> 🎓 **仓库定位**：面向架构与编译器视角的 TPU 编程  
> 本仓库不仅介绍 TPU API 使用，更关注 **TPU 硬件架构、XLA/HLO 编译流程、
> kernel 设计与性能建模之间的关系**。

这是一个TPU（张量处理单元）的编程和架构教程仓库。本仓库为开发者、研究人员和学生提供教学材料、实践教程、代码示例和学习资源。

### 🔗 资源
#### 我的文章
- [google最新的tpu架构信息](docs/architecture/Google's_latest_TPU_tracking.md)

#### 技术博客
> 中文互联网的朋友首先推荐ZOMI老师的AI Infra中关于Google TPU架构的演进系列
- [谷歌 TPU 发展历史以及架构演变 | ZOMI](https://infrasys-ai.github.io/aisystem-docs/02Hardware05Abroad/04TPUIntrol.html)
- [TPU 使用教程](https://shizhediao.github.io/TPU-Tutorial/)
- [TPU深度探索（TPU Deep Dive）](https://henryhmko.github.io/posts/tpu/tpu.html)
- [SemiAnalysis深度解读TPU--谷歌冲击“英伟达帝国”](https://wallstreetcn.com/articles/3760377)
- [TPU 架构：谷歌 7 代处理器完全指南](https://www.introl.io/blog/google-tpu-architecture-complete-guide-7-generations)
#### PPT
- [TPU数据中心的性能分析](https://andrewt0301.github.io/hse-acos-course/part1ca/16_TPU/NAETPUslides5Apr17v2.pdf)
- [TPU演进十年：Google的十大经验教训](https://www.cs.ucla.edu/wp-content/uploads/cs/PATTERSON-10-Lessons-4-TPU-gens-CO2e-45-minutes.pdf)

#### Paper
- [Ten Lessons From Three Generations Shaped Google’s TPUv4i](https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf)
- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/pdf/1704.04760)

#### 技术文档
- [Google Cloud TPU文档](https://cloud.google.com/tpu/docs)
- [TensorFlow TPU指南](https://www.tensorflow.org/guide/tpu)
- [PyTorch XLA文档](https://pytorch.org/xla/)
- [JAX on TPU](https://jax.readthedocs.io/)
- [Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html)
- [ensor Processing Units (TPU): A Technical Analysis and Their Impact on Artificial Intelligence](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf)
- [How to Scale Your Model | Google DeepMind](https://jax-ml.github.io/scaling-book/)

#### GitHub
- [TPU Starter](https://github.com/ayaka14732/tpu-starter/tree/main)

### 📊 资料汇总

> 依据 `docs/architecture/` 中文章的引用资料汇总（中文区不做语言限制）。

#### 技术博客
- [谷歌 TPU 发展历史以及架构演变 | ZOMI](https://infrasys-ai.github.io/aisystem-docs/02Hardware05Abroad/04TPUIntrol.html)
- [TPU 使用教程](https://shizhediao.github.io/TPU-Tutorial/)
- [TPU深度探索（TPU Deep Dive）](https://henryhmko.github.io/posts/tpu/tpu.html)
- [SemiAnalysis深度解读TPU--谷歌冲击“英伟达帝国”](https://wallstreetcn.com/articles/3760377)
- [TPU 架构：谷歌 7 代处理器完全指南](https://www.introl.io/blog/google-tpu-architecture-complete-guide-7-generations)
- [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
- [Google supercharges machine learning tasks with custom chip](https://cloud.google.com/blog/products/ai-machine-learning/google-supercharges-machine-learning-tasks-with-custom-chip)

#### PPT
- [TPU数据中心的性能分析](https://andrewt0301.github.io/hse-acos-course/part1ca/16_TPU/NAETPUslides5Apr17v2.pdf)
- [TPU演进十年：Google的十大经验教训](https://www.cs.ucla.edu/wp-content/uploads/cs/PATTERSON-10-Lessons-4-TPU-gens-CO2e-45-minutes.pdf)
- [Google’s Training Chips Revealed: TPUv2 and TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)
- [Ten Lessons From Three Generations Shaped Google’s TPUv4i (EPFL CS723)](https://parsa.epfl.ch/course-info/cs723/lectures/hw_accel.pdf)
- [A Machine Learning Supercomputer With An Optically Reconfigurable Interconnect and Embeddings Support](https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf)

#### Paper
- [Ten Lessons From Three Generations Shaped Google’s TPUv4i](https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf)
- [Ten Lessons From Three Generations Shaped Google’s TPUv4i Industrial Product](https://ieeexplore.ieee.org/document/9499913)
- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/pdf/1704.04760)
- [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)
- [The Design Process for Google’s Training Chips: TPUv2 and TPUv3 (IEEE)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9351692)
- [TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings](https://arxiv.org/pdf/2304.01433)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Tensor Processing Units (TPU): A Technical Analysis and Their Impact on Artificial Intelligence](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf)

#### 技术文档
- [Google Cloud TPU文档](https://cloud.google.com/tpu/docs)
- [TensorFlow TPU指南](https://www.tensorflow.org/guide/tpu)
- [PyTorch XLA文档](https://pytorch.org/xla/)
- [JAX on TPU](https://jax.readthedocs.io/)
- [Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html)
- [How to Scale Your Model | Google DeepMind](https://jax-ml.github.io/scaling-book/)
- [Total Cost of Ownership - Wikipedia](https://en.wikipedia.org/wiki/Total_cost_of_ownership)

#### GitHub
- [TPU Starter](https://github.com/ayaka14732/tpu-starter/tree/main)

### 🎯 学习内容

- **TPU架构**：理解TPU硬件设计、组件和性能特征
- **TPU编程**：TPU开发的编程模型、API和框架
- **优化技术**：TPU性能优化的最佳实践
- **实践示例**：真实世界的应用和用例
- **Cloud TPU与Edge TPU**：使用Google Cloud TPU和Edge TPU设备

### 📖 目录

- [资源](#资源)
- [快速开始](#快速开始)
- [仓库结构](#仓库结构)
- [教程](#教程)
- [文档](#文档)
- [示例](#示例)
- [贡献](#贡献)
- [许可证](#许可证)
- [社区](#社区)

### 🚀 快速开始

#### 前置要求

- 机器学习和深度学习基础知识
- 熟悉Python编程
- 了解TensorFlow或PyTorch（推荐）

#### 快速开始

```bash
# 克隆仓库
git clone https://github.com/Zhighway777/TPU-Programming-Tutorials.git

# 进入仓库目录
cd TPU-Programming-Tutorials

# 浏览教程
cd tutorials/
```

### 📁 仓库结构

```
TPU-Programming-Tutorials/
├── README.md                    # 本文件
├── CONTRIBUTING.md              # 贡献指南
├── LICENSE                      # 许可证信息
├── CODE_OF_CONDUCT.md          # 社区准则
├── docs/                        # 文档
│   ├── architecture/            # TPU架构文档
│   ├── programming-guides/      # 编程指南
│   └── api-reference/           # API参考
├── tutorials/                   # 分步教程
│   ├── beginner/               # 初级教程
│   ├── intermediate/           # 中级教程
│   └── advanced/               # 高级教程
├── examples/                    # 代码示例
│   ├── tensorflow/             # TensorFlow示例
│   ├── pytorch/                # PyTorch示例
│   └── jax/                    # JAX示例
└── resources/                   # 附加资源
    ├── papers/                 # 研究论文
    ├── presentations/          # 幻灯片和演示文稿
    └── references/             # 外部参考资料
```

### 📝 教程

#### 初级
- TPU介绍及其优势
- 搭建TPU开发环境
- 第一个TPU程序
- TPU上的基本张量操作

#### 中级
- TPU内存管理
- 数据流水线优化
- TPU上的模型并行
- 在TPU上训练神经网络

#### 高级
- 自定义TPU内核
- 性能分析和优化
- 大规模分布式训练
- TPU研究与前沿技术

### 📚 文档

- **[TPU架构指南](docs/architecture/)**：深入了解TPU硬件设计
- **[编程指南](docs/programming-guides/)**：全面的编程教程
- **[API参考](docs/api-reference/)**：详细的API文档

### 💡 示例

浏览我们的实践示例集合：
- 图像分类模型
- 自然语言处理
- 推荐系统
- 强化学习
- 自定义训练循环

### 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情：
- 如何提交问题
- 如何提议新教程
- 代码风格指南
- Pull Request流程

### 📄 许可证

本项目采用 **Apache License 2.0** 开源许可证。 - 详见[LICENSE](LICENSE)文件。


---

## 📊 Repository Statistics

![GitHub contributors](https://img.shields.io/github/contributors/Zhighway777/TPU-Programming-Tutorials)
![GitHub last commit](https://img.shields.io/github/last-commit/Zhighway777/TPU-Programming-Tutorials)
![GitHub repo size](https://img.shields.io/github/repo-size/Zhighway777/TPU-Programming-Tutorials)

---

**Note**: This repository is continuously updated with new tutorials and resources. Star ⭐ this repository to stay updated!
