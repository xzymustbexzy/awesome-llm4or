# awesome-llm4or

## 论文整理

### 模型结构优化
|  论文题目  |       发表时间       |    内容简介    |
| :-----------: | :--------: | :------------------: |
| Generating Long Sequences with Sparse Transformers |     2019.04     |      一种更加高效的Transformer结构设计，适用于长序列计算，有点类似于informer |
| Efficient Diffusion Training via Min-SNR Weighting Strategy | 2023.03 | 这篇论文主要研究Diffusion模型的训练过程收敛比较慢的问题作者发现收敛慢的一个原因是不同的timestep，模型loss更新方向是冲突的 |
| Resurrecting Recurrent Neural Networks for Long Sequences | 2023.03 | 一种RNN的改进版，可以做并行计算，适应长序列任务 |

### 多语言大模型
|  论文题目  |       发表时间       |    内容简介    |
| :-----------: | :--------: | :------------------: |
| Polyglot: Large Language Models of Well-balanced Competence in Multi-languages |     2022.09     |      一个语种更加平衡的多语言模型 |
| Evolutionary-scale prediction of atomic-level protein structure with a language model | 2023.03 | 一个直接预测蛋白质原子级别结构的大模型，15B参数 |
| BigTrans: Augmenting Large Language Models with Multilingual Translation Capability over 100 Languages | 2023.05 | 多语言模型，用超过100种语言的多语言翻译能力增强大型语言模型的功能 |

### 计算层优化
|  论文题目  |       发表时间       |    内容简介    |
| :-----------: | :--------: | :------------------: |
| FlexGen: High-throughput Generative Inference of Large Language Models with a Single GPU |     2023.03     |      FlexGen能够在显存容量有限的GPU中运行大语言模型。具体的方法是：在offloading的基础上做了一些tensor压缩，以及对其存储、读取的调度 |
| Fine-tuning Language Model with Just Forward Passes | 2023.05 | 用多次前向推导代替反向梯度计算，节省大模型训练时的内存开销使用MeZO作为优化器 |
| BigTrans: Augmenting Large Language Models with Multilingual Translation Capability over 100 Languages | 2023.05 | 多语言模型，用超过100种语言的多语言翻译能力增强大型语言模型的功能 |

### 大模型原理研究
|  论文题目  |       发表时间       |    内容简介    |
| :-----------: | :--------: | :------------------: |
| Model Dementia: Generated Data Makes Models Forget |     2023.05     |    论文讨论了一个叫“模型痴呆”的现象。就是用模型自己生成的内容去对他做训练时，会让模型能力存在缺陷 |
| Scan and Snap: Understanding Training Dynamics and Token Composition in 1-layer Transformer | 2023.05 | 研究单层Transformer中的动态训练和token组成 |
| Backpack Language Models | 2023.05 | 提出了一种新的网络架构，叫背包网络模型 |
| Lexinvariant Language Models | 2023.05 | 研究语言模型是否可以不使用固定的词嵌入，而仅依靠上下文中词汇的重叠和重复来掌握词义。 |

### LLM复杂任务
|  论文题目  |       发表时间       |    内容简介    |
| :-----------: | :--------: | :------------------: |
| SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks |     2023.05     |    类似于ReAct的一篇论文，讨论LLM如何做复杂的任务规划 |
| Ghost in the Minecraft: Generally Capable Agents for Open-World Enviroments via Large Language Models with Text-based Knowledge and Memory | 2023.05 | 用LLM玩《我的世界》。研究通过大模型的文本知识和记忆能力，为开放世界提供一般能力的Agent |
| Training Socially Aligned Language Models in Simulated Human Society | 2023.05 | 提出了一种训练范式，能从模拟的社交互动中学习，保证模型行为对齐人类价值观 |
| Deliberate Problem Solving with Large Language Models | 2023.05 | 不同于CoT、self-consistency、Progressive Hint，提出了一种新的基于LLM的思考范式 |

### GPT生成检测
|  论文题目  |       发表时间       |    内容简介    |
| :-----------: | :--------: | :------------------: |
| DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text |     2023.05     |    N-gram分析：一种无需检测的GPT文本检测算法 |

### LLM能力测评
|  论文题目  |       发表时间       |    内容简介    |
| :-----------: | :--------: | :------------------: |
| Do GPTs Produce Less Literal Translations? |     2023.05     |    发现GPT在翻译时字面意思较少，翻译效果较好 |

### 多模态大模型
|  论文题目  |       发表时间       |    内容简介    |
| :-----------: | :--------: | :------------------: |
| Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning | 2023.02 | 输入视频，为其添加讲解字幕 |
| Edit-A-Video: Single Video Editing with Object-Aware Consistency | 2023.03 | 一个用文本修改视频的模型输入是一个(text, video)的pair，输出一个根据text修改过的video大致原理是，给video加上噪声，然后再用text去指导diffusion过程 |
| PandaGPT:One Model To Instruction-Follow Them All |     2023.05     |    多模态大模型，接受视频、音频、深度、热、IMU |