# 1. The LLM architecture

An in-depth knowledge of the Transformer architecture is not required, but it's important to understand the **main steps** of modern LLMs:

1. converting text into numbers through **tokenization**
2. processing these tokens through layers including **attention mechanisms**
3. finally generating new text through various **sampling strategies**.

* [ ] **Architectural Overview** : Understand the evolution from encoder-decoder Transformers to decoder-only architectures like GPT, which form the basis of modern LLMs. Focus on how these models process and generate text at a high level.
* [ ] **Tokenization** : Learn the principles of tokenization - how text is converted into numerical representations that LLMs can process. Explore different tokenization strategies and their impact on model performance and output quality.
* [ ] **Attention mechanisms** : Master the core concepts of attention mechanisms, particularly self-attention and its variants. Understand how these mechanisms enable LLMs to process long-range dependencies and maintain context throughout sequences.
* [ ] **Sampling techniques** : Explore various text generation approaches and their tradeoffs. Compare deterministic methods like greedy search and beam search with probabilistic approaches like temperature sampling and nucleus sampling.

1.1 Architectural Overview (Transformer)

**概念解释**
指的是大模型（LLM, Large Language Model）整体的结构设计及其各个模块如何协同工作。主流LLM架构（如Transformer）包括编码器、解码器、嵌入层、多头注意力、前馈神经网络、归一化层等部分。

**核心要点：**

- Transformer 架构为何成功（并行、长距离依赖）
- 编码器和解码器有何不同
- 层（layer）的堆叠方式
- 参数量级对性能的影响

**常见面试题：**

- 简述 Transformer 的总体架构及其优势。
- 说明编码器和解码器在NLP任务中的不同应用场景。
- 为什么现代LLM都选择Transformer框架？

---

### 1.2 Tokenization

**概念解释**
文本分词，是指把原始文本转化为模型可处理的“token”单元。常用方法有字符级、词级、Subword（如BPE、WordPiece、Unigram）等。

**核心要点：**

- 什么是token、subword、BPE（Byte Pair Encoding）？
- 分词方法对长文本和多语言处理的影响
- OOV（out-of-vocabulary）问题的缓解方法

**常见面试题：**

- LLM中常用的tokenization算法有哪些？有何优缺点？
- 如何平衡tokenizer的词表大小和性能？
- 给定一句话，分别用BPE和WordPiece分词会发生什么不同？

---

### 1.3 Attention Mechanisms

**概念解释**
注意力机制（attention）允许模型聚焦于输入序列中最相关的部分。Transformer中的“自注意力”可为每个token分配语境相关的权重。

**核心要点：**

- Scaled dot-product attention的计算公式
- Multi-head Attention的好处
- 自注意力（self-attention）与交叉注意力（cross-attention）

**常见面试题：**

- 详细描述 self-attention 的计算过程。
- 为什么需要 Multi-head Attention？
- Attention Score 的归一化通常采用什么方法？为什么？

---

### 1.4 Sampling Techniques

**概念解释**
用于模型生成文本时，从输出概率分布中选取token的策略。常见有Greedy、Beam Search、Top-k、Top-p（nucleus）、Temperature等方法。

**核心要点：**

- 不同采样方法（greedy, beam, top-k, top-p）的原理及优缺点
- 采样策略对生成质量、多样性和连贯性的影响
- 实际应用时采样超参数的调节

**常见面试题：**

- Top-k与Top-p采样有何区别？各自适用什么场景？
- 你如何调节temperature来改变生成文本的特性？
- 为什么beam search对语言模型长文本生成有时会退化为重复输出？
