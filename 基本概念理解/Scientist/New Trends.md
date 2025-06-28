# New Trends

Here are notable topics that didn't fit into other categories. Some are established (model merging, multimodal) techniques, but others are more experimental (interpretability, test-time compute scaling) and the focus of numerous research papers.

* Model merging: Merging trained models has become a popular way of creating performant models without any fine-tuning. The popular mergekit library implements the most popular merging methods, like SLERP, DARE, and TIES.
* Multimodal models: These models (like CLIP, Stable Diffusion, or LLaVA) process multiple types of inputs (text, images, audio, etc.) with a unified embedding space, which unlocks powerful applications like text-to-image.
* Interpretability: Mechanistic interpretability techniques like Sparse Autoencoders (SAEs) have made remarkable progress to provide insights about the inner workings of LLMs. This has also been applied with techniques such as abliteration, which allow you to modify the behavior of models without training.
* Test-time compute: Reasoning models trained with RL techniques can be further improved by scaling the compute budget during test time. It can involve multiple calls, MCTS, or specialized models like a Process Reward Model (PRM). Iterative steps with precise scoring significantly improve performance for complex reasoning tasks.



## 1. Model Merging（模型融合）

**关键解释：**

- 模型融合指将多个预训练模型的权重进行合成，直接创造新模型，而无需重新微调。常见融合方式包括：
  - **SLERP**（球面线性插值）：权重在高维空间里按一指定比例平滑混合，合成效果通常好于简单平均。
  - **DARE、TIES**：进阶融合算法，侧重混合模型个性化特征和稳健性。
- 融合可以整合不同模型的能力或偏好，快速创造“特定用途”模型。
- **mergekit**是实现这些融合方法的主流工具库。

**常见面试题：**

- Q：相比微调，模型融合在什么情况下更优？
  A：当有多个性能互补的模型时，融合可在无需昂贵再次训练的情况下快速整合它们的长处，适合产能有限但资源多样环境。
- Q：SLERP融合技术的基本原理是什么？
  A：在权重空间内做球面插值，避免线性平均造成的分布漂移，融合后模型表现更加稳定。

---

## 2. Multimodal Models（多模态模型）

**关键解释：**

- 多模态模型能同时处理如文本、图像、音频等多种输入，常通过**统一的embedding空间**实现不同模态的互通。
- 典型例子有**CLIP（图文对齐）、Stable Diffusion（文本生成图像）、LLaVA（视觉语言）**等。
- 这为文本生成图片、跨模态检索等创新应用铺平了道路。

**常见面试题：**

- Q：CLIP如何实现图像和文本的关联？
  A：通过分别编码图文数据，再学得两者embedding对齐，最终能直接比较文本和图片间的相似性。
- Q：多模态模型能实现哪些此前单模态无法胜任的任务？
  A：如文本到图片生成（AIGC）、图像描述生成、跨媒体搜索与理解等。

---

## 3. Interpretability（可解释性）

**关键解释：**

- 机制可解释性（mechanistic interpretability）重在剖析模型内部结构“哪些神经元/token/层控制哪些能力”。
- **Sparse Autoencoders (SAE)**：通过建立稀疏正则化的自编码器，把模型复杂行为映射成少量、人可读的feature组合，便于追踪和干预。
- **Abliteration**等方法可定向修改模型内部表征，以无需再训练的方式影响模型输出，实现特征擦除或个性化调整。

**常见面试题：**

- Q：Sparse Autoencoders 在模型可解释性中的作用？
  A：将复杂特征空间压缩到人易理解的稀疏表征，帮助研究者定位特定功能的“神经元”。
- Q：可解释性研究能为模型安全/操控提供哪些帮助？
  A：理解模型内部机制有助于发现潜在风险点，也便于修正或定制特定行为。

---

## 4. Test-time Compute Scaling（推理时算力提升）

**关键解释：**

- 指在推理阶段“临时”提升算力/推理机会，从而弥补单步推理的能力短板。
- 常用方法：
  - **多轮生成/vote**：多次采样、集成输出，选最佳答案。
  - **MCTS（蒙特卡洛树搜索）**：如AlphaGo那样，多步“搜一部分未来”，提升复杂任务表现。
  - **PRM（Process Reward Model）**：专设奖励模型负责推理中的流程打分，逐步引导模型逼近高质量方案。
- 这些技术对于复杂推理、多步规划、对抗任务大大提升最终表现，但带来推理延迟和资源消耗。

**常见面试题：**

- Q：推理时算力扩展常用于哪些场合？
  A：如复杂数学题推理，规划多步步骤的大型工程任务，或对输出精度/安全性要求极高的应用场景。
- Q：MCTS如何用于大模型推理改进？
  A：构建解答树，通过多次采样、评估不同解法路径，优选推理过程中的最优分支。

---

## 总结

- 这些方向是LLM应用和研究的“新兴突破点”，也常见于面试和开放性问题考查。
- 建议面试时针对“应用落地”和“工程可行性”做举例说明，如：“如何用mergekit交叉开发新模型”，“如何用CLIP快速筛选图文关系”，“如何通过ablation调控输出的安全性”，“多轮解题和树搜索如何改进推理类任务”等，帮助展现工程理解和创新视野。
