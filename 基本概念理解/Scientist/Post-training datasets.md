# Post-training datasets

Post-training datasets have a precise structure with instructions and answers (supervised fine-tuning) or instructions and chosen/rejected answers (preference alignment). Conversational structures are a lot rarer than the raw text used for pre-training, which is why we often need to process seed data and refine it to improve the accuracy, diversity, and complexity of the samples. More information and examples are available in my repo 💾 LLM Datasets.

* Storage & chat templates: Because of the conversational structure, post-training datasets are stored in a specific format like ShareGPT or OpenAI/HF. Then, these formats are mapped to a chat template like ChatML or Alpaca to produce the final samples the model is trained on.
* Synthetic data generation: Create instruction-response pairs based on seed data using frontier models like GPT-4o. This approach allows for flexible and scalable dataset creation with high-quality answers. Key considerations include designing diverse seed tasks and effective system prompts.
* Data enhancement: Enhance existing samples using techniques like verified outputs (using unit tests or solvers), multiple answers with rejection sampling, Auto-Evol, Chain-of-Thought, Branch-Solve-Merge, personas, etc.
* Quality filtering: Traditional techniques involve rule-based filtering, removing duplicates or near-duplicates (with MinHash or embeddings), and n-gram decontamination. Reward models and judge LLMs complement this step with fine-grained and customizable quality control.



**简单解释：**

后训练（post-training）的数据集有两种常见格式：

* **SFT（Supervised Fine-Tuning）数据集：** 一个“指令-答案”对。比如，“请写一首诗”，模型给出答案。
* **偏好数据集（Preference Alignment，如RLHF）：** 一个指令+多个答案，标出哪个答案是首选（chosen）哪个被拒绝（rejected）。

**常见面试题：**

* SFT和偏好数据集有什么区别？
* 为什么会话类数据比原始文本稀缺？
* 为什么需要对种子数据进行处理？

---

## 二、存储与Chat模板（Storage & Chat Templates）

**简单解释：**

* 数据集通常用像ShareGPT、OpenAI/HF这样的格式保存对话内容。
* 这些原始格式再映射到“chat模板”（比如ChatML、Alpaca），用于模型实际训练。Chat模板就是规范每条对话的输入输出格式，让模型能“看懂”数据。

**常见面试题：**

* 为什么要用chat模板？
* ChatML和Alpaca的格式有什么区别？
* chat模板如何影响模型训练？

---

## 三、合成数据生成（Synthetic Data Generation）

**简单解释：**

* 用像GPT-4o这样的强大模型，根据设计好的种子任务（指令）批量生成优质的指令-答案对。
* 这样可以快速扩充数据集，尤其是一些罕见领域或复杂任务。

**常见面试题：**

* 合成数据相较于人工标注数据的优缺点是什么？
* 如何确保合成数据的多样性和质量？
* 设计好的system prompt对数据生成有何影响？

---

## 四、数据增强（Data Enhancement）

**简单解释：**

优化和扩展已有数据，常见方法有：

* **Verified outputs** ：针对代码类问题，用单元测试/验证器自动检查答案是否正确。
* **多答案+拒绝采样（rejection sampling）** ：让模型生成多个答案，只保留最优答案。
* **Auto-Evol** ：自动让指令变得更复杂（进化）。
* **Chain-of-Thought** ：让模型输出推理步骤，而不仅是最终答案。
* **Branch-Solve-Merge** ：拆解复杂问题，分别解决后合并。
* **Personas** ：让模型模拟不同身份（如老师、医生）来作答。

**常见面试题：**

* Chain-of-Thought（CoT）是什么？对模型训练有什么帮助？
* 什么是rejection sampling？如何在数据增强中使用？
* Personas能带来哪些好处？

---

## 五、质量过滤（Quality Filtering）

**简单解释：**

确保数据干净和高质量，常用手段有：

* **规则过滤** ：去除脏话、无意义内容、过短或过长的样本等。
* **去重** ：用MinHash或向量相似度去除重复/近重复样本，防模型记忆。
* **n-gram去污染（decontamination）** ：避免训练集和评测集有重叠，保证评测结果真实。
* **奖励模型（Reward Models）与Judge LLMs** ：用模型自动判分、打分过滤低质量样本，实现更细致的质量控制。

**常见面试题：**

* 为什么要去重？有哪些去重方法？
* 什么是n-gram去污染？为什么重要？
* 奖励模型和Judge LLM起到什么作用？

---

**总结一句：**

Post-training的数据集结构和处理关注“如何让LLM学到更像人的、对话式、丰富且高质量的知识”，涉及格式规范、智能数据生成、高级数据优化和多层次质量控制，是LLM训练环节中至关重要的一环。

如果你准备相关面试，可以根据每个要点多查些实际案例或深挖典型流程，会更有帮助！
