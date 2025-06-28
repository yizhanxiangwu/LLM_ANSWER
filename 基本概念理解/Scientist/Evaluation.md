# Evaluation

Reliably evaluating LLMs is a complex but essential task guiding data generation and training. It provides invaluable feedback about areas of improvement, which can be leveraged to modify the data mixture, quality, and training parameters. However, it's always good to remember Goodhart's law: "When a measure becomes a target, it ceases to be a good measure."

* Automated benchmarks: Evaluate models on specific tasks using curated datasets and metrics, like MMLU. It works well for concrete tasks but struggles with abstract and creative capabilities. It is also prone to data contamination.
* Human evaluation: It involves humans prompting models and grading responses. Methods range from vibe checks to systematic annotations with specific guidelines and large-scale community voting (arena). It is more suited for subjective tasks and less reliable for factual accuracy.
* Model-based evaluation: Use judge and reward models to evaluate model outputs. It highly correlates with human preferences but suffers from bias toward their own outputs and inconsistent scoring.
* Feedback signal: Analyze error patterns to identify specific weaknesses, such as limitations in following complex instructions, lack of specific knowledge, or susceptibility to adversarial prompts. This can be improved with better data generation and training parameters.

---

# LLM Evaluation（大模型评估）

## 1. Automated Benchmarks（自动化评测）

**关键解释：**

* 使用高度整理的数据集和明确的指标（如MMLU、GSM8K、HellaSwag等）自动测试模型在特定任务（问答、推理、数学等）上的表现。
* 优势：速度快、标准化、可横向对比。
* 局限：对抽象、创造性任务无能为力，并且存在“数据污染”（即训练集和benchmark重叠，评测不准）风险。

**面试常见题：**

* Q：MMLU属于哪类评测？它的主要目的是什么？

  A：MMLU属于自动化基准评测，用于横向衡量模型的多领域推理和知识覆盖能力。
* Q：自动化评测为什么容易发生数据污染？结果会带来什么风险？

  A：开源数据集容易被训练数据“泄露”或部分重合，导致评测分数虚高，难以真实反映模型泛化能力。

---

## 2. Human Evaluation（人工评测）

**关键解释：**

* 人工直接与模型互动，评判回答的自然度、创造性、准确性等。方法有主观感受打分（vibe check）、明确指标打分、社区大规模投票（如arena）。
* 适合主观性/开放性强的任务，能捕捉自动评测难以量化的潜在特征。
* 局限：主观性强、耗时昂贵，难以做到跨评测场景的一致性；尤其在事实核查方面可靠性差。

**面试常见题：**

* Q：人工评测在什么情况下优于自动评测？

  A：当涉及创新性、情感色彩等主观任务时，人工评测更能捕获实际影响和用户体验。
* Q：如何减少人工评测的主观偏差？

  A：制定详尽、统一的评分标准，采用多评审者投票/加权，或结合评分流程的聚合。

---

## 3. Model-based Evaluation（模型打分/判官）

**关键解释：**

* 利用Judge LLM、Reward Model等专用打分模型对LLM输出“自动评卷”，由于自身就是大型模型，对人类偏好打分高度相关。
* 优势：效率高，可扩展；对大规模候选答案进行细粒度排序。
* 局限：可能出现自我偏向（favor自身输出），以及打分一致性不如人工评测高。

**面试常见题：**

* Q：用Judge LLM评测有哪些陷阱？

  A：容易对自家模型产生偏向、未必和人类偏好完全同步，且对模型攻击/幻觉常不敏感。
* Q：为什么自动Judge的分数和人工分数仍需混合参考？

  A：自动Judge虽然高效但有一致性和偏见风险，人工评测仍然是唯一能充分捕捉主观/情感等高阶能力的方式。

---

## 4. Feedback Signal（反馈信号/问题分析）

**关键解释：**

* 通过分析测试和生产中的模型错误分布、失败类型发现短板。比如：说明性差、细节遗漏、复杂指令理解弱、易被对抗提示干扰等。
* 这些反馈直接指导数据迭代和训练参数优化——提升针对性和整体质量。

**面试常见题：**

* Q：如何用错误分析来提升模型最终性能？

  A：通过聚合和归类失败样本，发现问题类型，有针对性地补充数据、调整标签标准或优化训练参数，实现定向提升。
* Q：什么是Goodhart定律？在模型评测中有何体现？

  A：当某项指标成为最优化目标时（如只刷MMLU分），它不再反映本意或泛化质量，模型可能专门“投机取巧”而非真正提升能力。

---

## 总结与面试建议

* **自动化评测适合横向对比，人工评测关注主观表现，模型判官适合规模化、细粒度评测，反馈信号则直接驱动迭代。**
* 面试最好能对各种评测方式的适用场景、优缺点、工程trade-off有深入理解和具体举例（如错例分析如何反哺数据/训练优化）。
* **Goodhart法则** 是评测系统设计时常被问到的陷阱：不要让某个分数成为唯一追求，否则它就失去了评测的价值。
