# Preference Alignment

Preference alignment is a second stage in the post-training pipeline, focused on aligning generated answers with human preferences. This stage was designed to tune the tone of LLMs and reduce toxicity and hallucinations. However, it has become increasingly important to also boost their performance and improve usefulness. Unlike SFT, there are many preference alignment algorithms. Here, we'll focus on the three most important ones: DPO, GRPO, and PPO.

* Rejection sampling: For each prompt, use the trained model to generate multiple responses, and score them to infer chosen/rejected answers. This creates on-policy data, where both responses come from the model being trained, improving alignment stability.
  Direct Preference Optimization Directly optimizes the policy to maximize the likelihood of chosen responses over rejected ones. It doesn't require reward modeling, which makes it more computationally efficient than RL techniques but slightly worse in terms of quality. Great for creating chat models.
* Reward model: Train a reward model with human feedback to predict metrics like human preferences. It can leverage frameworks like TRL, verl, and OpenRLHF for scalable training.
* Reinforcement Learning: RL techniques like GRPO and PPO iteratively update a policy to maximize rewards while staying close to the initial behavior. They can use a reward model or reward functions to score responses. They tend to be computationally expensive and require careful tuning of hyperparameters, including learning rate, batch size, and clip range. Ideal for creating reasoning models.

---

# Preference Alignment（偏好对齐）

**简介**

Preference Alignment是大模型后训练(post-training)流程的第二阶段，核心目标是让模型输出更贴近人类偏好。其初衷在于调优模型风格、减少有害和幻觉内容，而现在也越来越注重提升模型实际效用和准确性。与SFT仅有一种主流方案不同，偏好对齐包含多种算法与技术。

---

## 1. Rejection Sampling（拒绝采样）

### 关键技术

* 给定同一指令，模型生成多个回答；用评估机制（人类或奖励模型）给每个回答打分。
* 选出表现最好（chosen）和最差（rejected）的样本，形成一组偏好对。
* 这种“on-policy”数据来源于当前在训模型，可降低数据-模型分布偏移，提高训练和对齐稳定性。

### 常见面试题

* **Q: 为什么偏好对齐阶段要使用rejection sampling？**

  **A:** 因为它确保数据和当前模型行为一致，通过比较同一模型多种答案，产生高质量偏好对，减少分布偏移和不稳定。
* **Q: on-policy 和 off-policy 数据有何区别？**

  **A:** on-policy（如rejection sampling）全部用当前模型产出的答案，off-policy常用过去模型/人工数据。on-policy更贴合训练目标，off-policy易引入分布偏移。

---

## 2. Direct Preference Optimization（DPO，直接偏好优化）

### 关键技术

* 不依赖奖励模型，直接用偏好数据进行最大似然优化（Maximize Likelihood of Chosen Over Rejected），常按Bradley-Terry模型的pairwise loss构建损失函数。
* 训练时仅需“指令+chosen+rejected”三元组，避免了奖励模型训练、推理带来的额外资源消耗。
* 极高效率，适合对对话类模型高效偏好对齐。
* 质量略低于PPO/GRPO等RL方法，尤其在复杂推理任务上。

### 常见面试题

* **Q: DPO与PPO最大的工程优势是什么？**

  **A:** DPO完全省略了奖励模型，直接用pairwise loss做优化，训练流程更短更快，所需显存和算力远低于PPO；但复杂推理场景下效果略弱。
* **Q: DPO为什么不需要reward model？**

  **A:** DPO通过直接最大化选中答案概率来“隐式”表达偏好奖励，不需要显式预测或生成奖励分数。

---

## 3. Reward Model（奖励模型）

### 关键技术

* 训练奖励模型（RM），输入为“指令+回答”；输出为分数（标量），通常反映助人度、无害性、真实感等偏好指标。
* 监督数据常采集自人类偏好判断（chosen/rejected pair），可大量扩展。
* 奖励模型为后续RL训练（如PPO/GRPO）或自动化数据评分（rejection sampling/数据过滤）提供基础。
* 主流开源框架有TRL、verl、OpenRLHF等。

### 常见面试题

* **Q: 为什么偏好对齐要训练奖励模型？**

  **A:** 奖励模型能大规模、自动化地估算答案优劣，为RL方法和大批量数据采样提供高效客观打分。
* **Q: 训练奖励模型有哪些难点？**

  **A:** 难点在于泛化和去偏（bias）：模型要能适应未见问题，同时纠正人工偏见，以免奖励模型导向错误目标。

---

## 4. RL方法：GRPO与PPO

### 关键技术

* **PPO（Proximal Policy Optimization）** ：主流的强化学习对齐算法。每轮用奖励模型为新生成答案打分，调整模型参数以最大化奖励；通过clip range保持新老行为接近，降低非平稳风险。
* **GRPO（Generalized Reward Policy Optimization）** ：PPO的推广，能适应更广泛的奖励类型和数据源，细节因论文和实现而异。
* 需要大量算力和资源，调参尤为关键（如学习率、batch size、clip参数等）。
* 适用于高要求推理/评测类场景。

### 常见面试题

* **Q: PPO如何防止策略更新过大造成模型崩溃？**

  **A:** PPO通过clip range限制每次更新的幅度，约束新策略与旧策略KL散度，保证训练稳定。
* **Q: 为什么复杂推理任务优先选用PPO或GRPO？**

  **A:** 这类RL方法通过奖励函数与模型的多轮互动，能更好挖掘推理和高难度问题的最佳解，比DPO等一次性优化更精细但耗时。

---

## 总结

* **数据生成与选择方式（rejection sampling）决定训练稳定性。**
* **DPO适合简单对话与大规模训练，效率高，不需reward model。**
* **奖励模型是自动化偏好评估和RL场景的关键工具。**
* **PPO/GRPO适合追求极致性能和复杂推理，需更强算力和调参。**
