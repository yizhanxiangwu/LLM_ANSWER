# Supervised Fine-Tuning

SFT turns base models into helpful assistants, capable of answering questions and following instructions. During this process, they learn how to structure answers and reactivate a subset of knowledge learned during pre-training. Instilling new knowledge is possible but superficial: it cannot be used to learn a completely new language. Always prioritize data quality over parameter optimization.

* Training techniques: Full fine-tuning updates all model parameters but requires significant compute. Parameter-efficient fine-tuning techniques like LoRA and QLoRA reduce memory requirements by training a small number of adapter parameters while keeping base weights frozen. QLoRA combines 4-bit quantization with LoRA to reduce VRAM usage. These techniques are all implemented in the most popular fine-tuning frameworks: TRL, Unsloth, and Axolotl.
* Training parameters: Key parameters include learning rate with schedulers, batch size, gradient accumulation, number of epochs, optimizer (like 8-bit AdamW), weight decay for regularization, and warmup steps for training stability. LoRA also adds three parameters: rank (typically 16-128), alpha (1-2x rank), and target modules.
* Distributed training: Scale training across multiple GPUs using DeepSpeed or FSDP. DeepSpeed provides three ZeRO optimization stages with increasing levels of memory efficiency through state partitioning. Both methods support gradient checkpointing for memory efficiency.
* Monitoring: Track training metrics including loss curves, learning rate schedules, and gradient norms. Monitor for common issues like loss spikes, gradient explosions, or performance degradation.

---

## 1. 训练技术（Training Techniques）

**关键技术解释：**

* **全量微调（Full fine-tuning）**
  * 更新模型所有参数，模型可以更彻底地学习新任务，但计算和显存消耗极高，通常只在小模型或特殊需求下采用。
* **参数高效微调（Parameter-efficient fine-tuning）**
  * 例如**LoRA** （Low-Rank Adaptation）：仅训练少量可学习的适配器（adapter）参数，将原始权重冻结，显著减少显存和存储需求。
  * **QLoRA** ：在LoRA基础上，将大模型权重进行4位量化，大幅降低显存占用，使消费级显卡也能运行大模型微调。
  * 这些方法都已在**TRL** 、**Unsloth** 、**Axolotl** 等主流SFT框架中实现。

**常见面试题：**

* Q: LoRA和全量微调相比最大优势是什么？

  A: LoRA只训练少量新增参数（适配器），原始参数冻结，极大减少显存和计算资源；而全量微调所有参数，资源消耗高。
* Q: QLoRA如何降低微调的显存消耗？

  A: QLoRA结合4位量化（权重量化为更低位）和LoRA适配器，仅微调极少量高效参数，进一步压缩显存与计算资源需求。

---

## 2. 训练参数（Training Parameters）

**关键技术解释：**

* 调控训练效果和性能的关键超参数包括：
  * **学习率及调度（Learning rate & scheduler）：** 控制优化步伐，调度器动态调整以适应训练不同阶段。
  * **批量大小（Batch size）、梯度累积（Gradient accumulation）：** 增加批量有效大小却不超显存，提升训练稳定性。
  * **训练轮次（Epochs）：** 完整遍历数据集次数。
  * **优化器和权重衰减（如8-bit AdamW, weight decay）：** 优化模型参数，防止过拟合。
  * **预热步数（Warmup steps）：** 训练初期缓慢升高学习率，防训练不稳定。
* 针对LoRA，还需指定：
  * **秩（rank，通常16-128）：** 控制适配器容量。
  * **alpha（缩放系数，通常为rank的1-2倍）：** 平衡适配器输出影响（$输出 = 原始输出 + \frac{alpha}{rank} \cdot \Delta W$）。
  * **目标模块（Target modules）：** 选择如Attention层的q_proj, v_proj等插入位置。

**常见面试题：**

* Q: 什么是梯度累积，如何提升资源利用效率？

  A: 梯度累积可通过多次小批量累加梯度再更新，大幅减少单步显存压力，实现等价于大批量训练的效果。
* Q: LoRA的alpha和rank参数的调节对模型行为有何影响？

  A: alpha调节适配器缩放幅度；rank越大，适配器表达能力越强，但显存与计算需求增大；一般让$\frac{alpha}{rank} \approx 1$保证输出幅度合适。

---

## 3. 分布式训练（Distributed Training）

**关键技术解释：**

* **DeepSpeed（ZeRO）和FSDP** 都能将模型和优化过程分区，支持多张显卡和超大模型微调。
  * **DeepSpeed ZeRO** 分为0/1/2/3四阶段，分别优化状态分布，3阶段能“全分片”权重、梯度和优化器状态，极大提升显存效率。
  * **FSDP** （Fully Sharded Data Parallel）也是参数、梯度和优化器全分片方案。
* **梯度检查点（Gradient checkpointing）：** 只存部分激活值，反向传播时重新计算未存储部分，用更多计算时间换取极大显存节省。

**常见面试题：**

* Q: ZeRO-2和ZeRO-3的核心区别？

  A: ZeRO-2分区优化器状态和梯度，ZeRO-3再进一步分区模型权重，显存效率最优。
* Q: 梯度检查点机制如何节省资源？

  A: 只保存关键前向激活值，其余在反向阶段重算，显著降低内存占用，适合大模型训练。

---

## 4. 训练监控（Monitoring）

**关键技术解释：**

* 训练期间需重点监控：
  * **损失曲线（Loss curve）** ：跟踪收敛速度与异常
  * **学习率变化、梯度范数（Gradient norm）**
  * **异常检测** ：如损失突然升高、梯度爆炸等，及时调整超参数或应用梯度裁剪

**常见面试题：**

* Q: 如果损失曲线出现异常“spike”（大幅飙升），可能什么原因，怎么做？

  A: 可能是批次异常、学习率过高或梯度爆炸。应立即调低学习率、增加梯度裁剪，上检查样本和数据处理流程。
* Q: 为什么训练过程中要监控梯度范数？

  A: 梯度过大可能引发梯度爆炸导致训练不稳定，监控和限制梯度范数（梯度裁剪）可有效预防该问题。

---

## 总结提示

SFT的核心是让基础大模型习得更“对人有用”的行为，过程涉及一系列工程与算法工具。面试时推荐能解释每种技术背后的原理、适用场景和优缺点，也能通过公式和实际例子阐述你的理解。
