# Quantization

Quantization is the process of converting the parameters and activations of a model using a lower precision. For example, weights stored using 16 bits can be converted into a 4-bit representation. This technique has become increasingly important to reduce the computational and memory costs associated with LLMs.

* Base techniques: Learn the different levels of precision (FP32, FP16, INT8, etc.) and how to perform naïve quantization with absmax and zero-point techniques.
* GGUF & llama.cpp: Originally designed to run on CPUs, llama.cpp and the GGUF format have become the most popular tools to run LLMs on consumer-grade hardware. It supports storing special tokens, vocabulary, and metadata in a single file.
* GPTQ & AWQ: Techniques like GPTQ/EXL2 and AWQ introduce layer-by-layer calibration that retains performance at extremely low bitwidths. They reduce catastrophic outliers using dynamic scaling, selectively skipping or re-centering the heaviest parameters.
* SmoothQuant & ZeroQuant: New quantization-friendly transformations (SmoothQuant) and compiler-based optimizations (ZeroQuant) help mitigate outliers before quantization. They also reduce hardware overhead by fusing certain ops and optimizing dataflow.


## 1. Base Techniques（基础量化技术）

**关键解释：**

* 量化是将神经网络中的权重和激活从高精度（如FP32）转换为低精度（如FP16, INT8, 4-bit等），以降低模型的存储和计算成本。
* **FP32（32位浮点）、FP16（16位浮点）、INT8（8位整数）**等均为常见精度。
* 基础量化有两类常用方法：
  * **Absmax** ：将所有权重按最大绝对值映射到低精度区间。
  * **Zero-point** ：记录最小值并将数值“平移”，便于还原、提升量化后的分布拟合。

**常见面试题：**

* Q：为什么8位或4位量化可以极大提升推理速度和节省内存？

  A：低位宽数据占用更少内存，也能被CPU/GPU专用指令高效处理，适合大规模并行推理。
* Q：absmax和zero-point量化方法的主要思路？

  A：absmax直接缩放到区间最大绝对值，zero-point则将最小值平移为0，再缩放，避免量化后数值偏移。

---

## 2. GGUF & llama.cpp

**关键解释：**

* **llama.cpp** 是一套为CPU推理优化的Llama系列大语言模型C++实现，侧重在消费级硬件和边缘设备部署。
* **GGUF格式** 是最新模型存储标准，将权重、特殊token、词表、元数据等全部打包进一个单一文件，方便跨设备加载与管理，大幅简化部署流程。

**常见面试题：**

* Q：GGUF格式相比传统存储格式有何优势？

  A：将权重、词表、元信息等整合在一个文件，易于管理和跨平台部署，且支持新版本模型的扩展需求。
* Q：llama.cpp为什么适用于消费级硬件？

  A：优化了低精度推理流程，支持多级量化（如4/5/8位），高效利用CPU内存带宽，使普通PC也能流畅跑大模型。

---

## 3. GPTQ & AWQ

**关键解释：**

* **GPTQ/EXL2** ：基于分层校准的极低比特（如4-bit）量化方案。逐层动校准（calibration），最大限度保留权重低精度下的性能。
* **AWQ（Activation-aware Weight Quantization）** ：类似GPTQ，进一步引入动态缩放和对大幅值（outlier）权重的特殊处理，包括选择性跳过/重中心处理，避免关键信息损失，主流于LLM推理社区。
* 关键：通过“动态调整量化窗”来覆盖极值，保证推理时的输出精度。

**常见面试题：**

* Q：GPTQ量化方法怎么能在极低位宽下保持模型性能？

  A：逐层校准每层权重分布，结合重采样和动态缩放，有效控制精度损失，防止推断性能剧烈下降。
* Q：AWQ如何处理量化时的catastrophic outlier问题？

  A：动态检测最重权重，选择跳过、不量化或重新分布它们，从而减少“信息塌缩”现象。

---

## 4. SmoothQuant & ZeroQuant

**关键解释：**

* **SmoothQuant** ：训练前通过Transform平滑化权重和激活分布（如重参数化、缩放），消除极端异常值，使量化更友好，模型精度下降更小。
* **ZeroQuant** ：结合编译器级优化与op融合，在量化前预先处理数据流，减缓硬件瓶颈，尤其提升INT4/INT8大模型推理性能。

**常见面试题：**

* Q：SmoothQuant的主要目标是什么？

  A：通过训练后对模型做变换，消除“突变值”，让模型更适合低精度量化，最终保持推理准确率。
* Q：ZeroQuant如何结合量化和编译器优化？

  A：在量化流程中自动识别可融合的算子和数据路径，通过编译器优化减轻硬件瓶颈，提高低精度推理吞吐。

---

## 总结

* 模型量化是降本增效的关键技术，可极大推动大模型在边缘设备、消费级硬件部署落地。
* 各类新兴方法（如GGUF/AWQ/SmoothQuant等）既有模型工程改进，也有数学、硬件与编译器的协作优化。
* 面试建议：重点准备常见量化精度、主流低比特/分层量化方法、异常值处理，还有“为什么能显著降低成本但性能损失有限”原理。
