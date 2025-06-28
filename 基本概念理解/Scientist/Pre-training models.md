# Pre-training models

Pre-training is a computationally intensive and expensive process. While it's not the focus of this course, it's important to have a solid understanding of **how models are pre-trained**, especially in terms of data and parameters. Pre-training can also be performed by hobbyists at a small scale with <1B models.

1. Data preparation: Pre-training requires massive datasets (e.g., Llama 3.1 was trained on 15 trillion tokens) that need careful curation, cleaning, deduplication, and tokenization. Modern pre-training pipelines implement sophisticated filtering to remove low-quality or problematic content.
2. Distributed training: Combine different parallelization strategies: data parallel (batch distribution), pipeline parallel (layer distribution), and tensor parallel (operation splitting). These strategies require optimized network communication and memory management across GPU clusters.
3. Training optimization: Use adaptive learning rates with warm-up, gradient clipping, and normalization to prevent explosions, mixed-precision training for memory efficiency, and modern optimizers (AdamW, Lion) with tuned hyperparameters.
4. Monitoring: Track key metrics (loss, gradients, GPU stats) using dashboards, implement targeted logging for distributed training issues, and set up performance profiling to identify bottlenecks in computation and communication across devices.
