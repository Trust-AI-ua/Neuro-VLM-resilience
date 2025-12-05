# Neuro-VLM-resilience: Are Neuro-Inspired Multi-Modal Vision-Language Models Resilient to Membership Inference Privacy Leakage?
The growing deployment of multi-modal models (MMs) has introduced new ways for adversaries to access sensitive training data, prompting the need to study privacy leakage. This paper investigates a black-box privacy attack, the Membership Inference Attack (MIA), on Vision-Language Models (VLMs). We present a [Neuroscience-inspired topological regularization (τ) framework](https://toponets.github.io/) to analyze VLM resilience on image-text-based inference privacy attacks. Our main finding is that the neuro VLM variant (where τ > 0) is generally more resilient against privacy attacks, without significantly compromising the model's performance. Our results further show that the MIA success rate drops by 24% on the BLIP model under the COCO dataset, with a mean ROC-AUC. We evaluate three VLMs, [BLIP](https://huggingface.co/docs/transformers/model_doc/blip), [PaliGemma-2](https://huggingface.co/blog/paligemma2), and [ViT-GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning), on three datasets, [COCO](https://cocodataset.org/), [NoCaps](https://nocaps.org/), and [CC3M](https://github.com/google-research-datasets/conceptual-captions),

under three threat models/τ-regularization levels:

- **BASELINE**: τ = 0  
- **NEURO**: τ = 2  
- **NEURO++**: τ = 3  

We measure similarity using:

- **MPNet** (semantic similarity)
- **ROUGE-2** (lexical overlap)

and report **reference-based MIA performance** (ROC-AUC).

More details can be found in our paper:
David Amebley, Sayanton Dibbo, "[Are Neuro-Inspired Multi-Modal Vision-Language Models Resilient to Membership Inference Privacy Leakage?](https://arxiv.org/abs/2511.20710)"

Our paper was inspired by and extends the **Reference Inference** attack type by Hu, Yuke, et al. This codebase is also built on top of and extends the original **[Membership Inference Attacks Against Vision–Language Models](https://github.com/YukeHu/vlm_mia)** repository, in particular their **reference-based non-member attack** (`reference_non_member_inference.py`).

## The MIA Pipeline
![MIA Pipeline Overview Diagram](./imgs/Overview-diagram.jpg)

In the Overview diagram illustrated above, our MIA pipeline consists of four main steps:
1. **VLM Fine-tuning**: fine-tune pre-trained VLMs with topological regularization $(\tau)$
2. **Caption Generation**: models generate captions for member and non-member image-text sets
3. **Similarity Analysis**: compute similarity of generated captions with ground-truth captions using MPNet and ROUGE-2
4. **Membership Inference**: perform a black-box membership inference attack

## Performance Comparison among BASELINE and $\tau$-regularized Neuroscience-inspired Models on the COCO Dataset

| Dataset | Model         | Threat Model | MPNet ↓ (Member) | MPNet ↓ (Non-Member) | ROUGE-2 ↓ (Member) | ROUGE-2 ↓ (Non-Member) | ROC-AUC ↑      |
|---------|---------------|--------------|------------------|----------------------|--------------------|------------------------|----------------|
| COCO    | BLIP          | Baseline     | 0.723            | 0.663                | 0.249              | 0.171                  | 94.00 ± 9.90   |
|         |               | Neuro        | 0.797            | 0.793                | 0.425              | 0.380                  | 70.88 ± 18.95  |
|         |               | Neuro++      | 0.698            | 0.687                | 0.319              | 0.312                  | 63.46 ± 10.88  |
|         | PaliGemma 2   | Baseline     | 0.605            | 0.603                | 0.227              | 0.203                  | 70.86 ± 19.53  |
|         |               | Neuro        | 0.717            | 0.712                | 0.111              | 0.102                  | 69.98 ± 13.48  |
|         |               | Neuro++      | 0.735            | 0.732                | 0.338              | 0.318                  | 66.76 ± 13.57  |
|         | ViT-GPT2      | Baseline     | 0.771            | 0.773                | 0.318              | 0.304                  | 55.51 ± 16.47  |
|         |               | Neuro        | 0.773            | 0.782                | 0.357              | 0.373                  | 29.38 ± 10.58  |
|         |               | Neuro++      | 0.775            | 0.776                | 0.357              | 0.368                  | 38.39 ± 10.28  |

Please see more detailed evaluation and model performance comparison in our [paper](https://arxiv.org/abs/2511.20710).
