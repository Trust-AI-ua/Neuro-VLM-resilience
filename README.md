# Neuro-VLM-resilience: Are Neuro-Inspired Multi-Modal Vision-Language Models Resilient to Membership Inference Privacy Leakage?
The growing deployment of multi-modal models (MMs) has introduced new ways for adversaries to access sensitive training data, prompting the need to study privacy leakage. This paper investigates a black-box privacy attack, the Membership Inference Attack (MIA), on Vision-Language Models (VLMs). We present a [neuroscience-inspired topological regularization τ framework](https://toponets.github.io/) to analyze VLM resilience against image-text-based inference privacy attacks. Our main finding is that the neuro VLM variant (where τ > 0) is generally more resilient against privacy attacks, without significantly compromising the model's performance. Our results further show that the MIA success rate drops by 24% on the BLIP model under the COCO dataset, with a mean ROC-AUC.

We evaluate three VLMs:

- **BLIP**
- **PaliGemma-2**
- **ViT-GPT2**

on three datasets:

- **COCO**
- **NoCaps**
- **CC3M**

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

This codebase is built on top of and extends the original  
**[Membership Inference Attacks Against Vision–Language Models](https://github.com/YukeHu/vlm_mia)** repository, in particular their **reference-based non-member attack** (`reference_non_member_inference.py`).

---
