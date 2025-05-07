# 🧠 Mechanistic Interpretability via Learning Differential Equations

---

## 📌 TL;DR

We present intermediate findings from our project, **“Mechanistic Interpretability Via Learning Differential Equations.”** Our goal is to better understand transformers by studying internal computation of toy models like the ODEformer and Time-Series Transformer when processing **time-series numerical data**.

Instead of language models, we focus on **mathematical transformers** that predict or infer underlying differential equations. Our hypothesis: this structured, formal setting makes it easier to probe and interpret the model’s internal representations.

Preliminary results show patterns reminiscent of **numerical differentiation** emerging in the model’s activations — suggesting the transformer learns meaningful internal abstractions. We're excited to continue validating and extending these results.

---

## 🧭 Project Motivation

Mechanistic interpretability aims to answer one big question:

> *What algorithms do neural networks implement internally?*

This involves:
- Identifying features encoded in the model’s activations
- Mapping them to meaningful data patterns or logic
- Understanding *how* and *why* decisions are made

But this is hard — especially in large language models (LLMs), where both the **data** (human language) and the **features** are complex and messy.

So instead, we **simplify the interpretability problem**:

- We study **transformers on structured, mathematical data** (e.g., time-series governed by differential equations)
- We use models like:
  - [**ODEFormer**](https://arxiv.org/abs/2301.12408): Learns symbolic ODEs from data
  - [**Hugging Face Time Series Transformer**](https://huggingface.co/docs/transformers/model_doc/time_series_transformer): Predicts next value in a time series

This cleaner setup makes it easier to reverse-engineer what the transformer is *thinking* — and hopefully transfer those insights back to LLMs.

---

## 🧪 Preliminary Results

- ✅ Built and set up interpretability tooling for both ODEFormer and Time-Series Transformer
- 🔬 Applied techniques such as:
  - **Logit lens**
  - **Attention pattern analysis**
  - **Activation probing**
  - **Sparse-Autoencoders**
- 🧭 Key insight: **Activation patterns in ODEFormer resemble numerical derivative estimation**, suggesting internal computation aligns with the ground truth algorithm

We’re now exploring whether these features are robust and generalizable, and which layers and attention heads contribute to them.

---

## 🔍 Why This Matters for LLM Interpretability

While we’re not directly analyzing language models, there are three reasons this matters:

1. **Toy models reveal core mechanisms**  
   Transformers might use similar internal logic across tasks — even very different ones.

2. **Natural abstractions hypothesis**  
   If transformers naturally learn mathematical structure from real-world data, then we should see similar patterns in LLMs too.

3. **Fundamental understanding is long-term leverage**  
   Just like electromagnetism theory preceded its applications, interpretability might pay off later — if we understand the “gears” of these models now.

---

## 🛠️ Repo Structure

```bash
.
├── models/               # ODEFormer & Time Series Transformer code
├── notebooks/            # Probing experiments & visualizations
├── scripts/              # Training & data preprocessing pipelines
├── results/              # Intermediate findings, charts, logs
└── README.md             # You are here :)
