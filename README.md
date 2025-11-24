# RAG-Powered-Medical-Symptom-Checker-Agent

## `README.md`
```markdown
# Agents Intensive – Capstone Project  
### Medical Symptom Checker Agent using RAG and LLM Fine-Tuning

---

## 1. Overview

This project was developed for the **Agents Intensive – Capstone Project** Kaggle competition.  
The goal was to build a functional agent that solves a real-world productivity or information problem using LLMs.

This project demonstrates a medical symptom checker agent that uses:

- Retrieval-Augmented Generation (RAG)
- TF-IDF matching
- Sentence-Transformer embeddings
- A fine-tuned Llama-2 model using QLoRA

It processes user symptom descriptions and returns possible matching conditions with appropriate safety disclaimers.

This system is **not intended for diagnosis** and is for educational and research purposes only.

---

## 2. Features

| Feature | Description |
|--------|------------|
| TF-IDF baseline | Rule-based similarity retrieval using cosine distance |
| RAG agent | Embedding-based semantic retrieval feeding a generative LLM |
| Fine-tuned LLM | Llama-2 trained with LoRA adapters on the disease-symptom dataset |
| Safety layer | Automated medical disclaimers, guardrails, and non-deterministic control |

---

## 3. Dataset

- **Source:** *Disease Symptom Prediction* dataset on Kaggle  
- **Availability:** Free and publicly accessible  
- **Use Compliance:** Meets Kaggle competition rules requiring accessible external data at no cost.

Dataset preprocessing steps included:

- Symptom text normalization  
- Consolidation of multiple symptom columns into one
- Duplicate removal
- Prompt-style formatting for fine-tuning and RAG use

---

## 4. Technical Architecture

```

User Input → Preprocessing → Retrieval Layer → LLM Reasoning → Safety Filter → Output

````

### Components:

- **Vectorization:** `TfidfVectorizer` (baseline)
- **Embeddings:** `all-MiniLM-L6-v2` sentence transformer
- **Models:**
  - Llama-2-7B-Chat (quantized)
  - Optional Bio-Mistral as alternative
- **Fine-Tuning:** QLoRA using `peft`, `bitsandbytes`, and `transformers`

---

## 5. Installation

Run in a Kaggle notebook with GPU enabled.

Required packages (install once if needed):

```bash
pip install sentence-transformers peft bitsandbytes accelerate transformers auto-gptq optimum
````

---

## 6. Instructions to Run

1. Run the notebook in sequence from top to bottom.
2. The agent will prompt for symptom text (example: `"fever, cough, chills"`).
3. Choose which model to run:

   * Rule-based
   * RAG agent
   * Fine-tuned model (highest accuracy)

---

## 7. Model Behavior and Limitations

This system:

* Does not provide diagnosis
* Cannot replace medical professionals
* Is trained on a limited symptom dataset and may generalize imperfectly
* May hallucinate without retrieval context (as expected for generative models)

---

## 8. Compliance Statement

This submission follows the Kaggle rules:

* No restricted data or paid tools used
* All external models and datasets are publicly available
* Code and trained artifacts are released under the required open license

---

## 9. License

This project, including original code and fine-tuned results, is released under:

**MIT**

External datasets and pretrained models maintain their original licenses.

---

## 10. Contact

Project Author: *Sathish Kumar*
Submission Type: **Kaggle Agents Intensive Capstone**

```

---
