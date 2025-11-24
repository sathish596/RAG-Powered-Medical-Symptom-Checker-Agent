Agents Intensive – Capstone Project
RAG-Powered Medical Symptom Checker Agent
1. Problem Statement & Motivation

When people feel unwell, they often type symptoms into a search engine and get flooded with scattered, sometimes misleading information. Visiting a doctor may not be immediately possible due to cost, distance, or time constraints.

This project implements a medical symptom checker agent that:

Accepts free-text symptom descriptions from users

Suggests possible conditions based on a disease–symptom knowledge base

Clearly reminds users that it is not a medical diagnosis and that they must consult a licensed medical professional

The goal is to demonstrate how LLM agents, RAG, and fine-tuning can be combined into a practical assistant that improves everyday productivity and access to information, while respecting strong safety disclaimers.

2. High-Level System Overview (Agent Design)

The final artifact is an agentic pipeline that routes a user’s symptom query through several components:

Input Layer (User → Agent)

User enters a natural-language description of symptoms (e.g. “fever, chills, headache, sore throat”).

Knowledge Base Layer (Disease–Symptom Data)

A structured dataset of diseases and their associated symptoms is used as the knowledge base.

Symptoms are cleaned, normalized, and combined into text form suitable for both classic NLP methods and dense embeddings.

Retrieval Layer (Two Modes)

TF-IDF + cosine similarity (rule-based baseline)

Sentence embeddings + semantic search (for RAG)

Generation Layer (LLM)

For RAG, the agent retrieves the top-k relevant disease records and passes them, plus the user’s symptoms, into a medical LLM (Llama-2 or BioMistral) to generate a concise, structured answer.

Safety Layer (Disclaimers & Guardrails)

Every response includes a prominent disclaimer that the tool is not a diagnostic system, and that users must consult a clinician, especially in emergencies.

This pipeline behaves like an LLM-powered retrieval agent: it retrieves relevant context, uses a domain-aware LLM to reason about it, and then returns a structured answer with safety messaging.

3. Data & External Resources (Compliance with Competition Rules)
3.1 Dataset (External Data)

There is no competition-provided dataset. Instead, I use a public Kaggle dataset:

Dataset: Disease Symptom Prediction – dataset.csv

Source: Kaggle public dataset by Karthik Udyawar

Access:

Publicly available to all Kaggle users

Free to use for educational / research purposes

Usage in this project:

Only dataset.csv is used

Columns: Disease, Symptom_1, Symptom_2, …

Symptoms are cleaned, merged, and used to build both classical and embedding-based retrieval.

This satisfies the External Data conditions in the rules:

Publicly available

Equally accessible to all participants at no cost

No private or restricted data is used

3.2 External Models & Tools

The project uses the following external models & libraries (all publicly available):

Hugging Face models

TheBloke/Llama-2-7B-Chat-GPTQ (quantized Llama-2 chat model)

NousResearch/Llama-2-7b-chat-hf (for QLoRA fine-tuning)

BioMistral/BioMistral-7B (medical-domain LLM, in the extension part)

Python libraries

pandas, scikit-learn (TF-IDF, cosine similarity)

sentence-transformers (embeddings, semantic search)

transformers, auto-gptq, optimum (for loading and running LLMs)

peft, bitsandbytes, datasets, accelerate (for QLoRA fine-tuning)

All of these are freely accessible with standard open-source or research licenses and meet the “reasonably accessible” standard in the competition rules.

As required by the rules:

I do not claim ownership of these pretrained models.

The organizer can independently obtain them from Hugging Face without undue expense.

My own code in this notebook can be licensed under CC-BY-SA 4.0, as requested.

4. Data Preprocessing & Feature Engineering

Using dataset.csv, I perform the following steps: 

medical_chatbot

Load the data with pandas.read_csv.

Identify symptom columns (all columns starting with "Symptom").

Clean symptom text

Replace underscores (_) with spaces for readability (e.g. high_fever → high fever).

Combine symptoms into a single column

For each disease, all symptoms in Symptom_* columns are:

Filtered for non-missing values

Joined into a single comma-separated string: "fever, cough, sore throat, ...".

Stored in a new column: Symptoms.

Original Symptom_* columns are dropped.

De-duplicate

I drop duplicate rows based on the Symptoms column to avoid repetitive entries.

The resulting dataframe is used for all downstream methods.

This produces a clean, compact dataset of (Disease, Symptoms) pairs suitable for:

Bag-of-words / TF-IDF

Sentence embeddings

Instruction-style prompts for fine-tuning LLMs

5. Baseline Agent: Rule-Based Chatbot (TF-IDF + Cosine Similarity)

As a baseline, I build a rule-based agent:

Group by Disease

For each disease, all symptom strings are joined into a single text string.

Vectorization

Use TfidfVectorizer on the Symptoms column to create a TF-IDF matrix of diseases vs. symptoms.

Inference (Agent Logic)

User types symptoms in free text.

Convert user input into TF-IDF vector.

Compute cosine similarity between user vector and each disease vector.

Select top-k (e.g. top 3) diseases with similarity above a threshold (e.g. 0.2).

Print candidate diseases to user, plus a disclaimer.

Limitations

Sensitive to exact token match, spelling, and phrasing.

Does not generalize well to unseen language.

No generative reasoning; simply matches patterns.

This baseline establishes a simple, fully deterministic agent behavior before moving to RAG + LLMs.

6. RAG-Based Medical Assistant Agent

The next version upgrades the agent to use embeddings + RAG + LLM.

6.1 Embedding Knowledge Base

Corpus Construction

For each (Disease, Symptoms) row, I build a text string:

"Disease: <Disease>. Symptoms: <Symptoms>".

These texts form the corpus (knowledge base).

Sentence Embeddings

Use SentenceTransformer("all-MiniLM-L6-v2") to encode each entry into a dense vector.

Store as corpus_embeddings.

6.2 LLM for Generation

Model Selection

Load a quantized Llama-2 chat model (TheBloke/Llama-2-7B-Chat-GPTQ) using transformers with AutoTokenizer and AutoModelForCausalLM.

Quantization allows inference on Kaggle GPU within reasonable memory limits.

Generation Function

Tokenize a prompt

Call model.generate() with:

max_new_tokens=300

do_sample=True, temperature=0.2, top_p=0.9

Decode the output and remove the prompt prefix to keep the LLM’s answer only.

6.3 RAG Flow (Agent Behavior)

The RAG agent for each user query:

Encode the user’s symptom text into an embedding.

Run util.semantic_search(query_embedding, corpus_embeddings, top_k=2) to retrieve the most relevant medical records.

Construct a prompt that includes:

System instructions: you are a medical assistant.

Retrieved medical records (disease + symptoms).

User’s symptom description.

Explicit instruction:

Suggest top 2 possible diseases.

Be concise and respond in bullet points.

Include a disclaimer that this is not a medical diagnosis.

Pass the prompt to the LLM and return the generated response to the user, plus an extra printed disclaimer line for safety.

This transforms the project into a retrieval-augmented agent: it searches the knowledge base, then reasons over the retrieved context using an LLM.

7. Fine-Tuned LLM Agent (QLoRA on Llama-2)

To push performance further, I fine-tune Llama-2 on the disease–symptom pairs using QLoRA, turning the LLM itself into a specialized classifier/generator.

7.1 Data Formatting for Fine-Tuning

I format each row as an instruction–response pair in the Llama-2 chat style:

"<s>[INST] <Symptoms> [/INST] <Disease>"

Store these in a text column and convert to a Hugging Face Dataset.

7.2 Quantization & PEFT Setup

Load NousResearch/Llama-2-7b-chat-hf with 4-bit quantization via BitsAndBytesConfig.

Wrap the model with LoRA using peft:

Train only low-rank adapters on attention projections (e.g. q_proj, v_proj).

This reduces memory and makes training feasible on Kaggle GPUs.

7.3 Training

Tokenize the formatted dataset with AutoTokenizer.

Create labels as a copy of input_ids for causal language modeling.

Use Trainer with moderate hyperparameters (1 epoch, small batch size, cosine schedule, etc.).

Save the resulting LoRA-adapted weights and tokenizer to /kaggle/working/llama2-med-chatbot.

7.4 Inference with Fine-Tuned Model

Reload the base model with quantization.

Attach the LoRA weights with PeftModel.from_pretrained.

For a new user query:

Format a prompt instructing the model: “List the top 2 possible diseases for these symptoms: …” using the Llama-2 [INST] format.

Generate a deterministic answer (do_sample=False, temperature=0.2).

Extract the part after [/INST] as the agent’s reply.

Print the response plus a safety disclaimer.

This agent removes the explicit retrieval step and relies on the knowledge internalized during fine-tuning.

8. Safety, Limitations & Ethical Considerations

The project is strictly educational and not a medical product.

Every interface clearly states:

Results are not a medical diagnosis.

Users must consult a licensed physician for any health decisions.

In emergencies, users should seek immediate medical care.

Limitations:

Dataset is limited and not clinically comprehensive.

LLMs can hallucinate or over-generalize, even when fine-tuned.

Fine-tuning and sampling parameters (e.g. temperature, top_p) can change outputs between runs.

No real-world clinical validation has been performed.

9. Reproducibility & Environment

To reproduce the results:

Hardware

Kaggle Notebook with GPU (e.g., T4) enabled.

Steps

Add the Kaggle dataset disease-symptom-prediction as an input dataset.

Install required libraries:

auto-gptq, optimum (for quantized LLM inference)

peft, bitsandbytes, datasets, accelerate (for QLoRA fine-tuning)

Run cells in order:

Data preprocessing

Rule-based chatbot

Embeddings + RAG chatbot

QLoRA fine-tuned Llama-2 chatbot

Licensing

My original code and writeup are released under CC-BY-SA 4.0, as required.

Pretrained models and datasets retain their original licenses as published by their creators.

10. Alignment with Agents Intensive Rules

External Data & Tools:

Only publicly available, free datasets and models are used.

All tools meet the “reasonably accessible” standard.

Winner License:

I am prepared to license my code and model training logic under CC-BY-SA 4.0 and provide the full repository and documentation to reproduce the solution.

No Multiple Accounts / Teams:

This submission is prepared under a single Kaggle account and follows all team size and submission limits.

Reproducibility:

The notebook includes all processing steps, model loading, and fine-tuning code necessary to reproduce the results on Kaggle hardware.
