# 🧠 NLP Foundations Toolkit

> *A hands-on, beginner-friendly collection of Natural Language Processing projects — from core text preprocessing to sequence-to-sequence code generation — built to teach NLP concepts through working code.*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NLTK-3.x-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/spaCy-3.x-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" />
  <img src="https://img.shields.io/badge/TextBlob-0.17%2B-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>

---

## 📌 What Is This?

The **NLP Foundations Toolkit** is a curated set of AI/ML mini-projects designed to take a beginner from *zero NLP knowledge* to *building sequence-to-sequence models* — one concept at a time.

Each module is:
- ✅ **Self-contained** — clear problem statement, no hidden dependencies
- ✅ **Runnable on Google Colab** — no local setup required
- ✅ **Explained with comments** — code reads like a textbook
- ✅ **Benchmarked** — outputs and results are verifiable

Whether you're studying for your first ML interview, preparing a college assignment, or adding depth to your portfolio, this toolkit gives you working code *and* conceptual grounding.

---

## 🗂️ Project Structure

```
NLP-Foundations-Toolkit/
│
├── notebooks/
│   └── NLP_Foundations_Toolkit.ipynb    # Core NLP concepts notebook (Colab-ready)
│
├── problem_statements/
│   └── NLP_Foundations_Toolkit.docx     # Problem statement: Code Generation via Seq2Seq
│
├── README.md
└── requirements.txt
```

---

## 🔬 Modules Covered

### 🟢 Module 1 — NLP Foundations Notebook
> **File:** `notebooks/NLP_Foundations_Toolkit.ipynb`

A step-by-step walkthrough of the essential NLP preprocessing pipeline using the three most popular Python NLP libraries.

| # | Topic | Library | What You Learn |
|---|-------|---------|----------------|
| Q1 | Environment Setup | pip / spaCy CLI | Installing NLTK, spaCy, TextBlob + downloading language models |
| Q2 | Library Imports | NLTK, spaCy, TextBlob | Correct import order, NLTK resource downloads (`punkt`, `averaged_perceptron_tagger`) |
| Q3 | Tokenization | NLTK | Word tokenization vs. sentence tokenization; token counts |
| Q4 | POS Tagging | spaCy | Part-of-Speech tags, fine-grained tags, `spacy.explain()` for human-readable descriptions |
| Q5 | Text Cleaning | Python stdlib | Lowercasing, punctuation removal using `str.translate()` and `string.punctuation` |

---

### 🔴 Module 2 — Code Generation via Seq2Seq *(Advanced)*
> **File:** `problem_statements/NLP_Foundations_Toolkit.docx`

Design and evaluate an **end-to-end NLP system** that converts a plain-English description of a programming task into syntactically correct Python code.

**You will:**
1. Preprocess a `(natural language, Python code)` dataset
2. Implement and compare **RNN**, **LSTM**, and **GRU**-based encoder-decoder (seq2seq) architectures
3. Train each model and evaluate performance using standard NLP metrics (BLEU score, token accuracy)

**Recommended Datasets:**

| Dataset | Description | URL |
|---------|-------------|-----|
| CoNaLa | Natural language ↔ Python code pairs (Stack Overflow–sourced) | [conala-corpus.github.io](https://conala-corpus.github.io/) |
| CodeSearchNet | Large-scale NL-to-code dataset (GitHub functions) | [github.com/github/CodeSearchNet](https://github.com/github/CodeSearchNet) |

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| **NLP Libraries** | NLTK, spaCy (`en_core_web_sm`), TextBlob |
| **Deep Learning** | PyTorch (seq2seq module) |
| **Language** | Python 3.8+ |
| **Environment** | Google Colab / Jupyter Notebook |
| **Data Handling** | pandas, JSON |

---

## 🚀 Getting Started

### Option A: Google Colab (Recommended — Zero Setup)

1. Open `NLP_Foundations_Toolkit.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run Cell 1 — it installs all dependencies automatically:
   ```bash
   !pip install nltk spacy textblob
   !python -m spacy download en_core_web_sm
   ```
3. Run cells sequentially — each one is self-contained

---

### Option B: Local Setup

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repo
git clone https://github.com/ARYA-5012/nlp-foundations-toolkit.git
cd nlp-foundations-toolkit

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm

# 5. Launch Jupyter
jupyter notebook notebooks/NLP_Foundations_Toolkit.ipynb
```

**`requirements.txt`**
```
nltk>=3.8
spacy>=3.5
textblob>=0.17
torch>=2.0
pandas>=2.0
jupyter
```

---

## 🧪 Approach & Methodology

### Module 1 — NLP Preprocessing Pipeline

The notebook follows a **bottom-up pipeline approach** — starting from raw text and progressively building richer representations:

```
Raw Text
   │
   ▼
Tokenization (NLTK)       ← Break text into words / sentences
   │
   ▼
POS Tagging (spaCy)        ← Assign grammatical roles to each token
   │
   ▼
Text Cleaning (Python)     ← Normalize: lowercase + remove punctuation
   │
   ▼
Clean, Structured Text     → Ready for downstream NLP tasks
```

**Key design decisions:**
- **NLTK** is used for tokenization because of its straightforward API, rich corpus access, and strong educational community support.
- **spaCy** is used for POS tagging because its statistical models (trained on OntoNotes 5.0) outperform rule-based taggers and the `en_core_web_sm` model provides both coarse POS tags and fine-grained Penn Treebank tags.
- **`str.translate()` for cleaning** is used instead of regex — it runs at C speed and is idiomatic Python for character-level operations.

---

### Module 2 — Seq2Seq Code Generation

The problem uses the **Encoder-Decoder (Seq2Seq) paradigm** — the standard architecture for any task that maps one variable-length sequence to another (translation, summarization, code generation).

```
Natural Language Input
        │
        ▼
   ┌─────────────┐
   │   ENCODER   │   ← Reads input token by token
   │ (RNN/LSTM/  │     Builds a fixed-size context vector
   │    GRU)     │
   └──────┬──────┘
          │  context vector
          ▼
   ┌─────────────┐
   │   DECODER   │   ← Generates Python tokens one by one
   │ (RNN/LSTM/  │     Conditioned on context + previous output
   │    GRU)     │
   └──────┬──────┘
          │
          ▼
   Python Code Output
```

**Model Comparison:**

| Model | Strengths | Limitations |
|-------|-----------|-------------|
| **RNN** | Simple, fast to train | Vanishing gradient on long sequences |
| **LSTM** | Handles long-range dependencies via cell state | More parameters, slower to train |
| **GRU** | Balanced — simpler than LSTM, better than RNN | Slightly less expressive than LSTM on complex tasks |

**Evaluation:** Models are assessed using **BLEU Score** (measures n-gram overlap between generated and reference code), which is the standard metric for code generation and machine translation.

---

## 📚 Research References

1. Yin, P. & Neubig, G. — *"A Syntactic Neural Model for General-Purpose Code Generation"* — ACL 2017
2. Iyer, S. et al. — *"Summarizing Source Code using a Neural Attention Model"* — ACL 2016
3. Ahmad, W.U. et al. — *"CoNaLa: The Code/Natural Language Challenge"* — EMNLP 2018

---

## 💡 Learning Outcomes

By completing this toolkit, you will be able to:

- [ ] Explain the NLP preprocessing pipeline and why each step matters
- [ ] Use NLTK, spaCy, and TextBlob confidently for text analysis tasks
- [ ] Understand the difference between word and sentence tokenization
- [ ] Read and interpret POS tag outputs for downstream NLP classification
- [ ] Design and implement a seq2seq encoder-decoder architecture from scratch
- [ ] Compare RNN, LSTM, and GRU models experimentally and justify architecture choices
- [ ] Evaluate generative NLP models using BLEU score

---

## 🤝 Contributing

Contributions are welcome! If you want to add a new NLP module, improve existing code, or fix a bug:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/add-ner-module`
3. Commit your changes: `git commit -m "Add NER module using spaCy"`
4. Push and open a Pull Request

Please keep notebooks Colab-compatible and add clear comments for each code block.

---

## 👤 Author

**Arya Yadav**
B.Tech CSE (Data Science & AI/ML) — Bennett University, 2026

[![GitHub](https://img.shields.io/badge/GitHub-ARYA--5012-181717?style=flat-square&logo=github)](https://github.com/ARYA-5012)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-arya--yadav-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/arya-yadav-75804a259)
[![Email](https://img.shields.io/badge/Email-aryayadav446%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:aryayadav446@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built to learn. Built to teach. Built to ship.</i>
</p>
