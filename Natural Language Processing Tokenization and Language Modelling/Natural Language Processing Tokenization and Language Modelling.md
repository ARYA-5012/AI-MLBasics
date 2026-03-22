# 🧠 NLP Project 01 — Tokenization & Language Modelling

> Part of a beginner-friendly AI/ML project collection — hands-on implementations of core concepts with clear problem statements and complete, well-commented code.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green?style=flat-square)
![Wikipedia API](https://img.shields.io/badge/wikipedia--api-latest-lightgrey?style=flat-square)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat-square&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Problem Statement

Given a **starting word**, predict a sentence of length **15** using two classic language modelling approaches:

- **Unigram Model** — treats every word as independent (no context)
- **Bigram Model** — predicts the next word based on the previous word

The goal is to build both models **from scratch** on a real-world corpus and observe how context improves the coherence of generated text.

---

## 📂 Project Structure

```
📁 NLP-Tokenization-Language-Modelling/
│
├── 📓 Natural_Language_Processing_Tokenization_and_Language_Modelling.ipynb   # Main notebook
├── 📄 Natural_Language_Processing_Tokenization_and_Language_Modelling.docx    # Lab assignment sheet
└── 📄 README.md
```

---

## 🚀 Approach

The project is implemented in a single, well-structured Jupyter Notebook across **8 cells**, each with a focused responsibility:

---

### Step 1 — Install & Import Dependencies

Install `wikipedia-api` and `nltk`. Download the `punkt` tokenizer and `stopwords` corpus from NLTK.

---

### Step 2 — Build a Real-World Corpus (10,000+ words)

Five Wikipedia articles are fetched programmatically using the `wikipedia-api` library:

- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Deep Learning
- Neural Network

These are concatenated into a single raw corpus guaranteed to exceed **10,000 words**.

---

### Step 3 — Preprocess & Tokenize

The raw text is cleaned using a custom `preprocess()` function:

- Converts to **lowercase**
- Removes **special characters**, digits, and extra whitespace
- Tokenizes into individual word tokens using `nltk.word_tokenize()`

The result is a clean list of tokens ready for modelling.

---

### Step 4 — Build the Unigram Model

Computes the probability of each word **independently**:

```
P(word) = count(word) / total_words
```

Words with no context — the simplest baseline model.

---

### Step 5 — Build the Bigram Model

Computes **conditional probabilities** using the previous word as context:

```
P(word_n | word_n-1) = count(word_n-1, word_n) / count(word_n-1)
```

If a word has no bigram history, the model gracefully **falls back to unigram sampling**.

---

### Step 6 — Sentence Prediction (length = 15)

Two prediction functions are implemented:

| Function | Model | Context Used |
|---|---|---|
| `predict_unigram(start, 15)` | Unigram | None — pure frequency sampling |
| `predict_bigram(start, 15)` | Bigram | Previous word |

Both sample the next word **probabilistically** (using `random.choices` with weights), so each run produces a different output.

---

### Step 7 — Comparison Across Multiple Starting Words

The models are evaluated side-by-side on five seed words:

`machine` · `language` · `data` · `network` · `model`

...making it easy to see the qualitative difference between the two approaches.

---

## 💡 Key Insight

> **Bigram output reads noticeably more natural than Unigram — because context matters.**

The **Unigram** model randomly samples from the entire vocabulary each time, producing incoherent word salad. The **Bigram** model, by conditioning on the previous word, generates sequences that follow real statistical patterns from the corpus — resulting in phrases that feel more like actual language.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| `wikipedia-api` | Corpus collection from Wikipedia |
| `nltk` | Tokenization (`punkt`) |
| `collections.Counter` | Frequency counting |
| `collections.defaultdict` | Bigram table construction |
| `random.choices` | Weighted probabilistic sampling |

---

## ▶️ How to Run

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

**2. Install dependencies**

```bash
pip install wikipedia-api nltk
```

**3. Open the notebook**

```bash
jupyter notebook Natural_Language_Processing_Tokenization_and_Language_Modelling.ipynb
```

**4.** Run all cells **top to bottom** — the corpus is fetched live, so an **internet connection is required** for Cell 3.

**5. Experiment!** Change `starting_word` in Cell 7 to any word from the vocabulary and observe how each model responds.

---

## 📊 Sample Output

```
🚀 Starting Word: 'learning'

════════════════════════════════════════════════════════════
📌 UNIGRAM Prediction (no context):
learning systems or the and to a in of neural as that with is used

📌 BIGRAM Prediction (uses previous word as context):
learning algorithms that are used in the training of deep neural networks
════════════════════════════════════════════════════════════
```

> Actual output will vary on each run due to probabilistic sampling.

---

## 🎓 Concepts Covered

- Corpus construction from live Wikipedia data
- Text preprocessing — lowercasing, regex cleaning
- Tokenization using NLTK
- N-gram language models — Unigram and Bigram
- Maximum Likelihood Estimation (MLE) for probability computation
- Probabilistic text generation with weighted sampling
- Fallback strategies for unseen contexts

---

## 📚 Part of the AI/ML Beginner Projects Collection

This notebook is one of several beginner-friendly projects in this repository, each designed to teach a fundamental AI/ML concept through a clear problem statement and complete, runnable code.

| # | Project | Core Concept |
|---|---|---|
| 01 | Tokenization & Language Modelling | Unigram / Bigram NLP |
| ... | More coming soon... | |

---

## 🤝 Contributing

Found a bug or want to add a **Trigram model**? Pull requests are welcome!
Please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
