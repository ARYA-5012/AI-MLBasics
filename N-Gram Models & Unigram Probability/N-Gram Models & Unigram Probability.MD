# NLP Assignment: N-Gram Models & Unigram Probability
### A Comprehensive Guide to the Assignment, Solutions, Code & Concepts

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `nlp_ngram_assignment.py` | Complete Python solution for all 4 sections |
| `README.md` | This file — full explanation of concepts, questions & solutions |

---

## 🚀 How to Run

No external libraries needed. Everything uses Python's standard library.

```bash
# Option 1: Run directly
python nlp_ngram_assignment.py

# Option 2: Google Colab
# Copy each SECTION block into a separate Colab cell and run top-to-bottom
```

**Requirements:** Python 3.6+, no `pip install` needed.

---

## 📚 Background: What is an N-Gram?

Before diving into the assignment, it is important to understand what an **N-gram** is and why it matters in Natural Language Processing.

An **N-gram** is a contiguous sequence of `n` items (words, characters, or tokens) from a given text. The value of `n` determines the type:

| Name | n | Example (from "I love NLP") |
|------|---|------------------------------|
| Unigram | 1 | `["I"]`, `["love"]`, `["NLP"]` |
| Bigram | 2 | `["I love"]`, `["love NLP"]` |
| Trigram | 3 | `["I love NLP"]` |

### Why N-Grams?

N-gram models are the foundation of statistical language modeling. The core idea is the **Markov Assumption**: the probability of a word depends only on the previous `n-1` words, not the entire history.

```
P(word | entire history) ≈ P(word | previous n-1 words)
```

This assumption makes language modeling computationally feasible and is the basis for autocomplete, spell checkers, speech recognition, and early machine translation systems.

---

## 🧩 Section 1: Filtering Stopwords using Regex in N-Gram Models

### The Question

> Given the sentence: *"I am learning NLP because NLP is fascinating and useful."*
> - Write a Regex pattern to remove common stopwords: `["I", "am", "is", "and", "the", "because"]`
> - Use the cleaned text to generate bigrams
> - Ensure stopwords are ignored in bigram formation

### Concept: What are Stopwords?

**Stopwords** are extremely common words that carry little semantic meaning on their own — words like "I", "am", "is", "and", "the". In NLP pipelines, they are routinely removed because:

1. They add noise to models without contributing meaning
2. They inflate bigram counts with meaningless pairs like ("I", "am") or ("is", "the")
3. Removing them results in more meaningful, semantically rich n-grams

### Concept: Regex for NLP

**Regular Expressions (Regex)** are pattern-matching tools. In NLP they are used for tokenization, cleaning, and filtering text. The key regex operator used here is the **word boundary** `\b`.

* `\b` matches the position between a word character and a non-word character
* This ensures we match **whole words only**, not substrings
* Without `\b`, a pattern for `"is"` would also match the `"is"` inside `"fascinating"` — which is wrong

### The Solution Code

```python
sentence  = "I am learning NLP because NLP is fascinating and useful."
stopwords = ["I", "am", "is", "and", "the", "because"]

# Step 1: Build the regex pattern dynamically
pattern = r'\b(' + '|'.join(re.escape(w) for w in stopwords) + r')\b'
# Result: \b(I|am|is|and|the|because)\b
```

**Breaking down the pattern:**

| Component | Meaning |
|-----------|---------|
| `\b` | Word boundary — match at the edge of a word |
| `(` ... `)` | Capturing group — wraps all alternatives |
| `I\|am\|is\|...` | Alternation — match any one of these words |
| `re.escape(w)` | Escapes special regex characters (safe for all words) |
| `\b` (closing) | Word boundary on the other side |

```python
# Step 2: Remove stopwords
cleaned = re.sub(pattern, '', sentence)
# "I am learning NLP because NLP is fascinating and useful."
# → "  learning NLP  NLP  fascinating  useful."

# Step 3: Remove punctuation
cleaned = re.sub(r'[^\w\s]', '', cleaned)
# Removes anything that isn't a word character or space

# Step 4: Collapse extra whitespace
cleaned = re.sub(r'\s+', ' ', cleaned).strip()
# "learning NLP NLP fascinating useful"
```

**Why three separate regex steps?** Each step has a single, clear responsibility. Combining them would create a much harder-to-read and maintain pattern.

```python
# Step 5: Tokenise and generate bigrams
tokens  = cleaned.split()
bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
```

The bigram generation uses a simple **sliding window of size 2** over the token list. For a list of length `n`, there are always `n-1` bigrams.

### The Output Explained

```
Tokens  : ['learning', 'NLP', 'NLP', 'fascinating', 'useful']

Bigrams:
  ('learning', 'NLP')       ← meaningful content pair
  ('NLP', 'NLP')            ← NLP appears twice consecutively
  ('NLP', 'fascinating')    ← meaningful content pair
  ('fascinating', 'useful') ← meaningful content pair
```

Notice that bigrams like `('I', 'am')`, `('am', 'learning')`, or `('is', 'fascinating')` are completely absent — the stopword filter worked correctly.

---

## 📊 Section 2: Identifying N-Gram Probabilities using Regex (MLE)

### The Question

> Given corpus: *"machine learning is amazing. deep learning is powerful. learning is continuous."*
> - Write a Regex pattern to count bigram occurrences
> - Calculate the bigram probability of `"learning is"`

### Concept: Maximum Likelihood Estimation (MLE)

**MLE** is the simplest way to assign probabilities to n-grams. The formula for a bigram is:

```
P(w₂ | w₁) = Count(w₁, w₂) / Count(w₁)
```

Intuitively: *"Out of all times we saw word w₁, what fraction of the time was it followed by w₂?"*

This is called **Maximum Likelihood** because it picks the probability that maximizes the likelihood of observing the training data — it is simply the relative frequency.

### The Solution Code

```python
corpus = "machine learning is amazing. deep learning is powerful. learning is continuous."

# Step 1: Normalise the corpus
corpus_clean  = re.sub(r'[^\w\s]', '', corpus.lower())
# Removes punctuation (the dots after sentences) and lowercases everything

corpus_tokens = corpus_clean.split()
# ['machine','learning','is','amazing','deep','learning','is',
#  'powerful','learning','is','continuous']
```

**Why lowercase?** So "Learning" and "learning" are treated as the same token. Consistent casing is fundamental to any NLP pipeline.

```python
# Step 2: Count unigrams and bigrams
unigram_counts = Counter(corpus_tokens)
bigram_counts  = Counter(
    (corpus_tokens[i], corpus_tokens[i+1])
    for i in range(len(corpus_tokens) - 1)
)
```

`Counter` is a Python dictionary subclass that automatically counts occurrences. The bigram counter uses a **generator expression** — a memory-efficient way to produce all consecutive word pairs.

```python
# Step 3: Apply MLE formula
target_bigram = ("learning", "is")
mle_prob      = bigram_counts[target_bigram] / unigram_counts["learning"]
# = 3 / 3 = 1.0
```

### The Output Explained

```
Unigram counts : {'learning': 3, 'is': 3, ...}
Bigram counts  : {('learning', 'is'): 3, ...}

MLE P(is | learning) = 3/3 = 1.0000
```

The probability is **1.0** — a perfect certainty. This is because in this specific corpus, every single occurrence of "learning" is followed by "is":
* *"machine **learning is** amazing"*
* *"deep **learning is** powerful"*
* *"**learning is** continuous"*

### The Problem This Reveals: The Zero-Probability Problem

What if we asked for `P(amazing | learning)`? The bigram `("learning", "amazing")` never appears, so MLE gives it **probability 0**. In a language model, a single zero probability collapses the **entire** sentence probability to zero — making the model useless for unseen text. This is exactly why **Laplace Smoothing** (Section 3) was invented.

---

## 🔧 Section 3: Constructing an N-Gram Model with Laplace Smoothing

### The Question

> Write a Python function to construct an N-Gram model using **Laplace (Add-One) Smoothing**.
> - Input: a corpus string and N
> - Output: a dictionary where keys are (N-1)-word context tuples and values are next-word probability dictionaries

### Concept: Laplace (Add-One) Smoothing

Laplace Smoothing solves the zero-probability problem by **pretending we saw every possible n-gram at least once**. The formula becomes:

```
P_smooth(w | context) = (Count(context, w) + 1) / (Count(context) + |V|)
```

Where `|V|` is the **vocabulary size** (number of unique words in the corpus).

**Why add `|V|` to the denominator?** Because we added `+1` to the numerator for every word in the vocabulary. To keep the probabilities summing to 1 across all possible next words, we must add `|V|` to the denominator.

**Trade-off:** Laplace smoothing can over-smooth — it gives too much probability to unseen events and takes too much away from seen ones. More advanced methods (Kneser-Ney, Good-Turing) address this, but Laplace is a solid, interpretable baseline.

### The Solution Code — Full Walkthrough

```python
def build_ngram_model(corpus: str, n: int) -> dict:
```

The function signature accepts any `n`, making it fully general — it works for unigrams, bigrams, trigrams, and beyond.

```python
    # Pre-processing
    text   = re.sub(r'[^\w\s]', '', corpus.lower())
    tokens = text.split()
    V      = len(set(tokens))   # vocabulary size
```

`set(tokens)` gives unique words. `len()` counts them. For `"I love NLP and I love machine learning"`, `V = 7` (i, love, nlp, and, machine, learning — 6 unique words... wait, let me recount: i, love, nlp, and, machine, learning = 6).

```python
    # Count contexts and n-grams
    context_counts = defaultdict(int)
    ngram_counts   = defaultdict(int)

    for i in range(len(tokens) - n + 1):
        ngram   = tuple(tokens[i : i + n])   # e.g. ('i', 'love') for n=2
        context = ngram[:-1]                  # all but last word
        context_counts[context] += 1
        ngram_counts[ngram]     += 1
```

`defaultdict(int)` is like a regular dict but returns `0` for missing keys instead of raising a `KeyError`. The loop uses **slice notation** `tokens[i : i+n]` to extract each n-gram window. `tuple()` is used because lists cannot be dictionary keys (they are mutable), but tuples can.

**Example for n=2 on "i love nlp and i love machine learning":**
```
Window 0: ('i', 'love')       context=('i',),       next='love'
Window 1: ('love', 'nlp')     context=('love',),    next='nlp'
Window 2: ('nlp', 'and')      context=('nlp',),     next='and'
Window 3: ('and', 'i')        context=('and',),     next='i'
Window 4: ('i', 'love')       context=('i',),       next='love'   ← same as window 0!
Window 5: ('love', 'machine') context=('love',),    next='machine'
Window 6: ('machine','learning') context=('machine',), next='learning'
```

```python
    # Build smoothed probability table
    model = defaultdict(dict)
    for ngram, cnt in ngram_counts.items():
        context   = ngram[:-1]
        next_word = ngram[-1]
        model[context][next_word] = round(
            (cnt + 1) / (context_counts[context] + V), 6
        )

    return dict(model)
```

For each n-gram, we compute and store its Laplace-smoothed probability. The structure is a **nested dictionary**: `model[context][next_word] = probability`.

### The Output Explained

```
Bigram model (n=2):
  Context ('i',):
    P('love' | ('i',)) = 0.375000
```

Let's verify manually:
* `Count('i', 'love') = 2` (appears twice in "I love NLP and **I love** machine")
* `Count('i') = 2`
* `V = 6`
* `P_smooth = (2+1) / (2+6) = 3/8 = 0.375` ✅

```
  Context ('love',):
    P('nlp'     | ('love',)) = 0.250000
    P('machine' | ('love',)) = 0.250000
```

Verify:
* `Count('love', 'nlp') = 1`, `Count('love', 'machine') = 1`
* `Count('love') = 2`
* `P_smooth = (1+1) / (2+6) = 2/8 = 0.25` ✅

Notice the probabilities **do not sum to 1.0** for a given context — they sum to less than 1 because Laplace implicitly reserves probability mass for every unseen word in the vocabulary.

---

## 📐 Section 4: Perplexity Calculation

### The Question

> Write a function to compute the **perplexity** of a test sentence using an n-gram model trained on a corpus, with Laplace Smoothing.
>
> - `train_corpus = "I love NLP and I love machine learning"`
> - `test_sentence = "I love NLP models"`

### Concept: Perplexity

**Perplexity** is the standard metric for evaluating language models. It measures how "surprised" or "confused" a model is by a test sentence. A lower perplexity is better.

The intuition: if a language model assigns high probability to a test sentence, it means the model "expected" that sentence — it is not perplexed. If it assigns low probability, the model is very confused — high perplexity.

**Mathematical definition:**

```
Perplexity(W) = exp( -1/N × Σ log P(wᵢ | context) )
```

Where:
* `W` = the test sentence
* `N` = number of n-grams evaluated
* `P(wᵢ | context)` = the smoothed probability of each word given its context

This is equivalent to the **geometric mean of the inverse probabilities** of each word — the higher the probabilities, the lower the perplexity.

**Why log probabilities?** Multiplying many small probabilities together leads to **numerical underflow** (the number becomes too small for a computer to represent). Summing log probabilities is mathematically equivalent and numerically stable.

### The Solution Code — Full Walkthrough

```python
def compute_perplexity(train_corpus: str, test_sentence: str, n: int = 2) -> float:

    # Step 1: Build the n-gram model from training data
    model = build_ngram_model(train_corpus, n)

    # Step 2: Get vocabulary size from training data
    train_clean = re.sub(r'[^\w\s]', '', train_corpus.lower()).split()
    V           = len(set(train_clean))
```

**Critical:** The vocabulary size `V` must come from the **training** corpus, not the test sentence. The model only knows about words it was trained on.

```python
    # Step 3: Pre-process the test sentence
    test_tokens = re.sub(r'[^\w\s]', '', test_sentence.lower()).split()
    # "I love NLP models" → ['i', 'love', 'nlp', 'models']
```

```python
    # Step 4: Accumulate log probabilities
    log_prob_sum = 0.0
    num_ngrams   = 0

    for i in range(len(test_tokens) - n + 1):
        ngram   = tuple(test_tokens[i : i + n])
        context = ngram[:-1]
        word    = ngram[-1]

        if context in model and word in model[context]:
            prob = model[context][word]      # seen bigram — use model
        else:
            prob = 1 / V                     # unseen — uniform fallback

        log_prob_sum += math.log(prob)
        num_ngrams   += 1
```

The **fallback probability `1/V`** handles two cases:
1. **Unseen context** — the context was never seen during training
2. **Unseen word in known context** — the context was seen but not with this specific next word

Using `1/V` is equivalent to Laplace smoothing with a count of zero: `(0+1)/(0+V) = 1/V`.

```python
    # Step 5: Compute perplexity
    perplexity = math.exp(-log_prob_sum / num_ngrams)
    return perplexity
```

### The Output Explained — Step by Step

**Test sentence:** "I love NLP models"
**Bigrams extracted:** `('i','love')`, `('love','nlp')`, `('nlp','models')`

```
P('love'   | ('i',))   = 0.375000  → log = -0.9808
```
`('i','love')` was seen in training. Model returns 0.375.

```
P('nlp'    | ('love',)) = 0.250000  → log = -1.3863
```
`('love','nlp')` was seen in training. Model returns 0.25.

```
P('models' | ('nlp',))  = 0.166667  → log = -1.7918
```
`"models"` was **never seen** in training. Fallback: `1/V = 1/6 ≈ 0.1667`.

**Perplexity calculation:**
```
avg_log = (-0.9808 + -1.3863 + -1.7918) / 3
        = -4.1589 / 3
        = -1.3863

Perplexity = exp(-(-1.3863)) = exp(1.3863) ≈ 4.0
```

**Interpreting Perplexity = 4.0:**
The model is about as uncertain as if it were choosing uniformly among 4 words at each step. On a tiny 8-word training corpus, this is actually quite reasonable. A real large language model trained on billions of words achieves perplexity in the single digits on standard benchmarks.

---

## 🧠 Key Concepts Summary Table

| Concept | Formula | Purpose |
|---------|---------|---------|
| **Bigram MLE** | `P(w₂\|w₁) = C(w₁,w₂) / C(w₁)` | Estimate probability from raw counts |
| **Laplace Smoothing** | `P(w\|ctx) = (C(ctx,w)+1) / (C(ctx)+V)` | Prevent zero probabilities for unseen n-grams |
| **Log Probability** | `log P(w\|ctx)` | Numerical stability — avoids underflow |
| **Perplexity** | `exp(-1/N × Σ log P(wᵢ\|ctx))` | Evaluate how well model fits test data |
| **Word Boundary `\b`** | Regex meta-character | Match whole words, not substrings |
| **Vocabulary Size `V`** | `len(set(tokens))` | Number of unique words — used in smoothing |

---

## 🔁 The Complete NLP Pipeline in This Assignment

```
Raw Text
   │
   ▼
[Regex Cleaning] ──────────────────────────────────────────────────────
   │   • Remove stopwords with \b(word)\b
   │   • Remove punctuation with [^\w\s]
   │   • Collapse whitespace with \s+
   │
   ▼
Cleaned Tokens
   │
   ├──► [N-Gram Generation] ──► Sliding window → (w₁,w₂) pairs
   │
   ├──► [Count Collection] ───► Counter(unigrams), Counter(bigrams)
   │
   ├──► [MLE Probability] ────► C(w₁,w₂) / C(w₁)
   │
   ├──► [Laplace Smoothing] ──► (C+1) / (C+V)  →  Language Model
   │
   └──► [Perplexity] ─────────► exp(-1/N × Σ log P)  →  Evaluation Score
```

---

## 💡 Design Decisions & Thinking Behind the Solution

### 1. Why `re.escape()` when building the stopword pattern?

```python
pattern = r'\b(' + '|'.join(re.escape(w) for w in stopwords) + r')\b'
```

`re.escape()` ensures that if any stopword contains a special regex character (like `.` or `*`), it is treated as a literal character. For the stopwords given it makes no difference, but it makes the code **robust and safe** for any input.

### 2. Why case-sensitive for stopword removal in Section 1 but lowercase in Sections 2–4?

Section 1 deliberately keeps the original casing when matching stopwords (removing `"I"` but not `"i"`) to demonstrate precise regex control. Sections 2–4 lowercase everything first with `.lower()` for consistency, which is standard practice in NLP — you generally want `"NLP"` and `"nlp"` to be the same token.

### 3. Why `defaultdict` instead of a regular `dict`?

```python
context_counts = defaultdict(int)
```

With a regular `dict`, you'd need to check `if key in dict` before every increment. `defaultdict(int)` initialises any missing key to `0` automatically, making the counting loop clean and readable.

### 4. Why `tuple` for n-gram keys?

```python
ngram = tuple(tokens[i : i + n])
```

Python dictionaries require **hashable** keys. Lists are mutable and therefore not hashable. Tuples are immutable and hashable, so they work perfectly as dictionary keys.

### 5. Why log probabilities instead of raw multiplication?

Imagine computing the probability of a 100-word sentence with each word having probability ~0.1:
```
0.1¹⁰⁰ = 10⁻¹⁰⁰
```
This is too small for standard 64-bit floating point to represent (underflow to 0.0). Taking logs converts multiplication to addition:
```
log(0.1¹⁰⁰) = 100 × log(0.1) = -100
```
This is a perfectly representable number.

### 6. Why `1/V` as the fallback for unseen n-grams in perplexity?

This is the Laplace-smoothed probability for a word with **zero counts** in a **zero-count context**:
```
P_laplace(w | unseen_context) = (0+1) / (0+V) = 1/V
```
It is the most honest estimate we can make — we have no information, so we treat all V vocabulary words as equally likely.

---

## 📈 Expected Output Summary

| Section | Key Result |
|---------|-----------|
| Section 1 | 4 bigrams from 5 content words, all stopwords removed |
| Section 2 | `P(is \| learning) = 1.0` (MLE, corpus-specific) |
| Section 3 | Smoothed bigram model — `P(love \| i) = 0.375` |
| Section 4 | `Perplexity = 4.0` for "I love NLP models" |

---

## 📖 Further Reading

* Jurafsky & Martin, *Speech and Language Processing* — Chapter 3 (N-gram Language Models) — the canonical textbook reference for all concepts in this assignment
* Manning & Schütze, *Foundations of Statistical NLP* — deeper mathematical treatment of MLE and smoothing methods
* [NLTK Documentation](https://www.nltk.org/) — Python's natural language toolkit, which automates much of what this assignment implements from scratch
* [Python `re` module docs](https://docs.python.org/3/library/re.html) — complete regex reference for Python

---

*Assignment: NLP Fundamentals — N-Gram Models & Unigram Probability*
