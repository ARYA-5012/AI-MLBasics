# 🔬 NLP Project 03 — Advanced Tokenization & Edit Distance

> Part of a beginner-friendly AI/ML project collection — hands-on implementations of core NLP concepts with clear problem statements, complete code, and deep explanatory notes.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Regex](https://img.shields.io/badge/re-built--in-green?style=flat-square)
![Algorithm](https://img.shields.io/badge/Algorithm-Dynamic%20Programming-purple?style=flat-square)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=flat-square&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Overview

This assignment covers **two foundational NLP problems** that every language engineer faces:

1. **Advanced Tokenization using Regular Expressions (Q1)** — 10 real-world text extraction tasks: URLs, emails, hashtags, mentions, numbers, punctuation, PAN numbers, repetitive characters, mobile numbers, and capitalized words.

2. **Edit Distance via Dynamic Programming (Q2)** — Given two strings, compute the minimum number of insertions, deletions, and substitutions required to transform one into the other — and trace every step.

Both problems build the intuition behind how NLP systems parse, normalize, and compare text before it ever reaches a model.

---

## 📂 Project Structure

```
📁 NLP-Advanced-Tokenization-EditDistance/
│
├── 📓 Advanced_Tokenization___Edit_Distance.ipynb   # Full solution notebook (14 cells)
├── 📄 Advanced_Tokenization___Edit_Distance.docx    # Original assignment sheet
└── 📄 README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| `re` (built-in) | Regular expressions for all tokenization tasks |
| Dynamic Programming | Edit distance computation — O(m×n) |
| `re.findall()` | Pattern matching and extraction |
| `re.sub()` | Pattern-based text substitution |

---

## 🗂️ Notebook Structure (14 Cells)

| Cell | Content |
|---|---|
| 0 | Imports |
| 1 | Helper `show_results()` utility |
| 2–11 | Q1: Tasks 1–10 (one cell per task) |
| 12 | Q2: Edit Distance full implementation |
| 13 | Empty / scratch |

---

---

# PART I — Advanced Tokenization with Regex

## 🧠 Why Regex for Tokenization?

Standard tokenizers (like NLTK's `word_tokenize`) split text on whitespace and punctuation — they have no concept of what a URL, an email, or a PAN number is. A regex-based tokenizer lets you define **exactly** what structure you're looking for, treating the pattern itself as the token boundary.

Every solution in this section uses a single import:

```python
import re
```

And a shared helper function that wraps every task:

```python
def show_results(task, text, pattern, flags=0):
    matches = re.findall(pattern, text, flags)
    print(f"📌 Task     : {task}")
    print(f"📝 Input    : {text}")
    print(f"🔍 Pattern  : {pattern}")
    print(f"✅ Matches  : {matches if matches else '⚠️  No matches found'}")
```

**Design thinking:** Rather than writing a `print()` block for every question, a single utility handles formatting. This keeps each solution to 3–4 lines of meaningful logic. It also makes the pattern itself visible in the output — so you can immediately compare the regex against what it matches.

---

## Q1.1 — Extracting URLs

### Problem
URLs contain `.`, `/`, `:`, `?`, `=`, `&` — characters that most tokenizers split on. The goal is to capture the entire URL as one token.

### Pattern
```python
pattern_url = r'https?://[^\s]+'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `https?` | Matches `http` or `https` — the `?` makes the `s` optional |
| `://` | Literal colon + double slash |
| `[^\s]+` | One or more characters that are **not** whitespace |

### Why `[^\s]+` instead of `\w+`?

`\w` matches only `[a-zA-Z0-9_]` — it would stop at `.`, `/`, `?`, `=` which are all valid URL characters. `[^\s]+` means "everything that isn't a space," which correctly captures the entire URL including query strings and paths.

### Test Cases & Output
```
Input   : "I love spending time at https://www.xy123z.com/"
Matches : ['https://www.xy123z.com/']

Input   : "Visit http://openai.com and https://github.com/ARYA-5012 for more."
Matches : ['http://openai.com', 'https://github.com/ARYA-5012']

Input   : "No URL here, just plain text."
Matches : ⚠️  No matches found
```

---

## Q1.2 — Identifying Email IDs

### Problem
Email addresses combine alphanumeric characters with special characters (`@`, `.`, `_`, `+`, `-`). A naive split on `@` gives you two broken tokens.

### Pattern
```python
pattern_email = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `[a-zA-Z0-9._%+\-]+` | Local part — letters, digits, dots, underscores, `%`, `+`, `-` |
| `@` | Literal `@` separator |
| `[a-zA-Z0-9.\-]+` | Domain name — letters, digits, dots, hyphens |
| `\.` | Literal dot before TLD |
| `[a-zA-Z]{2,}` | Top-level domain — at least 2 letters (`com`, `org`, `co.in`, etc.) |

### Why `{2,}` for the TLD?

TLDs range from `io` (2 chars) to `museum` (6 chars). The `{2,}` means "2 or more" — it future-proofs the pattern against new TLDs without needing a hardcoded list.

### Test Cases & Output
```
Input   : "My email ID is xyz111@gmail.com"
Matches : ['xyz111@gmail.com']

Input   : "Contact us at support@company.org or admin@test.in"
Matches : ['support@company.org', 'admin@test.in']

Input   : "No email in this line."
Matches : ⚠️  No matches found
```

---

## Q1.3 — Detecting Hashtags

### Problem
Hashtags start with `#` and contain alphanumeric characters with no spaces. They're a single semantic unit that tokenizers would split into `#` + `word`.

### Pattern
```python
pattern_hashtag = r'#[a-zA-Z0-9_]+'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `#` | Literal hash symbol |
| `[a-zA-Z0-9_]+` | One or more letters, digits, or underscores |

### Key design decision: Why not `\w+`?

`\w` = `[a-zA-Z0-9_]` — so `#\w+` would work identically here. The explicit character class is used instead because it makes the accepted character set transparent to readers. In production, `#\w+` is perfectly fine.

### Test Cases & Output
```
Input   : "#Sushant is trending now in the world."
Matches : ['#Sushant']

Input   : "Loving #MachineLearning and #AI2025 today!"
Matches : ['#MachineLearning', '#AI2025']
```

---

## Q1.4 — Finding Mentions

### Problem
Mentions (`@username`) follow the same structure as hashtags but with `@`. They identify users on social platforms and must be captured as a single token.

### Pattern
```python
pattern_mention = r'@[a-zA-Z0-9_]+'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `@` | Literal `@` symbol |
| `[a-zA-Z0-9_]+` | One or more valid username characters |

### Similarity to hashtags — intentional
The patterns for hashtags and mentions are structurally identical, swapping `#` for `@`. This is by design — both are "prefix + identifier" patterns. In a real tokenizer, you'd handle both in one pass with `[#@][a-zA-Z0-9_]+`.

### Test Cases & Output
```
Input   : "@Ajit, please help me"
Matches : ['@Ajit']

Input   : "Hey @elonmusk and @OpenAI, great work!"
Matches : ['@elonmusk', '@OpenAI']
```

---

## Q1.5 — Extracting Numbers (Integer & Decimal)

### Problem
Numbers appear in many formats — integers (`8853147`), decimals (`3.14159`) — and must be distinguished from dates, version strings, and surrounding words.

### Pattern
```python
pattern_numbers = r'\b\d+\.?\d*\b'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `\b` | Word boundary — prevents partial matches inside words |
| `\d+` | One or more digits (integer part) |
| `\.?` | Optional decimal point |
| `\d*` | Zero or more digits (decimal part — optional) |
| `\b` | Closing word boundary |

### Why `\.?` instead of `\.`?

`\.` would require a decimal point, only matching decimals. `\.?` makes it optional — so both `8853147` (integer) and `3.14159` (float) are matched by the same pattern.

### The `\b` anchors — why critical?

Without word boundaries, a pattern like `\d+` inside a string `"abc123def"` would match `123`. The `\b` anchors ensure only standalone numbers are extracted.

### Test Cases & Output
```
Input   : "8853147 sq. km of area washed away in floods."
Matches : ['8853147']

Input   : "Pi is approximately 3.14159 and e is 2.718."
Matches : ['3.14159', '2.718']
```

---

## Q1.6 — Identifying Punctuation

### Problem
Punctuation marks (`.`, `,`, `!`, `?`, `(`, `)`, `#`, etc.) are distinct tokens that standard tokenizers sometimes merge with adjacent words.

### Pattern
```python
pattern_punct = r'[^\w\s]'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `[^ ... ]` | Negated character class — matches anything NOT listed |
| `\w` | Word characters (`a-zA-Z0-9_`) |
| `\s` | Whitespace characters (space, tab, newline) |

### The logic is elegant negation

Instead of listing every punctuation character (there are dozens), this pattern says: *"if it's not a word character and not whitespace, it must be punctuation."* This covers `.`, `,`, `!`, `?`, `(`, `)`, `#`, `-`, `...`, `—`, and any Unicode punctuation in one character class.

### Test Cases & Output
```
Input   : "Corona virus killed #24506 people. #Corona is un(tolerable)."
Matches : ['#', '.', '#', '(', ')', '.']

Input   : "Wait... Are you serious?! Yes — absolutely."
Matches : ['.', '.', '.', '?', '!', '—', '.']
```

---

## Q1.7 — Validating PAN Numbers

### Problem
Indian PAN (Permanent Account Number) follows a strict government-defined format:
- **5 uppercase letters** — first 3 are alphabetic sequence, 4th encodes entity type, 5th is surname initial
- **4 digits** — serial number
- **1 uppercase letter** — check character

Invalid PANs (lowercase, wrong length, wrong structure) must be rejected.

### Pattern
```python
pattern_pan = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `\b` | Word boundary |
| `[A-Z]{5}` | Exactly 5 uppercase letters |
| `[0-9]{4}` | Exactly 4 digits |
| `[A-Z]{1}` | Exactly 1 uppercase letter |
| `\b` | Word boundary |

### Why does `lEcGD012eg` fail?

The pattern `[A-Z]{5}` requires **all uppercase** letters. `lEcGD` contains lowercase `l` and `e`, `g` — so zero matches are found. The `\b` boundaries also prevent partial matches from being extracted out of a longer string.

### Custom output (not `show_results`)

Q1.7 uses a custom print block instead of the shared helper because it adds a `✅ VALID / ❌ INVALID` label, not just a list of matches. This is good design — the helper is used where it fits, overridden when the task needs different semantics.

### Test Cases & Output
```
Input   : "ABCED3193P"           → ✅ VALID   (matches: ['ABCED3193P'])
Input   : "lEcGD012eg"           → ❌ INVALID (no matches)
Input   : "My PAN is DKPPA1234Z" → ✅ VALID   (matches: ['DKPPA1234Z'])
Input   : "ABCDE123"             → ❌ INVALID (only 3 digits, no final letter)
```

---

## Q1.8 — Removing Repetitive Characters

### Problem
Social media text contains exaggerated spelling: `"heyyy"`, `"verrrry"`, `"loong"`. These are the same word as `"hey"`, `"very"`, `"long"` but won't match in a vocabulary lookup without normalization.

### Approach
Unlike the other tasks that use `re.findall()`, this is a **substitution** task — `re.sub()` is used to collapse repeated characters.

```python
def remove_repetitions(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `(.)` | Capture group — matches any single character |
| `\1` | Backreference — matches the same character captured in group 1 |
| `{2,}` | Two or more times (so total repetitions ≥ 3) |
| `r'\1\1'` | Replacement — keep exactly 2 of the character |

### Why keep 2 instead of 1?

Some legitimate English words use double letters: `"oo"` in `"book"`, `"ee"` in `"feed"`, `"ll"` in `"bell"`. Collapsing to 1 would corrupt valid words. Collapsing **3+ repetitions** to 2 preserves real doubles while removing exaggerated ones. This is the standard heuristic used in text normalization.

### The backreference `\1` — the clever bit

`(.)` captures any character into group 1. `\1{2,}` then matches **that exact same character** repeated 2+ more times. This means the pattern works for any character — `y`, `r`, `o`, `!`, or anything else — in a single general expression.

### Test Cases & Output
```
Original  : "heyyy this is a verrrry loong texttt"
Cleaned   : "heyy this is a verry long textt"

Original  : "Noooo waaay this is happpening"
Cleaned   : "Noo waay this is happening"

Original  : "This is completely normal text."
Cleaned   : "This is completely normal text."  ← unchanged
```

---

## Q1.9 — Finding Indian Mobile Numbers

### Problem
Indian mobile numbers follow a specific national format:
- Exactly **10 digits**
- Must **start with 6, 7, 8, or 9** (numbers starting with 0–5 are landlines or invalid)
- Must be a standalone token (not part of a longer number)

### Pattern
```python
pattern_mobile = r'\b[6-9][0-9]{9}\b'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `\b` | Word boundary |
| `[6-9]` | First digit must be 6, 7, 8, or 9 |
| `[0-9]{9}` | Exactly 9 more digits (total = 10) |
| `\b` | Word boundary |

### Why does `1234567890` fail?

The first character `1` does not match `[6-9]`, so the pattern rejects it. This correctly filters out numbers that start with 0–5.

### Why does `+91 7654321098` partially match?

The `+91` prefix is separated by a space. After the space, `7654321098` is a standalone 10-digit number starting with 7 — so it correctly matches. A production-grade solution would add an optional `(\+91[\s-]?)?` prefix group, but this scope is outside the assignment.

### Test Cases & Output
```
Input   : "9990001796 is a phone number of the PMO office."
Matches : ['9990001796']  ✅

Input   : "Call me on 8800112233 or +91 7654321098."
Matches : ['8800112233', '7654321098']  ✅

Input   : "Landline: 011-23456789 — not a mobile."
Matches : ⚠️  No matches found  ✅ (correctly rejected)

Input   : "Invalid: 1234567890"
Matches : ⚠️  No matches found  ✅ (starts with 1)
```

---

## Q1.10 — Words Starting with Capital Letters

### Problem
Proper nouns, names, and the starts of sentences begin with a capital letter. Extracting these is a primitive form of Named Entity Recognition (NER) — identifying words that are likely names or significant terms.

### Pattern
```python
pattern_capital = r'\b[A-Z][a-zA-Z]*\b'
```

### Pattern Breakdown

| Component | Meaning |
|---|---|
| `\b` | Word boundary |
| `[A-Z]` | First character must be an uppercase letter |
| `[a-zA-Z]*` | Zero or more letters (upper or lower) following |
| `\b` | Word boundary |

### Why `[a-zA-Z]*` and not `[a-z]*`?

Acronyms like `NSA`, `USA`, `SpaceX` are all uppercase or mixed — using `[a-zA-Z]*` captures them correctly. Using `[a-z]*` would match only the first character of `NSA`.

### Limitation worth knowing

This pattern also matches the first word of every sentence (e.g., `"Ajit"` in `"Ajit Doval..."`) regardless of whether it's a proper noun. True NER requires context, not just pattern matching. This is by design for this exercise — the goal is to demonstrate the capability and its boundary.

### Test Cases & Output
```
Input   : "Ajit Doval is the National Security Advisor in India."
Matches : ['Ajit', 'Doval', 'National', 'Security', 'Advisor', 'India']

Input   : "Elon Musk founded SpaceX and Tesla."
Matches : ['Elon', 'Musk', 'SpaceX', 'Tesla']

Input   : "no capital words here at all."
Matches : ⚠️  No matches found
```

---

---

# PART II — Edit Distance (Dynamic Programming)

## 🧠 What is Edit Distance?

**Edit Distance** (also called **Levenshtein Distance**) is a measure of how different two strings are — defined as the **minimum number of single-character operations** needed to transform one string into another.

The three allowed operations are:

| Operation | Cost | Example |
|---|---|---|
| **Insert** | 1 | `"cat"` → `"cats"` (insert `s`) |
| **Delete** | 1 | `"cats"` → `"cat"` (delete `s`) |
| **Replace** | 1 | `"cat"` → `"bat"` (replace `c` with `b`) |

### Real-world applications

- **Spell checkers** — `"teh"` has edit distance 1 from `"the"` → suggest correction
- **DNA sequence alignment** — mutations as edit operations
- **Fuzzy string matching** — search engines, deduplication
- **NLP preprocessing** — comparing tokenized outputs

---

## 🔢 The Algorithm — Dynamic Programming

### Why Dynamic Programming?

The naive recursive approach recalculates the same subproblems exponentially. DP stores results in a table — each cell is computed exactly once.

**Time complexity:** O(m × n)
**Space complexity:** O(m × n)

Where `m = len(str1)` and `n = len(str2)`.

---

### Step 1 — Build the DP Table

```python
m, n = len(str1), len(str2)
dp = [[0] * (n + 1) for _ in range(m + 1)]
```

The table is `(m+1) × (n+1)` — the extra row and column represent transforming **from/to an empty string**.

### Step 2 — Initialize Base Cases

```python
for i in range(m + 1):
    dp[i][0] = i    # Delete all i characters of str1

for j in range(n + 1):
    dp[0][j] = j    # Insert all j characters of str2
```

`dp[i][0] = i` means: to transform the first `i` characters of `str1` into an empty string, you need `i` deletions.

`dp[0][j] = j` means: to transform an empty string into the first `j` characters of `str2`, you need `j` insertions.

### Step 3 — Fill the Table

```python
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if str1[i-1] == str2[j-1]:
            dp[i][j] = dp[i-1][j-1]          # Characters match — no cost
        else:
            dp[i][j] = 1 + min(
                dp[i-1][j],                   # Delete from str1
                dp[i][j-1],                   # Insert into str1
                dp[i-1][j-1]                  # Replace in str1
            )
```

**The recurrence relation — the heart of the algorithm:**

If `str1[i-1] == str2[j-1]`: characters already match → no operation, inherit diagonal cost.

Otherwise, take the cheapest of three options:
- `dp[i-1][j]` + 1 → **delete** `str1[i-1]` (move up in the table)
- `dp[i][j-1]` + 1 → **insert** `str2[j-1]` (move left in the table)
- `dp[i-1][j-1]` + 1 → **replace** `str1[i-1]` with `str2[j-1]` (move diagonally)

The final answer is `dp[m][n]`.

---

### Step 4 — Traceback (Reconstruct Operations)

Computing the distance alone is not enough — the assignment asks for the **sequence of operations**. Traceback walks backward from `dp[m][n]` to `dp[0][0]` following the decision that was made at each cell.

```python
i, j = m, n
while i > 0 or j > 0:

    if i > 0 and j > 0 and str1[i-1] == str2[j-1]:
        i -= 1; j -= 1                        # Match — came from diagonal, no op

    elif j > 0 and (i == 0 or dp[i][j-1] < dp[i-1][j] and dp[i][j-1] <= dp[i-1][j-1]):
        operations.append(f"Insert '{str2[j-1]}' at position {i}")
        j -= 1                                 # Insert — came from left

    elif i > 0 and (j == 0 or dp[i-1][j] < dp[i][j-1] and dp[i-1][j] <= dp[i-1][j-1]):
        operations.append(f"Delete '{str1[i-1]}' at position {i-1}")
        i -= 1                                 # Delete — came from above

    else:
        operations.append(f"Replace '{str1[i-1]}' → '{str2[j-1]}' at position {i-1}")
        i -= 1; j -= 1                         # Replace — came from diagonal

operations.reverse()   # Traceback goes backward → reverse to get forward sequence
```

**Why reverse at the end?** Traceback moves from `dp[m][n]` → `dp[0][0]`, recording operations in **reverse order**. `.reverse()` flips them to present the transformation from str1 → str2.

---

## 📊 Test Case Walkthroughs

### Test 1: `"kitten"` → `"sitting"` (Edit Distance = 3)

```
Step 1: Replace 'k' → 's'   at position 0    (kitten → sitten)
Step 2: Replace 'e' → 'i'   at position 4    (sitten → sittin)
Step 3: Insert  'g'          at position 6    (sittin → sitting)
```

**Visual DP table (partial):**
```
     ""   s   i   t   t   i   n   g
""  [ 0   1   2   3   4   5   6   7 ]
k   [ 1   1   2   3   4   5   6   7 ]
i   [ 2   2   1   2   3   4   5   6 ]
t   [ 3   3   2   1   2   3   4   5 ]
t   [ 4   4   3   2   1   2   3   4 ]
e   [ 5   5   4   3   2   2   3   4 ]
n   [ 6   6   5   4   3   3   2   3 ]
```
Final answer: `dp[6][7] = 3` ✅

---

### Test 2: `"flaw"` → `"lawn"` (Edit Distance = 2)

```
Step 1: Delete  'f'          at position 0    (flaw → law)
Step 2: Insert  'n'          at position 4    (law  → lawn)
```

---

### Test 3: `"distance"` → `"editing"` (Edit Distance = 5)

```
Step 1: Replace 'd' → 'e'   at position 0
Step 2: Replace 'i' → 'd'   at position 1
Step 3: Replace 's' → 'i'   at position 2
Step 4: Delete  't'          at position 3
Step 5: Replace 'n' → 'g'   at position 7
```

---

## 🎓 Concepts Covered

**Part I — Tokenization:**
- Regular expressions (`re.findall`, `re.sub`)
- Character classes `[...]`, negated classes `[^...]`
- Quantifiers (`+`, `*`, `?`, `{n}`, `{n,}`)
- Anchors (`\b` word boundary)
- Backreferences (`\1`) for repeated character detection
- Special sequences (`\d`, `\w`, `\s` and their negations)
- Practical patterns: URL, email, hashtag, mention, number, PAN, mobile
- Text normalization via substitution

**Part II — Edit Distance:**
- Levenshtein distance definition
- Dynamic programming — table construction, base cases, recurrence
- Traceback for operation reconstruction
- Three edit operations: insert, delete, replace
- Time/space complexity analysis (O(m×n))

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/ARYA-5012/<repo-name>.git
cd <repo-name>
```

**2. No external dependencies needed**
```bash
# Everything uses Python's built-in 're' module
python -c "import re; print('Ready.')"
```

**3. Open the notebook**
```bash
jupyter notebook Advanced_Tokenization___Edit_Distance.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/) — no setup required.

**4. Run all cells top-to-bottom.** Each cell is independent and self-contained.

---

## 📚 Part of the AI/ML Beginner Projects Collection

| # | Project | Core Concept |
|---|---|---|
| 01 | Tokenization & Language Modelling | Unigram / Bigram NLP |
| 02 | Advanced Tokenization & Edit Distance | Regex + Dynamic Programming |
| ... | More coming soon... | |

---

## 🤝 Contributing

Found an edge case the regex doesn't handle? Want to add **Trigram distance** or a **Needleman-Wunsch** alignment variant? Pull requests are welcome. Please open an issue first.

---

## 👤 Author

**Arya Yadav**
B.Tech CSE (Data Science & AI/ML) — Bennett University, 2022–2026

[![GitHub](https://img.shields.io/badge/GitHub-ARYA--5012-181717?style=flat-square&logo=github)](https://github.com/ARYA-5012)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-arya--yadav-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/arya-yadav-75804a259)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

*"All models are wrong, but some regex patterns are useful."*
