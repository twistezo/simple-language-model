# Tiny LLM – Interactive Language Model Demo

This project shows **how a language model predicts the next word**, step by step, using **plain TypeScript** and **no machine learning libraries**.

It is inspired by the YouTube video:  
“How LLMs Actually Generate Text (Every Developer Should Know This)”

The goal is **understanding**, not performance.

---

## What this project is (and is not)

### ✅ What it is

- A **working next-token prediction model**
- Based on **statistics and context (n-grams)**
- Fully deterministic in training, **probabilistic in generation**
- Split into clear, well-typed TypeScript files
- Faithful to how early language models worked
- Conceptually aligned with modern LLMs

### ❌ What it is NOT

- Not a neural network
- Not a transformer
- Not trained with backpropagation
- Not using embeddings or attention
- Not “AI magic”

Everything here is **explicit and visible**.

---

## High-level idea

A language model does only one thing:

**Given the previous words, predict the next word.**

Example:  
If we see `Ala has`, we expect `"a cat"` or `"a dog"` probabilistically.

This project teaches a computer to learn such patterns **from text**, not from rules.

---

## Project structure

`src/` contains:

- `trainingData.ts` – example sentences (categorized)
- `vocabulary.ts` – word ↔ number mapping
- `tokenizer.ts` – text → tokens
- `context.ts` – context windows (n-grams)
- `model.ts` – statistical language model + probabilistic sampling
- `index.ts` – training + interactive text generation

Each file has **one responsibility**.

---

## Step-by-step: how the model works

1. **Training text**
   - Sentences grouped in categories: Animals, Technology, Food, Nature
   - Examples: `"Ala has a cat"`, `"Computer has a keyboard"`
   - File: `trainingData.ts`

2. **Vocabulary (words → numbers)**
   - Maps each unique word to a number token
   - File: `vocabulary.ts`
   - Learn more: [Tokenization](<https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)>)

3. **Tokenization**
   - Converts sentences to arrays of tokens
   - File: `tokenizer.ts`

4. **Context windows (n-grams)**
   - The model looks at the last `N` tokens
   - Example with context size = 2: `[Ala, has] → a`
   - File: `context.ts`
   - Learn more: [N-gram](https://en.wikipedia.org/wiki/N-gram)

5. **Training the model (statistics)**
   - Counts how often a token follows a given context
   - File: `model.ts`

6. **Probabilistic token sampling with temperature**
   - Instead of always picking the most frequent token, the next token is chosen **probabilistically** proportional to frequency
   - Temperature parameter `T` controls randomness:
     - `T < 1` → more deterministic
     - `T = 1` → proportional to frequency
     - `T > 1` → more random
   - File: `model.ts`

7. **Autoregressive text generation**
   - Starts with user input
   - Predicts next token
   - Appends token, slides context window
   - Repeats for `n` tokens
   - File: `index.ts`

---

## Pipeline Diagram (ASCII)

+------------------+
| trainingData.ts |
+------------------+
|
v
+------------------+
| tokenizer.ts | <- text → tokens
+------------------+
|
v
+------------------+
| context.ts | <- build n-grams / context windows
+------------------+
|
v
+------------------+
| model.ts | <- count frequencies & probabilistic sampling
+------------------+
|
v
+------------------+
| index.ts | <- interactive autoregressive generation
+------------------+
|
v
Generated Text

---

## How it works in practice

- Enter the beginning of a sentence in the terminal.
- The model generates 5 next words **probabilistically**, so different runs may produce different outputs.
- Examples:

Enter the beginning of a sentence: Ala has  
Generated: Ala has a dog

Enter the beginning of a sentence: Computer has  
Generated: Computer has a keyboard

- Start words lead the model into **categories**:

| Prompt start   | Likely category |
| -------------- | --------------- |
| `Ala has`      | Animals / Pets  |
| `Computer has` | Technology      |
| `Pizza has`    | Food / Cooking  |
| `Sun rises`    | Nature          |

---

## Why this is similar to real LLMs

- Tokenization → context → next-token prediction → autoregressive generation
- Uses **probabilistic sampling with temperature**, just like GPT
- Small dataset instead of billions of parameters

---

## How to run

1. Compile TypeScript
2. Run `index.ts` in terminal
3. Enter a starting phrase and observe generated text

---

## License

MIT — use it to learn, teach, or experiment.
