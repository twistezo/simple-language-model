# AI - LLM

This project shows how a language model predicts the next word, step by step, using plain TypeScript and no machine learning libraries.

It is inspired by the YouTube video: ["How LLMs Actually Generate Text" by LearnThatStack](https://www.youtube.com/watch?v=NKnZYvZA7w4).

The goal is understanding, not performance.

---

## What this project is (and is not)

### ✅ What it is

- A **working next-token prediction model**
- Implements **all 5 steps** from the video: Tokenization → Embeddings → Attention → Logits → Sampling
- Based on **statistics and context (n-grams)** for prediction
- Uses **simplified embeddings** and **self-attention** mechanism
- Supports **temperature** and **Top P (nucleus) sampling**
- Fully deterministic in training, **probabilistic in generation**
- Split into clear, well-typed TypeScript files
- Conceptually aligned with modern LLM architecture

### ❌ What it is NOT

- Not a neural network (no backpropagation, no gradient descent)
- Not trained with deep learning
- Not using learned embeddings (random initialization instead)
- Not "AI magic"

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

- `vocabulary.ts` – word ↔ number mapping (token IDs)
- `tokenizer.ts` – text → tokens (STEP 1)
- `embeddings.ts` – tokens → vectors in semantic space (STEP 2)
- `attention.ts` – self-attention mechanism (STEP 3)
- `context.ts` – context windows (n-grams)
- `model.ts` – statistical language model + sampling with temperature and Top P (STEP 4 & 5)
- `llm.ts` – combines all components into the LLM
- `constants.ts` – default configuration values
- `index.ts` – training + interactive text generation

Each file has **one responsibility**.

---

## Step-by-step: how the model works

Following the 5 steps from the video:

### STEP 1: Tokenization

- Converts text into tokens (word → unique number ID)
- File: `tokenizer.ts`, `vocabulary.ts`
- Learn more: [Tokenization](<https://en.wikipedia.org/wiki/Tokenization_\(lexical_analysis\)\>\)

### STEP 2: Embeddings

- Converts token IDs into vectors in semantic space
- Each token becomes a list of numbers (coordinates)
- Similar words would be positioned close together
- File: `embeddings.ts`
- Note: Real LLMs learn embeddings through training; we initialize randomly

### STEP 3: Attention Mechanism

- Allows the model to understand relationships between tokens
- Implements scaled dot-product self-attention
- Supports multiple layers (simplified transformer depth)
- Includes residual connections
- File: `attention.ts`
- Learn more: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### STEP 4: Probability Distribution

- Counts how often a token follows a given context (n-gram statistics)
- Converts counts to probabilities (like softmax on logits)
- File: `model.ts`

### STEP 5: Sampling

- **Temperature**: Controls randomness
  - `T < 1` → more deterministic (precision)
  - `T = 1` → proportional to probability
  - `T > 1` → more random (creativity)
- **Top P (Nucleus Sampling)**: Only considers tokens whose cumulative probability adds up to P
  - Prevents sampling from very unlikely tokens
  - Maintains diversity within the "nucleus"
- File: `model.ts`

### Autoregressive Generation

- Starts with user input
- Predicts next token using all 5 steps
- Appends token, slides context window
- Repeats for `n` tokens
- File: `llm.ts`, `index.ts`

---

## Pipeline Diagram

```
+------------------+
|   Input Text     |
+------------------+
        |
        v
+------------------+
| STEP 1:          |
| tokenizer.ts     |  text → token IDs
| vocabulary.ts    |
+------------------+
        |
        v
+------------------+
| STEP 2:          |
| embeddings.ts    |  token IDs → vectors
+------------------+
        |
        v
+------------------+
| STEP 3:          |
| attention.ts     |  vectors → context-aware vectors
+------------------+
        |
        v
+------------------+
| STEP 4:          |
| model.ts         |  context → probability distribution
| (n-gram stats)   |
+------------------+
        |
        v
+------------------+
| STEP 5:          |
| model.ts         |  probabilities → sampled token
| (temp + top-p)   |
+------------------+
        |
        v
+------------------+
| Autoregressive   |
| Loop (llm.ts)    |  append token, repeat
+------------------+
        |
        v
   Generated Text
```

---

## How it works in practice

- Enter the beginning of a sentence in the terminal.
- The model generates 5 next words **probabilistically**, so different runs may produce different outputs.
- Examples:

```
Enter the beginning of a sentence: Ala has  
Generated: Ala has a dog

Enter the beginning of a sentence: Computer has  
Generated: Computer has a keyboard
```

- Start words lead the model into **categories**:

| Prompt start   | Likely category |
| -------------- | --------------- |
| `Ala has`      | Animals / Pets  |
| `Computer has` | Technology      |
| `Pizza has`    | Food / Cooking  |
| `Sun rises`    | Nature          |

---

## Configuration

Default values in `constants.ts`:

| Parameter                     | Default | Description                               |
| ----------------------------- | ------- | ----------------------------------------- |
| `DEFAULT_CONTEXT_SIZE`        | 3       | N-gram size (how many words as context)   |
| `DEFAULT_TEMPERATURE`         | 0.7     | Sampling randomness                       |
| `DEFAULT_GENERATION_LENGTH`   | 15      | Number of tokens to generate              |
| `DEFAULT_TOP_P`               | 0.9     | Nucleus sampling threshold                |
| `DEFAULT_EMBEDDING_DIMENSION` | 64      | Size of embedding vectors                 |
| `DEFAULT_ATTENTION_LAYERS`    | 4       | Number of attention layers                |

---

## Why this is similar to real LLMs

| Feature                | This Project          | Real LLMs (GPT, etc.)          |
| ---------------------- | --------------------- | ------------------------------ |
| Tokenization           | ✅ Word-based          | ✅ BPE/WordPiece sub-tokens     |
| Embeddings             | ✅ Random vectors      | ✅ Learned embeddings           |
| Attention              | ✅ Simplified          | ✅ Multi-head, many layers      |
| Probability → Sampling | ✅ Temperature + Top P | ✅ Temperature + Top P/Top K    |
| Autoregressive         | ✅ Token by token      | ✅ Token by token               |
| Training               | ❌ Statistics only     | ✅ Backpropagation              |

---

## How to run

1. Install dependencies: `bun install`
2. Run: `bun run dev`
3. Enter a starting phrase and observe generated text

---

## License

MIT — use it to learn, teach, or experiment.
