# AI - LLM

This project was written as a hands-on learning exercise to understand how language models work by building one from scratch. It uses plain TypeScript with no machine learning libraries, walking through each step of next-word prediction.

It is inspired by the YouTube video: ["How LLMs Actually Generate Text" by LearnThatStack](https://www.youtube.com/watch?v=NKnZYvZA7w4).

## Details

### What it is?

An implementation of a basic LLM pipeline as a working next-token prediction model based on statistics and context (n-grams).

1. Tokenization
   - Text -> Tokens
2. Embeddings
   - Tokens -> Vectors
3. Transformer
   - Context processing
4. Probabilites
   - Token scores
5. Sampling
   - Select one

## Project structure

`dataset/` contains [rahular/simple-wikipedia](https://huggingface.co/datasets/rahular/simple-wikipedia) data in Parquet format. It consists of 87 MB of 770k rows of text from English Wikipedia.

`src/` contains:

- `attention.ts` – self-attention mechanism
- `constants.ts` – default configuration values
- `context.ts` – context windows (n-grams)
- `embeddings.ts` – tokens -> vectors in semantic space
- `index.ts` – training + interactive text generation
- `llm.ts` – combines all components into the LLM
- `model.ts` – statistical language model + sampling with temperature and Top P
- `tokenizer.ts` – text -> tokens
- `vocabulary.ts` – word <-> number mapping (token IDs)

## Step-by-step

### 1. Tokenization

- Converts text into tokens (word -> unique number ID)
- File: `tokenizer.ts`, `vocabulary.ts`
- [Wikipedia: Tokenization](https://en.wikipedia.org/wiki/Large_language_model#Tokenization)

### 2. Embeddings

- Converts token IDs into vectors (lists of numbers) where similar words are positioned close together in semantic space
- File: `embeddings.ts`
- Note: Real LLMs learn embeddings through training. We initialize randomly.
- [Wikipedia: Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)

### 3. Attention Mechanism

- Helps the model understand relationships between tokens by computing attention scores across the context
- Note: Attention is used here as a conceptual demonstration—it shows how the mechanism works, but without a neural network it cannot be used for prediction.
- File: `attention.ts`
- [Wikipedia: Attention](<https://en.wikipedia.org/wiki/Attention_(machine_learning)>)

### 4. Probability Distribution

- Counts how often each word follows a given context and converts those counts into probabilities
- File: `model.ts`

### 5. Sampling

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

## Configuration

Default values in `constants.ts`:

| Parameter                     | Default | Description                             |
| ----------------------------- | ------- | --------------------------------------- |
| `DEFAULT_CONTEXT_SIZE`        | 3       | N-gram size (how many words as context) |
| `DEFAULT_TEMPERATURE`         | 0.7     | Sampling randomness                     |
| `DEFAULT_GENERATION_LENGTH`   | 15      | Number of tokens to generate            |
| `DEFAULT_TOP_P`               | 0.9     | Nucleus sampling threshold              |
| `DEFAULT_EMBEDDING_DIMENSION` | 64      | Size of embedding vectors               |
| `DEFAULT_ATTENTION_LAYERS`    | 4       | Number of attention layers              |

## How to run

1. Install dependencies: `bun install`
2. Run: `bun run dev`
3. Enter a starting phrase and observe generated text
