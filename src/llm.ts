import { buildSamples } from './context'
import { sampleNextTokenWithTemperature, SimpleLanguageModel } from './model'
import { tokenize } from './tokenizer'
import { Vocabulary } from './vocabulary'

export interface LLM {
  contextSize: number
  model: SimpleLanguageModel
  vocab: Vocabulary
}

export function generate(llm: LLM, prompt: string, length: number, temperature: number): string {
  let context = tokenize(prompt, llm.vocab, false)
  const output = [...prompt.split(/\s+/)]

  for (let i = 0; i < length; i++) {
    const row = llm.model.getRow(context)
    if (!row) break

    const next = sampleNextTokenWithTemperature(row, temperature)
    if (next === null) break

    const word = llm.vocab.decode(next)
    output.push(word)
    context = [...context.slice(1), next]
  }

  return output.join(' ')
}

export function trainLLM(texts: string[], contextSize: number): LLM {
  const vocab = new Vocabulary()
  const samples = texts.flatMap(text => buildSamples(tokenize(text, vocab, true), contextSize))

  const model = new SimpleLanguageModel()
  model.train(samples)

  return { contextSize, model, vocab }
}
