import { Token, Vocabulary } from './vocabulary'

export function tokenize(text: string, vocab: Vocabulary, training: boolean): Token[] {
  return text
    .toLowerCase()
    .split(/\s+/)
    .map(word => {
      const token = training ? vocab.add(word) : vocab.encode(word)

      if (token === undefined) {
        throw new Error(`Unknown word: ${word}`)
      }

      return token
    })
}
