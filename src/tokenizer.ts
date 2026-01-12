import type { TokenIdentifier, Vocabulary } from './vocabulary'

export const tokenizeText = (
  inputText: string,
  vocabulary: Vocabulary,
  isTrainingMode: boolean,
): TokenIdentifier[] => {
  const lowercaseText = inputText.toLowerCase()
  const words = lowercaseText.split(/\s+/)

  return words.map(word => {
    const tokenIdentifier = isTrainingMode
      ? vocabulary.addWord(word)
      : vocabulary.encodeWordToToken(word)

    if (tokenIdentifier === undefined) {
      throw new Error(`Unknown word: ${word}`)
    }

    return tokenIdentifier
  })
}
