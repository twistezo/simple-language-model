import type { TokenIdentifier, Vocabulary } from './vocabulary'

/**
 * Converts input text into an array of token identifiers using the provided vocabulary.
 * The function lowercases the text, splits it into words, and maps each word to its token ID.
 * In training mode, new words are added to the vocabulary; otherwise, unknown words throw an error.
 */
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
