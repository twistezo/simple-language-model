export type TokenIdentifier = number

export type Vocabulary = {
  addWord: (word: string) => TokenIdentifier
  decodeTokenToWord: (tokenIdentifier: TokenIdentifier) => string
  encodeWordToToken: (word: string) => TokenIdentifier | undefined
}

/**
 * Creates a vocabulary manager that maps words to unique token identifiers and vice versa.
 * Provides methods to add words, encode words to tokens, and decode tokens back to words.
 *
 * @see Unit tests for usage examples
 */
export const createVocabulary = (): Vocabulary => {
  const tokenIdentifierToWord = new Map<TokenIdentifier, string>()
  const wordToTokenIdentifier = new Map<string, TokenIdentifier>()

  const addWord = (word: string): TokenIdentifier => {
    if (!wordToTokenIdentifier.has(word)) {
      const tokenIdentifier = wordToTokenIdentifier.size
      wordToTokenIdentifier.set(word, tokenIdentifier)
      tokenIdentifierToWord.set(tokenIdentifier, word)
    }

    return wordToTokenIdentifier.get(word)!
  }

  const decodeTokenToWord = (tokenIdentifier: TokenIdentifier): string => {
    const word = tokenIdentifierToWord.get(tokenIdentifier)
    if (word === undefined) throw new Error('Unknown token')

    return word
  }

  const encodeWordToToken = (word: string): TokenIdentifier | undefined =>
    wordToTokenIdentifier.get(word)

  return { addWord, decodeTokenToWord, encodeWordToToken }
}
