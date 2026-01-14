import chalk from 'chalk'

import type { TokenId, Vocabulary } from './vocabulary'

/**
 * Converts input text into an array of token identifiers using the provided vocabulary.
 */
export const tokenizeText = (
  inputText: string,
  vocabulary: Vocabulary,
  isTrainingMode: boolean,
): TokenId[] => {
  const lowercaseText: string = inputText.toLowerCase()
  const words: string[] = lowercaseText.split(/\s+/)

  return words.map((word: string): TokenId => {
    const tokenId: TokenId | undefined = isTrainingMode
      ? vocabulary.add(word)
      : vocabulary.encodeWordToToken(word)

    if (tokenId === undefined) {
      throw new Error(chalk.red(`Unknown word: ${word}`))
    }

    return tokenId
  })
}
