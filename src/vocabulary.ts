export type TokenId = number

export type Vocabulary = {
  add: (word: string) => TokenId
  decodeTokenToWord: (tokenId: TokenId) => string
  encodeWordToToken: (word: string) => TokenId | undefined
}

export const createVocabulary = (): Vocabulary => {
  const tokenIdToWord: Map<number, string> = new Map<TokenId, string>()
  const wordToTokenId: Map<string, number> = new Map<string, TokenId>()

  const add = (word: string): TokenId => {
    if (!wordToTokenId.has(word)) {
      const tokenId: number = wordToTokenId.size
      wordToTokenId.set(word, tokenId)
      tokenIdToWord.set(tokenId, word)
    }
    return wordToTokenId.get(word)!
  }

  const decodeTokenToWord = (tokenId: TokenId): string => {
    const word: string | undefined = tokenIdToWord.get(tokenId)
    if (word === undefined) {
      throw new Error('Unknown token')
    }
    return word
  }

  const encodeWordToToken = (word: string): TokenId | undefined => wordToTokenId.get(word)

  return { add, decodeTokenToWord, encodeWordToToken }
}
