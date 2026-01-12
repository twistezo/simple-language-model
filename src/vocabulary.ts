export type TokenIdentifier = number

export class Vocabulary {
  private tokenIdentifierToWord = new Map<TokenIdentifier, string>()
  private wordToTokenIdentifier = new Map<string, TokenIdentifier>()

  addWord(word: string): TokenIdentifier {
    if (!this.wordToTokenIdentifier.has(word)) {
      const tokenIdentifier = this.wordToTokenIdentifier.size
      this.wordToTokenIdentifier.set(word, tokenIdentifier)
      this.tokenIdentifierToWord.set(tokenIdentifier, word)
    }

    return this.wordToTokenIdentifier.get(word)!
  }

  decodeTokenToWord(tokenIdentifier: TokenIdentifier): string {
    const word = this.tokenIdentifierToWord.get(tokenIdentifier)
    if (word === undefined) throw new Error('Unknown token')

    return word
  }

  encodeWordToToken(word: string): TokenIdentifier | undefined {
    return this.wordToTokenIdentifier.get(word)
  }
}
