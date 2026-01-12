export type Token = number

export class Vocabulary {
  private idToWord = new Map<Token, string>()
  private wordToId = new Map<string, Token>()

  add(word: string): Token {
    if (!this.wordToId.has(word)) {
      const id = this.wordToId.size
      this.wordToId.set(word, id)
      this.idToWord.set(id, word)
    }
    return this.wordToId.get(word)!
  }

  decode(token: Token): string {
    const word = this.idToWord.get(token)
    if (!word) throw new Error('Unknown token')
    return word
  }

  encode(word: string): Token | undefined {
    return this.wordToId.get(word)
  }
}
