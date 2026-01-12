import type { TokenIdentifier } from './vocabulary'

export type EmbeddingVector = number[]

export class EmbeddingLayer {
  private embeddingDimension: number
  private tokenEmbeddings = new Map<TokenIdentifier, EmbeddingVector>()

  constructor(embeddingDimension: number) {
    this.embeddingDimension = embeddingDimension
  }

  getEmbeddingForToken = (tokenIdentifier: TokenIdentifier): EmbeddingVector => {
    if (!this.tokenEmbeddings.has(tokenIdentifier)) {
      this.initializeTokenEmbedding(tokenIdentifier)
    }

    return this.tokenEmbeddings.get(tokenIdentifier)!
  }

  getEmbeddingsForTokenSequence = (tokenSequence: TokenIdentifier[]): EmbeddingVector[] =>
    tokenSequence.map(this.getEmbeddingForToken)

  getEmbeddingDimension = (): number => this.embeddingDimension

  initializeTokenEmbedding = (tokenIdentifier: TokenIdentifier): void => {
    if (!this.tokenEmbeddings.has(tokenIdentifier)) {
      const randomVector = Array.from(
        { length: this.embeddingDimension },
        () => Math.random() * 2 - 1,
      )
      this.tokenEmbeddings.set(tokenIdentifier, this.normalizeToUnitLength(randomVector))
    }
  }

  private normalizeToUnitLength = (vector: EmbeddingVector): EmbeddingVector => {
    const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
    if (magnitude === 0) return vector

    return vector.map(value => value / magnitude)
  }
}
