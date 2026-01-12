import type { TokenIdentifier } from './vocabulary'

export type EmbeddingLayer = {
  getEmbeddingDimension: () => number
  getEmbeddingForToken: (tokenIdentifier: TokenIdentifier) => EmbeddingVector
  getEmbeddingsForTokenSequence: (tokenSequence: TokenIdentifier[]) => EmbeddingVector[]
  initializeTokenEmbedding: (tokenIdentifier: TokenIdentifier) => void
}

export type EmbeddingVector = number[]

const normalizeToUnitLength = (vector: EmbeddingVector): EmbeddingVector => {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
  if (magnitude === 0) return vector

  return vector.map(value => value / magnitude)
}

export const createEmbeddingLayer = (embeddingDimension: number): EmbeddingLayer => {
  const tokenEmbeddings = new Map<TokenIdentifier, EmbeddingVector>()

  const initializeTokenEmbedding = (tokenIdentifier: TokenIdentifier): void => {
    if (!tokenEmbeddings.has(tokenIdentifier)) {
      const randomVector = Array.from({ length: embeddingDimension }, () => Math.random() * 2 - 1)
      tokenEmbeddings.set(tokenIdentifier, normalizeToUnitLength(randomVector))
    }
  }

  const getEmbeddingForToken = (tokenIdentifier: TokenIdentifier): EmbeddingVector => {
    if (!tokenEmbeddings.has(tokenIdentifier)) {
      initializeTokenEmbedding(tokenIdentifier)
    }

    return tokenEmbeddings.get(tokenIdentifier)!
  }

  const getEmbeddingsForTokenSequence = (tokenSequence: TokenIdentifier[]): EmbeddingVector[] =>
    tokenSequence.map(getEmbeddingForToken)

  const getEmbeddingDimension = (): number => embeddingDimension

  return {
    getEmbeddingDimension,
    getEmbeddingForToken,
    getEmbeddingsForTokenSequence,
    initializeTokenEmbedding,
  }
}
