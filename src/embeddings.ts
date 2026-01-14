import { Matrix } from 'ml-matrix'

import type { TokenId } from './vocabulary'

export type EmbeddingLayer = {
  getEmbeddingDimension: () => number
  getEmbeddingForToken: (token: TokenId) => EmbeddingVector
  getEmbeddingsForTokenSequence: (tokens: TokenId[]) => EmbeddingVector[]
  initializeTokenEmbedding: (token: TokenId) => void
}

export type EmbeddingVector = number[]

/**
 * Creates an embedding layer that maps tokens to dense vector representations.
 * Each token is assigned a random normalized vector of the specified dimension.
 *
 * Source: Wikipedia, various blog posts and articles
 */
export const createEmbeddingLayer = (dimension: number): EmbeddingLayer => {
  const embeddings = new Map<TokenId, EmbeddingVector>()

  const initializeTokenEmbedding = (token: TokenId): void => {
    if (!embeddings.has(token)) {
      const random = Array.from({ length: dimension }, () => Math.random() * 2 - 1)
      embeddings.set(token, normalizeToUnitLength(random))
    }
  }

  const getEmbeddingForToken = (token: TokenId): EmbeddingVector => {
    if (!embeddings.has(token)) initializeTokenEmbedding(token)

    return embeddings.get(token)!
  }

  return {
    getEmbeddingDimension: (): number => dimension,
    getEmbeddingForToken,
    getEmbeddingsForTokenSequence: (tokens: TokenId[]): EmbeddingVector[] =>
      tokens.map(getEmbeddingForToken),
    initializeTokenEmbedding,
  }
}

/**
 * Normalizes an embedding vector to unit length.
 *
 * Source: Wikipedia, various blog posts and articles
 */
const normalizeToUnitLength = (vector: EmbeddingVector): EmbeddingVector => {
  const v = Matrix.rowVector(vector)
  const magnitude = v.norm()

  // If the vector has zero magnitude, returns the original vector unchanged.
  return magnitude === 0 ? vector : v.div(magnitude).to1DArray()
}
