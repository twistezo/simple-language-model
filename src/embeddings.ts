import { Matrix } from 'ml-matrix'

import type { TokenIdentifier } from './vocabulary'

export type EmbeddingLayer = {
  getEmbeddingDimension: () => number
  getEmbeddingForToken: (token: TokenIdentifier) => EmbeddingVector
  getEmbeddingsForTokenSequence: (tokens: TokenIdentifier[]) => EmbeddingVector[]
  initializeTokenEmbedding: (token: TokenIdentifier) => void
}

export type EmbeddingVector = number[]

/**
 * Normalize vector to unit length (L2 normalization)
 */
const normalizeToUnitLength = (vector: EmbeddingVector): EmbeddingVector => {
  const v = Matrix.rowVector(vector)
  const magnitude = v.norm()

  return magnitude === 0 ? vector : v.div(magnitude).to1DArray()
}

export const createEmbeddingLayer = (dimension: number): EmbeddingLayer => {
  const embeddings = new Map<TokenIdentifier, EmbeddingVector>()

  const initializeTokenEmbedding = (token: TokenIdentifier): void => {
    if (!embeddings.has(token)) {
      const random = Array.from({ length: dimension }, () => Math.random() * 2 - 1)
      embeddings.set(token, normalizeToUnitLength(random))
    }
  }

  const getEmbeddingForToken = (token: TokenIdentifier): EmbeddingVector => {
    if (!embeddings.has(token)) initializeTokenEmbedding(token)

    return embeddings.get(token)!
  }

  return {
    getEmbeddingDimension: () => dimension,
    getEmbeddingForToken,
    getEmbeddingsForTokenSequence: tokens => tokens.map(getEmbeddingForToken),
    initializeTokenEmbedding,
  }
}
