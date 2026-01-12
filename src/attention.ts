import { Matrix } from 'ml-matrix'

import type { EmbeddingVector } from './embeddings'

export type AttentionWeightMatrix = number[][]

/**
 * Softmax: converts raw scores to probabilities (sum = 1)
 * Uses numerical stability trick: subtract max before exp
 */
export const convertScoresToProbabilities = (scores: number[]): number[] => {
  const max = Math.max(...scores)
  const exp = scores.map(s => Math.exp(s - max))
  const sum = exp.reduce((a, b) => a + b, 0)

  return exp.map(e => e / sum)
}

/**
 * Scaled dot-product attention: scores = (Q @ K^T) / sqrt(d)
 * Then softmax per row to get attention weights
 */
export const calculateScaledAttentionScores = (
  embeddings: EmbeddingVector[],
): AttentionWeightMatrix => {
  if (embeddings.length === 0) return []

  const E = new Matrix(embeddings)
  const scores = E.mmul(E.transpose()).div(Math.sqrt(E.columns))

  return scores.to2DArray().map(convertScoresToProbabilities)
}

/**
 * Apply attention: output = weights @ values
 */
export const applyAttentionWeightsToEmbeddings = (
  embeddings: EmbeddingVector[],
  weights: AttentionWeightMatrix,
): EmbeddingVector[] => {
  if (embeddings.length === 0) return []

  return new Matrix(weights).mmul(new Matrix(embeddings)).to2DArray()
}

export const applySelfAttention = (embeddings: EmbeddingVector[]): EmbeddingVector[] =>
  applyAttentionWeightsToEmbeddings(embeddings, calculateScaledAttentionScores(embeddings))

/**
 * Multi-layer attention with residual connections: output = input + attention(input)
 */
export const applyMultiLayerAttentionWithResidualConnections = (
  embeddings: EmbeddingVector[],
  layers: number,
): EmbeddingVector[] => {
  if (embeddings.length === 0) return []

  let current = new Matrix(embeddings)

  for (let i = 0; i < layers; i++) {
    const attended = new Matrix(applySelfAttention(current.to2DArray()))
    current = current.add(attended)
  }

  return current.to2DArray()
}
