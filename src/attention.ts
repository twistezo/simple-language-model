import { Matrix } from 'ml-matrix'

import type { EmbeddingVector } from './embeddings'

export type AttentionWeightMatrix = number[][]

/**
 * Applies multiple layers of self-attention to embeddings with residual connections.
 */
export const applyMultiLayerAttentionWithResidualConnections = (
  embeddings: EmbeddingVector[],
  layers: number,
): EmbeddingVector[] => {
  if (embeddings.length === 0) return []

  let current = new Matrix(embeddings)
  for (let i = 0; i < layers; i++) {
    const attended: Matrix = new Matrix(applySelfAttention(current.to2DArray()))
    current = current.add(attended)
  }

  return current.to2DArray()
}

export const applySelfAttention = (embeddings: EmbeddingVector[]): EmbeddingVector[] => {
  return applyAttentionWeightsToEmbeddings(embeddings, calculateScaledAttentionScores(embeddings))
}

/**
 * Applies attention weights to embedding vectors by performing matrix multiplication.
 * This transforms the embeddings based on the learned attention patterns,
 * allowing the model to focus on relevant parts of the input.
 */
export const applyAttentionWeightsToEmbeddings = (
  embeddings: EmbeddingVector[],
  weights: AttentionWeightMatrix,
): EmbeddingVector[] => {
  if (embeddings.length === 0) return []

  return new Matrix(weights).mmul(new Matrix(embeddings)).to2DArray()
}

/**
 * Computes attention scores between embedding vectors using scaled dot-product attention.
 */
export const calculateScaledAttentionScores = (
  embeddings: EmbeddingVector[],
): AttentionWeightMatrix => {
  if (embeddings.length === 0) return []

  const E: Matrix = new Matrix(embeddings)
  const scores: Matrix = E.mmul(E.transpose()).div(Math.sqrt(E.columns))

  return scores.to2DArray().map(convertScoresToProbabilities)
}

/**
 * Converts an array of scores into a probability distribution using the softmax function.
 */
export const convertScoresToProbabilities = (scores: number[]): number[] => {
  const max: number = Math.max(...scores)
  const exp: number[] = scores.map(s => Math.exp(s - max))
  const sum: number = exp.reduce((a, b) => a + b, 0)

  return exp.map(e => e / sum)
}
