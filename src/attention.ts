import { Matrix } from 'ml-matrix'

import type { EmbeddingVector } from './embeddings'

export type AttentionWeightMatrix = number[][]

/**
 * Applies multiple layers of self-attention to embeddings with residual connections.
 * Each layer computes attention and adds the result back to the input,
 * allowing the model to preserve original information while learning new patterns.
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

const applySelfAttention = (embeddings: EmbeddingVector[]): EmbeddingVector[] =>
  applyAttentionWeightsToEmbeddings(embeddings, calculateScaledAttentionScores(embeddings))

/**
 * Computes attention scores between embedding vectors using scaled dot-product attention.
 * Multiplies the embedding matrix by its transpose, scales by the square root of the
 * embedding dimension, and converts the resulting scores to probability distributions.
 */
const calculateScaledAttentionScores = (embeddings: EmbeddingVector[]): AttentionWeightMatrix => {
  if (embeddings.length === 0) return []

  const E = new Matrix(embeddings)
  const scores = E.mmul(E.transpose()).div(Math.sqrt(E.columns))

  return scores.to2DArray().map(convertScoresToProbabilities)
}

/**
 * Converts an array of scores into a probability distribution using the softmax function.
 * Subtracts the maximum score for numerical stability before applying exponential.
 * The resulting probabilities sum to 1.
 */
const convertScoresToProbabilities = (scores: number[]): number[] => {
  const max = Math.max(...scores)
  const exp = scores.map(s => Math.exp(s - max))
  const sum = exp.reduce((a, b) => a + b, 0)

  return exp.map(e => e / sum)
}

/**
 * Applies attention weights to embedding vectors by performing matrix multiplication.
 * This transforms the embeddings based on the learned attention patterns,
 * allowing the model to focus on relevant parts of the input.
 */
const applyAttentionWeightsToEmbeddings = (
  embeddings: EmbeddingVector[],
  weights: AttentionWeightMatrix,
): EmbeddingVector[] => {
  if (embeddings.length === 0) return []

  return new Matrix(weights).mmul(new Matrix(embeddings)).to2DArray()
}
