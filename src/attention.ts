import type { EmbeddingVector } from './embeddings'

export type AttentionWeightMatrix = number[][]

export const convertScoresToProbabilities = (scores: number[]): number[] => {
  const maximumScore = Math.max(...scores)
  const exponentiatedScores = scores.map(score => Math.exp(score - maximumScore))
  const totalSum = exponentiatedScores.reduce((accumulator, value) => accumulator + value, 0)

  return exponentiatedScores.map(exponentiatedScore => exponentiatedScore / totalSum)
}

const calculateDotProduct = (vectorA: number[], vectorB: number[]): number =>
  vectorA.reduce((sum, value, index) => sum + value * (vectorB[index] ?? 0), 0)

export const calculateScaledAttentionScores = (
  embeddingSequence: EmbeddingVector[],
): AttentionWeightMatrix => {
  if (embeddingSequence.length === 0) return []

  const embeddingDimension = embeddingSequence[0]?.length ?? 0
  const scalingFactor = Math.sqrt(embeddingDimension)

  return embeddingSequence.map(queryVector =>
    convertScoresToProbabilities(
      embeddingSequence.map(
        keyVector => calculateDotProduct(queryVector, keyVector) / scalingFactor,
      ),
    ),
  )
}

export const applyAttentionWeightsToEmbeddings = (
  embeddingSequence: EmbeddingVector[],
  attentionWeights: AttentionWeightMatrix,
): EmbeddingVector[] => {
  if (embeddingSequence.length === 0) return []

  const embeddingDimension = embeddingSequence[0]?.length ?? 0

  return attentionWeights.map(weightRow =>
    Array.from({ length: embeddingDimension }, (_, dimensionIndex) =>
      weightRow.reduce(
        (sum, weight, positionIndex) =>
          sum + weight * (embeddingSequence[positionIndex]?.[dimensionIndex] ?? 0),
        0,
      ),
    ),
  )
}

export const applySelfAttention = (embeddingSequence: EmbeddingVector[]): EmbeddingVector[] => {
  const attentionWeights = calculateScaledAttentionScores(embeddingSequence)

  return applyAttentionWeightsToEmbeddings(embeddingSequence, attentionWeights)
}

export const applyMultiLayerAttentionWithResidualConnections = (
  embeddingSequence: EmbeddingVector[],
  numberOfLayers: number,
): EmbeddingVector[] => {
  let currentEmbeddings = embeddingSequence

  for (let layerIndex = 0; layerIndex < numberOfLayers; layerIndex++) {
    const attendedEmbeddings = applySelfAttention(currentEmbeddings)

    currentEmbeddings = currentEmbeddings.map((embedding, embeddingIndex) =>
      embedding.map(
        (value, dimensionIndex) =>
          value + (attendedEmbeddings[embeddingIndex]?.[dimensionIndex] ?? 0),
      ),
    )
  }

  return currentEmbeddings
}
