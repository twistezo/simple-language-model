import { describe, expect, it } from 'bun:test'

import {
  applyAttentionWeightsToEmbeddings,
  applyMultiLayerAttentionWithResidualConnections,
  applySelfAttention,
  calculateScaledAttentionScores,
  convertScoresToProbabilities,
} from '../src/attention'

describe('convertScoresToProbabilities', () => {
  it('should convert scores to probabilities that sum to 1', () => {
    const scores = [1, 2, 3]
    const probabilities = convertScoresToProbabilities(scores)

    const sum = probabilities.reduce((acc, p) => acc + p, 0)
    expect(sum).toBeCloseTo(1, 5)
  })

  it('should give higher probability to higher scores', () => {
    const scores = [1, 2, 3]
    const probabilities = convertScoresToProbabilities(scores)

    expect(probabilities[2]).toBeGreaterThan(probabilities[1]!)
    expect(probabilities[1]).toBeGreaterThan(probabilities[0]!)
  })

  it('should handle equal scores with equal probabilities', () => {
    const scores = [1, 1, 1]
    const probabilities = convertScoresToProbabilities(scores)

    expect(probabilities[0]).toBeCloseTo(1 / 3, 5)
    expect(probabilities[1]).toBeCloseTo(1 / 3, 5)
    expect(probabilities[2]).toBeCloseTo(1 / 3, 5)
  })

  it('should handle negative scores', () => {
    const scores = [-1, 0, 1]
    const probabilities = convertScoresToProbabilities(scores)

    const sum = probabilities.reduce((acc, p) => acc + p, 0)
    expect(sum).toBeCloseTo(1, 5)
    expect(probabilities[2]).toBeGreaterThan(probabilities[1]!)
    expect(probabilities[1]).toBeGreaterThan(probabilities[0]!)
  })

  it('should handle very large scores without overflow', () => {
    const scores = [1000, 1001, 1002]
    const probabilities = convertScoresToProbabilities(scores)

    const sum = probabilities.reduce((acc, p) => acc + p, 0)
    expect(sum).toBeCloseTo(1, 5)
    expect(probabilities.every(p => !isNaN(p) && isFinite(p))).toBe(true)
  })

  it('should handle very small scores', () => {
    const scores = [-1000, -999, -998]
    const probabilities = convertScoresToProbabilities(scores)

    const sum = probabilities.reduce((acc, p) => acc + p, 0)
    expect(sum).toBeCloseTo(1, 5)
  })

  it('should handle single element', () => {
    const probabilities = convertScoresToProbabilities([5])
    expect(probabilities[0]).toBe(1)
  })

  it('should handle zeros', () => {
    const probabilities = convertScoresToProbabilities([0, 0, 0])

    expect(probabilities[0]).toBeCloseTo(1 / 3, 5)
    expect(probabilities[1]).toBeCloseTo(1 / 3, 5)
    expect(probabilities[2]).toBeCloseTo(1 / 3, 5)
  })
})

describe('calculateScaledAttentionScores', () => {
  it('should return empty array for empty input', () => {
    const result = calculateScaledAttentionScores([])
    expect(result).toEqual([])
  })

  it('should return matrix with same dimensions as input', () => {
    const embeddings = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    const weights = calculateScaledAttentionScores(embeddings)

    expect(weights.length).toBe(3)
    expect(weights[0]?.length).toBe(3)
  })

  it('should have rows that sum to 1 (valid probabilities)', () => {
    const embeddings = [
      [1, 0.5],
      [0.5, 1],
    ]
    const weights = calculateScaledAttentionScores(embeddings)

    for (const row of weights) {
      const sum = row.reduce((acc, w) => acc + w, 0)
      expect(sum).toBeCloseTo(1, 5)
    }
  })

  it('should attend more to similar vectors', () => {
    const embeddings = [
      [1, 0, 0],
      [1, 0, 0],
      [0, 0, 1],
    ]
    const weights = calculateScaledAttentionScores(embeddings)

    const row0 = weights[0]
    expect(row0).toBeDefined()
    expect(row0![1]).toBeGreaterThan(row0![2]!)
  })

  it('should handle single embedding', () => {
    const embeddings = [[1, 2, 3]]
    const weights = calculateScaledAttentionScores(embeddings)

    expect(weights.length).toBe(1)
    expect(weights[0]?.length).toBe(1)
    expect(weights[0]?.[0]).toBe(1)
  })

  it('should handle high-dimensional embeddings', () => {
    const dimension = 128
    const embeddings = [
      Array.from({ length: dimension }, () => Math.random()),
      Array.from({ length: dimension }, () => Math.random()),
    ]
    const weights = calculateScaledAttentionScores(embeddings)

    expect(weights.length).toBe(2)
    expect(weights[0]?.length).toBe(2)
  })

  it('should scale by sqrt(dimension)', () => {
    const embeddings = [
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ]
    const weights = calculateScaledAttentionScores(embeddings)

    const row0 = weights[0]!
    expect(row0[0]).toBeCloseTo(row0[1]!, 5)
  })
})

describe('applyAttentionWeightsToEmbeddings', () => {
  it('should return empty array for empty input', () => {
    const result = applyAttentionWeightsToEmbeddings([], [])
    expect(result).toEqual([])
  })

  it('should return weighted sum of embeddings', () => {
    const embeddings = [
      [1, 0],
      [0, 1],
    ]
    const weights = [
      [0.5, 0.5],
      [0.5, 0.5],
    ]
    const result = applyAttentionWeightsToEmbeddings(embeddings, weights)

    expect(result[0]?.[0]).toBeCloseTo(0.5, 5)
    expect(result[0]?.[1]).toBeCloseTo(0.5, 5)
  })

  it('should preserve dimension of embeddings', () => {
    const embeddings = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]
    const weights = [
      [1, 0],
      [0, 1],
    ]
    const result = applyAttentionWeightsToEmbeddings(embeddings, weights)

    expect(result[0]?.length).toBe(4)
    expect(result[1]?.length).toBe(4)
  })

  it('should return original embedding with identity weights', () => {
    const embeddings = [
      [1, 2, 3],
      [4, 5, 6],
    ]
    const weights = [
      [1, 0],
      [0, 1],
    ]
    const result = applyAttentionWeightsToEmbeddings(embeddings, weights)

    expect(result[0]).toEqual([1, 2, 3])
    expect(result[1]).toEqual([4, 5, 6])
  })

  it('should handle full attention to single position', () => {
    const embeddings = [
      [10, 20],
      [30, 40],
    ]
    const weights = [
      [0, 1],
      [1, 0],
    ]
    const result = applyAttentionWeightsToEmbeddings(embeddings, weights)

    expect(result[0]).toEqual([30, 40])
    expect(result[1]).toEqual([10, 20])
  })
})

describe('applySelfAttention', () => {
  it('should return empty array for empty input', () => {
    const result = applySelfAttention([])
    expect(result).toEqual([])
  })

  it('should return same number of embeddings', () => {
    const embeddings = [
      [1, 0],
      [0, 1],
      [1, 1],
    ]
    const result = applySelfAttention(embeddings)

    expect(result.length).toBe(3)
  })

  it('should preserve embedding dimension', () => {
    const embeddings = [[1, 2, 3, 4, 5]]
    const result = applySelfAttention(embeddings)

    expect(result[0]?.length).toBe(5)
  })

  it('should produce deterministic output', () => {
    const embeddings = [
      [1, 0],
      [0, 1],
    ]
    const result1 = applySelfAttention(embeddings)
    const result2 = applySelfAttention(embeddings)

    expect(result1).toEqual(result2)
  })
})

describe('applyMultiLayerAttentionWithResidualConnections', () => {
  it('should return input unchanged with 0 layers', () => {
    const embeddings = [
      [1, 0],
      [0, 1],
    ]
    const result = applyMultiLayerAttentionWithResidualConnections(embeddings, 0)

    expect(result).toEqual(embeddings)
  })

  it('should apply residual connections (values increase)', () => {
    const embeddings = [
      [1, 0],
      [0, 1],
    ]
    const result = applyMultiLayerAttentionWithResidualConnections(embeddings, 1)

    const embedding0 = embeddings[0]!
    const result0 = result[0]!
    const inputMagnitude = Math.sqrt(embedding0[0]! ** 2 + embedding0[1]! ** 2)
    const outputMagnitude = Math.sqrt(result0[0]! ** 2 + result0[1]! ** 2)

    expect(outputMagnitude).toBeGreaterThanOrEqual(inputMagnitude)
  })

  it('should support multiple layers', () => {
    const embeddings = [[1, 1]]
    const result1 = applyMultiLayerAttentionWithResidualConnections(embeddings, 1)
    const result2 = applyMultiLayerAttentionWithResidualConnections(embeddings, 2)

    expect(result2[0]![0]).toBeGreaterThan(result1[0]![0]!)
  })

  it('should handle empty input', () => {
    const result = applyMultiLayerAttentionWithResidualConnections([], 5)
    expect(result).toEqual([])
  })

  it('should produce deterministic output', () => {
    const embeddings = [
      [1, 2],
      [3, 4],
    ]
    const result1 = applyMultiLayerAttentionWithResidualConnections(embeddings, 3)
    const result2 = applyMultiLayerAttentionWithResidualConnections(embeddings, 3)

    expect(result1).toEqual(result2)
  })

  it('should scale values with number of layers', () => {
    const embeddings = [[1, 1]]

    const results = [1, 2, 3, 4, 5].map(layers => {
      const result = applyMultiLayerAttentionWithResidualConnections(embeddings, layers)
      const result0 = result[0]!

      return Math.sqrt(result0[0]! ** 2 + result0[1]! ** 2)
    })

    for (let i = 1; i < results.length; i++) {
      expect(results[i]).toBeGreaterThan(results[i - 1]!)
    }
  })
})
