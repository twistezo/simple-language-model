import { describe, expect, it } from 'bun:test'

import { buildTrainingSamples } from '../src/context'

describe('Context builder', () => {
  it('should create correct samples for context size 2', () => {
    const tokens = [0, 1, 2, 3]
    const samples = buildTrainingSamples(tokens, 2)
    expect(samples.length).toBe(2)
    expect(samples[0]).toEqual({ contextTokens: [0, 1], nextToken: 2 })
    expect(samples[1]).toEqual({ contextTokens: [1, 2], nextToken: 3 })
  })

  it('should return empty array if tokens.length <= contextSize', () => {
    const samples = buildTrainingSamples([0, 1], 2)
    expect(samples.length).toBe(0)
  })

  it('should handle context size 1', () => {
    const tokens = [0, 1, 2]
    const samples = buildTrainingSamples(tokens, 1)

    expect(samples.length).toBe(2)
    expect(samples[0]).toEqual({ contextTokens: [0], nextToken: 1 })
    expect(samples[1]).toEqual({ contextTokens: [1], nextToken: 2 })
  })

  it('should handle context size 3', () => {
    const tokens = [0, 1, 2, 3, 4]
    const samples = buildTrainingSamples(tokens, 3)

    expect(samples.length).toBe(2)
    expect(samples[0]).toEqual({ contextTokens: [0, 1, 2], nextToken: 3 })
    expect(samples[1]).toEqual({ contextTokens: [1, 2, 3], nextToken: 4 })
  })

  it('should return empty array for empty tokens', () => {
    const samples = buildTrainingSamples([], 2)
    expect(samples).toEqual([])
  })

  it('should return empty array when tokens.length equals contextSize', () => {
    const samples = buildTrainingSamples([0, 1, 2], 3)
    expect(samples).toEqual([])
  })

  it('should return exactly one sample when tokens.length is contextSize + 1', () => {
    const tokens = [0, 1, 2]
    const samples = buildTrainingSamples(tokens, 2)

    expect(samples.length).toBe(1)
    expect(samples[0]).toEqual({ contextTokens: [0, 1], nextToken: 2 })
  })

  it('should create sliding window correctly', () => {
    const tokens = [10, 20, 30, 40, 50]
    const samples = buildTrainingSamples(tokens, 2)

    expect(samples[0]?.contextTokens).toEqual([10, 20])
    expect(samples[0]?.nextToken).toBe(30)

    expect(samples[1]?.contextTokens).toEqual([20, 30])
    expect(samples[1]?.nextToken).toBe(40)

    expect(samples[2]?.contextTokens).toEqual([30, 40])
    expect(samples[2]?.nextToken).toBe(50)
  })

  it('should handle large context size', () => {
    const tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const samples = buildTrainingSamples(tokens, 10)

    expect(samples.length).toBe(1)
    expect(samples[0]?.contextTokens).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    expect(samples[0]?.nextToken).toBe(10)
  })

  it('should handle repeated tokens', () => {
    const tokens = [1, 1, 1, 1]
    const samples = buildTrainingSamples(tokens, 2)

    expect(samples.length).toBe(2)
    expect(samples[0]).toEqual({ contextTokens: [1, 1], nextToken: 1 })
    expect(samples[1]).toEqual({ contextTokens: [1, 1], nextToken: 1 })
  })
})
