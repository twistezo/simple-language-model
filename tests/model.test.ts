import { describe, expect, it } from 'bun:test'

import {
  NgramLanguageModel,
  sampleNextToken,
  sampleNextTokenWithNucleusSampling,
  sampleNextTokenWithTemperature,
} from '../src/model'

describe('NgramLanguageModel', () => {
  it('should store and retrieve samples', () => {
    const model = new NgramLanguageModel()
    const samples = [
      { contextTokens: [0, 1], nextToken: 2 },
      { contextTokens: [1, 2], nextToken: 3 },
    ]
    model.trainOnSamples(samples)
    expect(model.getNextTokenDistribution([0, 1])!.get(2)).toBe(1)
    expect(model.getNextTokenDistribution([1, 2])!.get(3)).toBe(1)
  })

  it('should return undefined for unknown context', () => {
    const model = new NgramLanguageModel()
    expect(model.getNextTokenDistribution([9, 9])).toBeUndefined()
  })

  it('should accumulate counts for repeated samples', () => {
    const model = new NgramLanguageModel()
    const samples = [
      { contextTokens: [0, 1], nextToken: 2 },
      { contextTokens: [0, 1], nextToken: 2 },
      { contextTokens: [0, 1], nextToken: 2 },
      { contextTokens: [0, 1], nextToken: 3 },
    ]
    model.trainOnSamples(samples)

    const distribution = model.getNextTokenDistribution([0, 1])!
    expect(distribution.get(2)).toBe(3)
    expect(distribution.get(3)).toBe(1)
  })

  it('should handle multiple next tokens for same context', () => {
    const model = new NgramLanguageModel()
    const samples = [
      { contextTokens: [0, 1], nextToken: 10 },
      { contextTokens: [0, 1], nextToken: 20 },
      { contextTokens: [0, 1], nextToken: 30 },
    ]
    model.trainOnSamples(samples)

    const distribution = model.getNextTokenDistribution([0, 1])!
    expect(distribution.size).toBe(3)
    expect(distribution.get(10)).toBe(1)
    expect(distribution.get(20)).toBe(1)
    expect(distribution.get(30)).toBe(1)
  })

  it('should handle empty samples array', () => {
    const model = new NgramLanguageModel()
    model.trainOnSamples([])

    expect(model.getNextTokenDistribution([0, 1])).toBeUndefined()
  })
})

describe('sampleNextTokenWithTemperature', () => {
  it('should sample next token probabilistically', () => {
    const dist = new Map<number, number>()
    dist.set(0, 10)
    dist.set(1, 30)

    const counts = [0, 0]
    for (let i = 0; i < 100; i++) {
      const token = sampleNextTokenWithTemperature(dist, 1)
      if (token === 0) counts[0]!++
      if (token === 1) counts[1]!++
    }
    expect(counts[1]!).toBeGreaterThan(counts[0]!)
  })

  it('should return null for empty distribution', () => {
    const dist = new Map<number, number>()
    const token = sampleNextTokenWithTemperature(dist, 1)

    expect(token).toBeNull()
  })

  it('should always return the only token for single-element distribution', () => {
    const dist = new Map<number, number>()
    dist.set(42, 1)

    for (let i = 0; i < 10; i++) {
      expect(sampleNextTokenWithTemperature(dist, 1)).toBe(42)
    }
  })

  it('should be more deterministic with low temperature', () => {
    const dist = new Map<number, number>()
    dist.set(0, 100)
    dist.set(1, 1)

    let highTokenCount = 0
    for (let i = 0; i < 50; i++) {
      if (sampleNextTokenWithTemperature(dist, 0.1) === 0) {
        highTokenCount++
      }
    }

    // With very low temperature, should almost always pick the highest count
    expect(highTokenCount).toBeGreaterThan(45)
  })

  it('should be more random with high temperature', () => {
    const dist = new Map<number, number>()
    dist.set(0, 10)
    dist.set(1, 10)
    dist.set(2, 10)

    const counts = [0, 0, 0]
    for (let i = 0; i < 300; i++) {
      const token = sampleNextTokenWithTemperature(dist, 2.0)
      if (token !== null && token >= 0 && token < counts.length) counts[token]!++
    }

    // With equal counts and high temp, should be roughly uniform
    expect(counts[0]).toBeGreaterThan(50)
    expect(counts[1]).toBeGreaterThan(50)
    expect(counts[2]).toBeGreaterThan(50)
  })
})

describe('sampleNextTokenWithNucleusSampling', () => {
  it('should return null for empty distribution', () => {
    const distribution = new Map<number, number>()
    const token = sampleNextTokenWithNucleusSampling(distribution, 0.9, 1)

    expect(token).toBeNull()
  })

  it('should always return the only token for single-element distribution', () => {
    const distribution = new Map<number, number>()
    distribution.set(99, 5)

    for (let i = 0; i < 10; i++) {
      expect(sampleNextTokenWithNucleusSampling(distribution, 0.9, 1)).toBe(99)
    }
  })

  it('should limit sampling to top probability tokens', () => {
    const distribution = new Map<number, number>()
    distribution.set(0, 100)
    distribution.set(1, 10)
    distribution.set(2, 1)

    const sampled = new Set<number>()
    for (let i = 0; i < 100; i++) {
      const token = sampleNextTokenWithNucleusSampling(distribution, 0.5, 1)
      if (token !== null) sampled.add(token)
    }

    expect(sampled.has(0)).toBe(true)
  })

  it('should include more tokens with higher topP', () => {
    const distribution = new Map<number, number>()
    distribution.set(0, 40)
    distribution.set(1, 30)
    distribution.set(2, 20)
    distribution.set(3, 10)

    const sampledLowP = new Set<number>()
    const sampledHighP = new Set<number>()

    for (let i = 0; i < 200; i++) {
      const tokenLow = sampleNextTokenWithNucleusSampling(distribution, 0.3, 1)
      const tokenHigh = sampleNextTokenWithNucleusSampling(distribution, 0.99, 1)
      if (tokenLow !== null) sampledLowP.add(tokenLow)
      if (tokenHigh !== null) sampledHighP.add(tokenHigh)
    }

    expect(sampledHighP.size).toBeGreaterThanOrEqual(sampledLowP.size)
  })
})

describe('sampleNextToken', () => {
  it('should use temperature sampling when topP is undefined', () => {
    const dist = new Map<number, number>()
    dist.set(0, 100)

    const token = sampleNextToken(dist, 1)
    expect(token).toBe(0)
  })

  it('should use TopP sampling when topP is between 0 and 1', () => {
    const dist = new Map<number, number>()
    dist.set(0, 100)

    const token = sampleNextToken(dist, 1, 0.9)
    expect(token).toBe(0)
  })

  it('should use temperature sampling when topP is 0 or 1', () => {
    const dist = new Map<number, number>()
    dist.set(0, 100)

    expect(sampleNextToken(dist, 1, 0)).toBe(0)
    expect(sampleNextToken(dist, 1, 1)).toBe(0)
  })
})
