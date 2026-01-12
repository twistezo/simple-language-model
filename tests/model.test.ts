import { describe, expect, it } from 'bun:test'

import { sampleNextTokenWithTemperature, SimpleLanguageModel } from '../src/model'

describe('SimpleLanguageModel', () => {
  it('should store and retrieve samples', () => {
    const model = new SimpleLanguageModel()
    const samples = [
      { context: [0, 1], next: 2 },
      { context: [1, 2], next: 3 },
    ]
    model.train(samples)
    expect(model.getRow([0, 1])!.get(2)).toBe(1)
    expect(model.getRow([1, 2])!.get(3)).toBe(1)
  })

  it('should return undefined for unknown context', () => {
    const model = new SimpleLanguageModel()
    expect(model.getRow([9, 9])).toBeUndefined()
  })

  it('should sample next token probabilistically', () => {
    const dist = new Map()
    dist.set(0, 10)
    dist.set(1, 30)

    const counts = [0, 0]
    for (let i = 0; i < 100; i++) {
      const token = sampleNextTokenWithTemperature(dist, 1)
      if (token === 0) counts[0]++
      if (token === 1) counts[1]++
    }
    expect(counts[1]).toBeGreaterThan(counts[0])
  })
})
