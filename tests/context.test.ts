import { describe, expect, it } from 'bun:test'

import { buildSamples } from '../src/context'

describe('Context builder', () => {
  it('should create correct samples for context size 2', () => {
    const tokens = [0, 1, 2, 3]
    const samples = buildSamples(tokens, 2)
    expect(samples.length).toBe(2)
    expect(samples[0]).toEqual({ context: [0, 1], next: 2 })
    expect(samples[1]).toEqual({ context: [1, 2], next: 3 })
  })

  it('should return empty array if tokens.length <= contextSize', () => {
    const samples = buildSamples([0, 1], 2)
    expect(samples.length).toBe(0)
  })
})
