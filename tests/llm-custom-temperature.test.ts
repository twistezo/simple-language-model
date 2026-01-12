import { describe, expect, it } from 'bun:test'

import { generate, trainLLM } from '../src/llm'

const data = ['a cat has a tail', 'a cat has whiskers', 'a cat has fur']

describe('Temperature behavior', () => {
  const llm = trainLLM(data, 2)

  it('higher temperature produces more diverse outputs', () => {
    const lowTempResults = new Set<string>()
    const highTempResults = new Set<string>()

    for (let i = 0; i < 10; i++) {
      lowTempResults.add(generate(llm, 'a cat', 3, 0.01))
      highTempResults.add(generate(llm, 'a cat', 3, 1.5))
    }

    console.log('low temp:', lowTempResults)
    console.log('high temp:', highTempResults)

    expect(highTempResults.size).toBeGreaterThan(lowTempResults.size)
  })
})
