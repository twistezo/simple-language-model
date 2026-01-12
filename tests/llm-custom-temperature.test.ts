import { describe, expect, it } from 'bun:test'

import { generateText, trainLanguageModel } from '../src/llm'

const data = ['a cat has a tail', 'a cat has whiskers', 'a cat has fur']

describe('Temperature behavior', () => {
  const languageModel = trainLanguageModel(data, 2)

  it('higher temperature produces more diverse outputs', () => {
    const lowTempResults = new Set<string>()
    const highTempResults = new Set<string>()

    for (let i = 0; i < 20; i++) {
      lowTempResults.add(generateText(languageModel, 'a cat', 3, 0.01))
      highTempResults.add(generateText(languageModel, 'a cat', 3, 2.0))
    }

    console.log('low temp:', lowTempResults)
    console.log('high temp:', highTempResults)

    expect(highTempResults.size).toBeGreaterThanOrEqual(lowTempResults.size)
  })
})
