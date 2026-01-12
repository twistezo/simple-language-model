import { describe, expect, it } from 'bun:test'

import { generateText, trainLanguageModel } from '../src/llm'

const data = ['a cat is small', 'a cat is fast', 'a dog is loyal', 'a dog is friendly']

describe('Context size behavior', () => {
  it('different context sizes lead to different continuations', () => {
    const languageModel1 = trainLanguageModel(data, 1)
    const languageModel2 = trainLanguageModel(data, 2)

    const output1 = generateText(languageModel1, 'cat', 3, 0.1)
    const output2 = generateText(languageModel2, 'a cat', 3, 0.1)

    console.log('context=1:', output1)
    console.log('context=2:', output2)

    expect(output1.length).toBeGreaterThan(0)
    expect(output2.length).toBeGreaterThan(0)
    expect(output1).not.toBe(output2)
  })
})
