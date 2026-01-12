import { describe, expect, it } from 'bun:test'

import { generate, trainLLM } from '../src/llm'

const data = ['a cat is small', 'a cat is fast', 'a dog is loyal', 'a dog is friendly']

describe('Context size behavior', () => {
  it('different context sizes lead to different continuations', () => {
    const llm1 = trainLLM(data, 1)
    const llm2 = trainLLM(data, 2)

    const out1 = generate(llm1, 'cat', 3, 0.1)
    const out2 = generate(llm2, 'a cat', 3, 0.1)

    console.log('context=1:', out1)
    console.log('context=2:', out2)

    expect(out1.length).toBeGreaterThan(0)
    expect(out2.length).toBeGreaterThan(0)
    expect(out1).not.toBe(out2)
  })
})
