import { describe, expect, it } from 'bun:test'

import { generate, trainLLM } from '../src/llm'

const trainingData = [
  'A cat is on the mat',
  'A dog is in the park',
  'Birds can fly',
  'The sun rises in the east',
  'Computer has a keyboard',
]

describe('LLM integration', () => {
  const llm = trainLLM(trainingData, 2)

  it('continues animal sentence', () => {
    const out = generate(llm, 'A cat', 3, 0.7)
    console.log(out)
    expect(out.startsWith('A cat')).toBe(true)
  })

  it('continues computer sentence', () => {
    const out = generate(llm, 'Computer', 3, 0.7)
    console.log(out)
    expect(out.startsWith('Computer')).toBe(true)
  })

  it('continues nature sentence', () => {
    const out = generate(llm, 'Birds', 3, 0.7)
    console.log(out)
    expect(out.startsWith('Birds')).toBe(true)
  })
})
