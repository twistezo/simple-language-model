import { describe, expect, it } from 'bun:test'

import { generateText, trainLanguageModel } from '../../src/llm'

const trainingData = [
  'A cat is on the mat',
  'A dog is in the park',
  'Birds can fly',
  'The sun rises in the east',
  'Computer has a keyboard',
]

describe('LLM integration', () => {
  const languageModel = trainLanguageModel(trainingData, 2)

  it('continues animal sentence', () => {
    const output = generateText({ model: languageModel, prompt: 'A cat' })
    console.log(output)
    expect(output.startsWith('A cat')).toBe(true)
  })

  it('continues computer sentence', () => {
    const output = generateText({ model: languageModel, prompt: 'Computer' })
    console.log(output)
    expect(output.startsWith('Computer')).toBe(true)
  })

  it('continues nature sentence', () => {
    const output = generateText({ model: languageModel, prompt: 'Birds' })
    console.log(output)
    expect(output.startsWith('Birds')).toBe(true)
  })
})

describe('LLM training', () => {
  it('should create LLM with correct context size', () => {
    const languageModel = trainLanguageModel(['hello world test'], 2)
    expect(languageModel.contextWindowSize).toBe(2)
  })

  it('should create LLM with specified attention layers', () => {
    const languageModel = trainLanguageModel(['hello world test'], 2, 64, 4)
    expect(languageModel.attentionLayerCount).toBe(4)
  })

  it('should create embedding layer with specified dimension', () => {
    const languageModel = trainLanguageModel(['hello world test'], 2, 128)
    expect(languageModel.embeddingLayer.getEmbeddingDimension()).toBe(128)
  })

  it('should build vocabulary from training data', () => {
    const languageModel = trainLanguageModel(['cat dog bird'], 1)

    expect(languageModel.vocabulary.encodeWordToToken('cat')).toBeDefined()
    expect(languageModel.vocabulary.encodeWordToToken('dog')).toBeDefined()
    expect(languageModel.vocabulary.encodeWordToToken('bird')).toBeDefined()
  })

  it('should train model with n-gram statistics', () => {
    const languageModel = trainLanguageModel(['a b c', 'a b d'], 2)

    const context = [
      languageModel.vocabulary.encodeWordToToken('a')!,
      languageModel.vocabulary.encodeWordToToken('b')!,
    ]
    const distribution = languageModel.ngramModel.getNextToken(context)

    expect(distribution).toBeDefined()
    expect(distribution!.size).toBe(2)
  })
})

describe('LLM generation', () => {
  it('should return prompt when no continuation is possible', () => {
    const languageModel = trainLanguageModel(['hello world'], 2)
    const output = generateText({ model: languageModel, prompt: 'hello world' })

    expect(output).toBe('hello world')
  })

  it('should generate specified number of tokens when possible', () => {
    const languageModel = trainLanguageModel(['a b c d e f g h i j'], 1)
    const output = generateText({ model: languageModel, prompt: 'a' })

    const words = output.split(' ')
    expect(words.length).toBeGreaterThan(1)
  })

  it('should stop generation when context not found', () => {
    const languageModel = trainLanguageModel(['cat dog bird'], 2)
    const output = generateText({ model: languageModel, prompt: 'cat dog' })

    expect(output.split(' ').length).toBeLessThanOrEqual(4)
  })

  it('should throw on unknown prompt word', () => {
    const languageModel = trainLanguageModel(['hello world'], 2)

    expect(() => generateText({ model: languageModel, prompt: 'unknown word' })).toThrow()
  })

  it('should handle topP parameter', () => {
    const languageModel = trainLanguageModel(['a b c', 'a b d', 'a b e'], 2)
    const output = generateText({ model: languageModel, prompt: 'a b' })

    expect(output.startsWith('a b')).toBe(true)
  })
})

describe('LLM edge cases', () => {
  it('should handle single word training data with context 1', () => {
    const languageModel = trainLanguageModel(['a b'], 1)
    const output = generateText({ model: languageModel, prompt: 'a' })

    expect(output).toBe('a b')
  })

  it('should handle repeated words in training data', () => {
    const languageModel = trainLanguageModel(['the the the the'], 2)
    const output = generateText({ model: languageModel, prompt: 'the the' })

    expect(output.startsWith('the the the the')).toBe(true)
  })

  it('should handle multiple training sentences', () => {
    const data = ['i like cats', 'i like dogs', 'i like birds']
    const languageModel = trainLanguageModel(data, 2)

    const output = generateText({ model: languageModel, prompt: 'i like' })
    const words = output.split(' ')

    expect(words.length).toBe(3)
    expect(['cats', 'dogs', 'birds']).toContain(words[2]!)
  })
})
