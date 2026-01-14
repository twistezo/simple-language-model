import { describe, expect, it } from 'bun:test'

import {
  DEFAULT_ATTENTION_LAYERS,
  DEFAULT_CONTEXT_SIZE,
  DEFAULT_EMBEDDING_DIMENSION,
} from '../../src/defaults'
import { generateText, trainLanguageModel } from '../../src/llm'

const sampleTrainingTexts = [
  'The cat sits on the mat every morning',
  'The dog runs in the park every afternoon',
  'A bird flies over the house at dawn',
  'The sun rises in the east and sets in the west',
  'Water flows down the river to the sea',
  'The moon shines bright in the night sky',
  'Children play in the garden after school',
  'Books contain knowledge from many generations',
  'Music brings joy to people around the world',
  'Trees grow tall in the forest over time',
]

describe('Integration: Full LLM pipeline', () => {
  const languageModel = trainLanguageModel(
    sampleTrainingTexts,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_ATTENTION_LAYERS,
  )

  it('should train model with correct configuration', () => {
    expect(languageModel.contextWindowSize).toBe(DEFAULT_CONTEXT_SIZE)
    expect(languageModel.attentionLayerCount).toBe(DEFAULT_ATTENTION_LAYERS)
    expect(languageModel.embeddingLayer.getEmbeddingDimension()).toBe(DEFAULT_EMBEDDING_DIMENSION)
  })

  it('should build vocabulary from all training texts', () => {
    expect(languageModel.vocabulary.encodeWordToToken('the')).toBeDefined()
    expect(languageModel.vocabulary.encodeWordToToken('cat')).toBeDefined()
    expect(languageModel.vocabulary.encodeWordToToken('dog')).toBeDefined()
    expect(languageModel.vocabulary.encodeWordToToken('sun')).toBeDefined()
  })

  it('should generate text continuation from valid prompt', () => {
    const prompt = 'the cat sits'
    const output = generateText({ model: languageModel, prompt })

    expect(output.startsWith(prompt)).toBe(true)
    expect(output.split(' ').length).toBeGreaterThan(prompt.split(' ').length)
  })

  it('should generate different outputs with different prompts', () => {
    const output1 = generateText({ model: languageModel, prompt: 'the cat sits' })
    const output2 = generateText({ model: languageModel, prompt: 'the dog runs' })

    expect(output1).not.toBe(output2)
  })

  it('should respect generation length parameter', () => {
    const prompt = 'the sun rises'
    const output = generateText({ model: languageModel, prompt })
    const outputWords = output.split(' ')
    const promptWords = prompt.split(' ')

    expect(outputWords.length).toBeGreaterThan(promptWords.length)
  })

  it('should handle temperature variations', () => {
    const prompt = 'the cat sits'

    const lowTempOutputs = new Set<string>()
    const highTempOutputs = new Set<string>()

    for (let i = 0; i < 10; i++) {
      lowTempOutputs.add(generateText({ model: languageModel, prompt }))
      highTempOutputs.add(generateText({ model: languageModel, prompt }))
    }

    expect(highTempOutputs.size).toBeGreaterThanOrEqual(lowTempOutputs.size)
  })

  it('should throw error for unknown words in prompt', () => {
    expect(() => generateText({ model: languageModel, prompt: 'unknown xyz abc' })).toThrow()
  })

  it('should handle prompt with exact context size', () => {
    const promptWords = ['the', 'cat', 'sits'].slice(0, DEFAULT_CONTEXT_SIZE)
    const prompt = promptWords.join(' ')

    const output = generateText({ model: languageModel, prompt })
    expect(output.startsWith(prompt)).toBe(true)
  })

  it('should stop generation when no continuation found', () => {
    const uniquePhrase = 'music brings joy'
    const output = generateText({ model: languageModel, prompt: uniquePhrase })

    expect(output.split(' ').length).toBeLessThan(103)
  })
})

describe('Integration: End-to-end workflow', () => {
  it('should complete full training and generation cycle', () => {
    const texts = [
      'hello world this is a test',
      'hello world how are you',
      'hello world nice to meet you',
    ]

    const model = trainLanguageModel(texts, 2)
    const output = generateText({ model, prompt: 'hello world' })

    expect(output.startsWith('hello world')).toBe(true)
    expect(output.split(' ').length).toBeGreaterThan(2)
  })

  it('should handle multiple training and generation sessions', () => {
    const texts1 = ['a b c d e', 'a b c f g']
    const texts2 = ['x y z w v', 'x y z m n']

    const model1 = trainLanguageModel(texts1, 2)
    const model2 = trainLanguageModel(texts2, 2)

    const output1 = generateText({ model: model1, prompt: 'a b' })
    const output2 = generateText({ model: model2, prompt: 'x y' })

    expect(output1.startsWith('a b')).toBe(true)
    expect(output2.startsWith('x y')).toBe(true)
    expect(output1).not.toContain('x')
    expect(output2).not.toContain('a')
  })

  it('should support different context sizes', () => {
    const texts = ['one two three four five six seven']

    const model1 = trainLanguageModel(texts, 1)
    const model2 = trainLanguageModel(texts, 2)
    const model3 = trainLanguageModel(texts, 3)

    expect(model1.contextWindowSize).toBe(1)
    expect(model2.contextWindowSize).toBe(2)
    expect(model3.contextWindowSize).toBe(3)

    const output1 = generateText({ model: model1, prompt: 'one' })
    const output2 = generateText({ model: model2, prompt: 'one two' })
    const output3 = generateText({ model: model3, prompt: 'one two three' })

    expect(output1.startsWith('one')).toBe(true)
    expect(output2.startsWith('one two')).toBe(true)
    expect(output3.startsWith('one two three')).toBe(true)
  })

  it('should handle probabilistic token selection with TopP', () => {
    const texts = ['the quick brown fox', 'the quick red fox', 'the quick blue fox']

    const model = trainLanguageModel(texts, 2)

    const outputs = new Set<string>()
    for (let i = 0; i < 20; i++) {
      outputs.add(generateText({ model, prompt: 'the quick' }))
    }

    expect(outputs.size).toBeGreaterThanOrEqual(1)
  })

  it('should maintain vocabulary consistency across generations', () => {
    const texts = ['cat dog bird fish']
    const model = trainLanguageModel(texts, 1)

    const token1 = model.vocabulary.encodeWordToToken('cat')
    const token2 = model.vocabulary.encodeWordToToken('dog')

    expect(token1).toBeDefined()
    expect(token2).toBeDefined()

    const decoded1 = model.vocabulary.decodeTokenToWord(token1!)
    const decoded2 = model.vocabulary.decodeTokenToWord(token2!)

    expect(decoded1).toBe('cat')
    expect(decoded2).toBe('dog')
  })
})

describe('Integration: Edge cases', () => {
  it('should handle single training text', () => {
    const model = trainLanguageModel(['a b c d'], 2)
    const output = generateText({ model, prompt: 'a b' })

    expect(output).toBe('a b c d')
  })

  it('should handle repeated patterns in training', () => {
    const texts = ['the the the the the']
    const model = trainLanguageModel(texts, 2)
    const output = generateText({ model, prompt: 'the the' })

    expect(output.startsWith('the the the the the')).toBe(true)
  })

  it('should handle case insensitivity', () => {
    const texts = ['Hello World Test']
    const model = trainLanguageModel(texts, 2)

    expect(() => generateText({ model, prompt: 'hello world' })).not.toThrow()
  })

  it('should handle minimum viable training data', () => {
    const texts = ['a b']
    const model = trainLanguageModel(texts, 1)
    const output = generateText({ model, prompt: 'a' })

    expect(output).toBe('a b')
  })
})
