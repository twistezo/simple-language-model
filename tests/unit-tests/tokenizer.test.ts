import { describe, expect, it } from 'bun:test'

import { tokenizeText } from '../../src/tokenizer'
import { createVocabulary } from '../../src/vocabulary'

describe('Tokenizer', () => {
  it('should tokenize text into tokens when training=true', () => {
    const vocabulary = createVocabulary()
    const tokens = tokenizeText('A cat', vocabulary, true)
    expect(tokens.length).toBe(2)
    expect(tokens[0]).toBe(0)
    expect(tokens[1]).toBe(1)
  })

  it('should tokenize known words when training=false', () => {
    const vocabulary = createVocabulary()
    vocabulary.addWord('a')
    vocabulary.addWord('cat')
    const tokens = tokenizeText('A cat', vocabulary, false)
    expect(tokens).toEqual([0, 1])
  })

  it('should throw error on unknown word when training=false', () => {
    const vocabulary = createVocabulary()
    expect(() => tokenizeText('unknown', vocabulary, false)).toThrow('Unknown word: unknown')
  })

  it('should convert text to lowercase', () => {
    const vocabulary = createVocabulary()
    const tokens = tokenizeText('HELLO World', vocabulary, true)

    expect(vocabulary.decodeTokenToWord(tokens[0]!)).toBe('hello')
    expect(vocabulary.decodeTokenToWord(tokens[1]!)).toBe('world')
  })

  it('should handle multiple spaces between words', () => {
    const vocabulary = createVocabulary()
    const tokens = tokenizeText('hello    world', vocabulary, true)

    expect(tokens.length).toBe(2)
  })

  it('should handle single word', () => {
    const vocabulary = createVocabulary()
    const tokens = tokenizeText('hello', vocabulary, true)

    expect(tokens.length).toBe(1)
    expect(vocabulary.decodeTokenToWord(tokens[0]!)).toBe('hello')
  })

  it('should handle repeated words', () => {
    const vocabulary = createVocabulary()
    const tokens = tokenizeText('cat cat cat', vocabulary, true)

    expect(tokens.length).toBe(3)
    expect(tokens[0]).toBe(tokens[1])
    expect(tokens[1]).toBe(tokens[2])
  })

  it('should maintain word order', () => {
    const vocabulary = createVocabulary()
    const tokens = tokenizeText('one two three four five', vocabulary, true)

    expect(tokens).toEqual([0, 1, 2, 3, 4])
    expect(vocabulary.decodeTokenToWord(0)).toBe('one')
    expect(vocabulary.decodeTokenToWord(4)).toBe('five')
  })

  it('should handle mixed case consistently', () => {
    const vocabulary = createVocabulary()
    vocabulary.addWord('hello')

    const tokens1 = tokenizeText('HELLO', vocabulary, false)
    const tokens2 = tokenizeText('Hello', vocabulary, false)
    const tokens3 = tokenizeText('hello', vocabulary, false)

    expect(tokens1).toEqual(tokens2)
    expect(tokens2).toEqual(tokens3)
  })

  it('should handle punctuation attached to words', () => {
    const vocabulary = createVocabulary()
    const tokens = tokenizeText('hello, world!', vocabulary, true)

    expect(tokens.length).toBe(2)
    expect(vocabulary.decodeTokenToWord(tokens[0]!)).toBe('hello,')
    expect(vocabulary.decodeTokenToWord(tokens[1]!)).toBe('world!')
  })
})
