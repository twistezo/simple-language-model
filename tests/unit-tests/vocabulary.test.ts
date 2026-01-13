import { describe, expect, it } from 'bun:test'

import { createVocabulary } from '../../src/vocabulary'

describe('Vocabulary', () => {
  it('should add and encode words correctly', () => {
    const vocabulary = createVocabulary()
    const token1 = vocabulary.addWord('cat')
    const token2 = vocabulary.addWord('dog')

    expect(token1).toBe(0)
    expect(token2).toBe(1)

    expect(vocabulary.encodeWordToToken('cat')).toBe(0)
    expect(vocabulary.encodeWordToToken('dog')).toBe(1)
  })

  it('should decode tokens correctly', () => {
    const vocabulary = createVocabulary()
    const tokenIdentifier = vocabulary.addWord('bird')
    expect(vocabulary.decodeTokenToWord(tokenIdentifier)).toBe('bird')
  })

  it('should throw on decoding unknown token', () => {
    const vocabulary = createVocabulary()
    expect(() => vocabulary.decodeTokenToWord(999)).toThrow('Unknown token')
  })

  it('should return same token for duplicate word additions', () => {
    const vocabulary = createVocabulary()
    const first = vocabulary.addWord('hello')
    const second = vocabulary.addWord('hello')
    const third = vocabulary.addWord('hello')

    expect(first).toBe(second)
    expect(second).toBe(third)
    expect(first).toBe(0)
  })

  it('should return undefined for encoding unknown word', () => {
    const vocabulary = createVocabulary()
    vocabulary.addWord('known')

    expect(vocabulary.encodeWordToToken('unknown')).toBeUndefined()
  })

  it('should handle empty string as valid word', () => {
    const vocabulary = createVocabulary()
    const tokenIdentifier = vocabulary.addWord('')

    expect(tokenIdentifier).toBe(0)
    expect(vocabulary.encodeWordToToken('')).toBe(0)
    expect(vocabulary.decodeTokenToWord(tokenIdentifier)).toBe('')
  })

  it('should handle special characters', () => {
    const vocabulary = createVocabulary()
    const t1 = vocabulary.addWord('hello!')
    const t2 = vocabulary.addWord('@#$%')
    const t3 = vocabulary.addWord('über')

    expect(vocabulary.decodeTokenToWord(t1)).toBe('hello!')
    expect(vocabulary.decodeTokenToWord(t2)).toBe('@#$%')
    expect(vocabulary.decodeTokenToWord(t3)).toBe('über')
  })

  it('should handle large vocabulary', () => {
    const vocabulary = createVocabulary()

    for (let i = 0; i < 10000; i++) {
      vocabulary.addWord(`word${i}`)
    }

    expect(vocabulary.encodeWordToToken('word0')).toBe(0)
    expect(vocabulary.encodeWordToToken('word9999')).toBe(9999)
    expect(vocabulary.decodeTokenToWord(5000)).toBe('word5000')
  })

  it('should maintain consistency between add, encode, and decode', () => {
    const vocabulary = createVocabulary()
    const words = ['apple', 'banana', 'cherry', 'date', 'elderberry']

    const tokens = words.map(word => vocabulary.addWord(word))

    for (let i = 0; i < words.length; i++) {
      expect(vocabulary.encodeWordToToken(words[i]!)).toBe(tokens[i])
      expect(vocabulary.decodeTokenToWord(tokens[i]!)).toBe(words[i]!)
    }
  })
})
