import { describe, expect, it } from 'bun:test'

import { Vocabulary } from '../src/vocabulary'

describe('Vocabulary', () => {
  it('should add and encode words correctly', () => {
    const vocab = new Vocabulary()
    const token1 = vocab.add('cat')
    const token2 = vocab.add('dog')

    expect(token1).toBe(0)
    expect(token2).toBe(1)

    expect(vocab.encode('cat')).toBe(0)
    expect(vocab.encode('dog')).toBe(1)
  })

  it('should decode tokens correctly', () => {
    const vocab = new Vocabulary()
    const t = vocab.add('bird')
    expect(vocab.decode(t)).toBe('bird')
  })

  it('should throw on decoding unknown token', () => {
    const vocab = new Vocabulary()
    expect(() => vocab.decode(999)).toThrow()
  })
})
