import { describe, expect, it } from 'bun:test'

import { tokenize } from '../src/tokenizer'
import { Vocabulary } from '../src/vocabulary'

describe('Tokenizer', () => {
  it('should tokenize text into tokens when training=true', () => {
    const vocab = new Vocabulary()
    const tokens = tokenize('A cat', vocab, true)
    expect(tokens.length).toBe(2)
    expect(tokens[0]).toBe(0)
    expect(tokens[1]).toBe(1)
  })

  it('should tokenize known words when training=false', () => {
    const vocab = new Vocabulary()
    vocab.add('a')
    vocab.add('cat')
    const tokens = tokenize('A cat', vocab, false)
    expect(tokens).toEqual([0, 1])
  })

  it('should throw error on unknown word when training=false', () => {
    const vocab = new Vocabulary()
    expect(() => tokenize('unknown', vocab, false)).toThrow()
  })
})
