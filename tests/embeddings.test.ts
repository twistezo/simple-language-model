import { describe, expect, it } from 'bun:test'

import { EmbeddingLayer } from '../src/embeddings'

describe('EmbeddingLayer', () => {
  it('should create embeddings with correct dimension', () => {
    const layer = new EmbeddingLayer(64)
    const embedding = layer.getEmbeddingForToken(0)

    expect(embedding.length).toBe(64)
  })

  it('should return same embedding for same token', () => {
    const layer = new EmbeddingLayer(32)
    const first = layer.getEmbeddingForToken(42)
    const second = layer.getEmbeddingForToken(42)

    expect(first).toEqual(second)
  })

  it('should return different embeddings for different tokens', () => {
    const layer = new EmbeddingLayer(32)
    const emb1 = layer.getEmbeddingForToken(0)
    const emb2 = layer.getEmbeddingForToken(1)

    expect(emb1).not.toEqual(emb2)
  })

  it('should return correct dimension from getEmbeddingDimension', () => {
    const layer = new EmbeddingLayer(128)
    expect(layer.getEmbeddingDimension()).toBe(128)
  })

  it('should handle dimension of 1', () => {
    const layer = new EmbeddingLayer(1)
    const embedding = layer.getEmbeddingForToken(0)

    expect(embedding.length).toBe(1)
  })

  it('should handle large dimensions', () => {
    const layer = new EmbeddingLayer(1024)
    const embedding = layer.getEmbeddingForToken(0)

    expect(embedding.length).toBe(1024)
  })

  it('should handle large token IDs', () => {
    const layer = new EmbeddingLayer(16)
    const embedding = layer.getEmbeddingForToken(999999)

    expect(embedding.length).toBe(16)
  })
})

describe('EmbeddingLayer.getEmbeddingsForTokenSequence', () => {
  it('should embed multiple tokens', () => {
    const layer = new EmbeddingLayer(16)
    const tokens = [0, 1, 2]
    const embeddings = layer.getEmbeddingsForTokenSequence(tokens)

    expect(embeddings.length).toBe(3)
    expect(embeddings[0]?.length).toBe(16)
    expect(embeddings[1]?.length).toBe(16)
    expect(embeddings[2]?.length).toBe(16)
  })

  it('should return empty array for empty input', () => {
    const layer = new EmbeddingLayer(16)
    const embeddings = layer.getEmbeddingsForTokenSequence([])

    expect(embeddings).toEqual([])
  })

  it('should return consistent embeddings for repeated tokens', () => {
    const layer = new EmbeddingLayer(16)
    const embeddings = layer.getEmbeddingsForTokenSequence([5, 5, 5])

    expect(embeddings[0]).toEqual(embeddings[1])
    expect(embeddings[1]).toEqual(embeddings[2])
  })

  it('should handle large sequence', () => {
    const layer = new EmbeddingLayer(8)
    const tokens = Array.from({ length: 100 }, (_, i) => i)
    const embeddings = layer.getEmbeddingsForTokenSequence(tokens)

    expect(embeddings.length).toBe(100)
  })

  it('should preserve token order', () => {
    const layer = new EmbeddingLayer(8)
    const embeddings = layer.getEmbeddingsForTokenSequence([0, 1, 2])

    const emb0 = layer.getEmbeddingForToken(0)
    const emb1 = layer.getEmbeddingForToken(1)
    const emb2 = layer.getEmbeddingForToken(2)

    expect(embeddings[0]).toEqual(emb0)
    expect(embeddings[1]).toEqual(emb1)
    expect(embeddings[2]).toEqual(emb2)
  })
})

describe('EmbeddingLayer.initializeTokenEmbedding', () => {
  it('should initialize token embedding', () => {
    const layer = new EmbeddingLayer(32)
    layer.initializeTokenEmbedding(99)

    const embedding = layer.getEmbeddingForToken(99)
    expect(embedding.length).toBe(32)
  })

  it('should not reinitialize existing token', () => {
    const layer = new EmbeddingLayer(32)
    layer.initializeTokenEmbedding(10)
    const first = [...layer.getEmbeddingForToken(10)]

    layer.initializeTokenEmbedding(10)
    const second = layer.getEmbeddingForToken(10)

    expect(first).toEqual(second)
  })

  it('should initialize multiple tokens independently', () => {
    const layer = new EmbeddingLayer(16)

    layer.initializeTokenEmbedding(1)
    layer.initializeTokenEmbedding(2)
    layer.initializeTokenEmbedding(3)

    const emb1 = layer.getEmbeddingForToken(1)
    const emb2 = layer.getEmbeddingForToken(2)
    const emb3 = layer.getEmbeddingForToken(3)

    expect(emb1).not.toEqual(emb2)
    expect(emb2).not.toEqual(emb3)
    expect(emb1).not.toEqual(emb3)
  })
})

describe('Embedding normalization', () => {
  it('should produce unit-length vectors', () => {
    const layer = new EmbeddingLayer(64)
    const embedding = layer.getEmbeddingForToken(0)

    const magnitude = Math.sqrt(embedding.reduce((sum: number, val: number) => sum + val * val, 0))
    expect(magnitude).toBeCloseTo(1, 5)
  })

  it('should produce values between -1 and 1', () => {
    const layer = new EmbeddingLayer(100)

    for (let token = 0; token < 10; token++) {
      const embedding = layer.getEmbeddingForToken(token)
      for (const val of embedding) {
        expect(val).toBeGreaterThanOrEqual(-1)
        expect(val).toBeLessThanOrEqual(1)
      }
    }
  })

  it('should produce normalized vectors for all tokens', () => {
    const layer = new EmbeddingLayer(32)

    for (let token = 0; token < 50; token++) {
      const embedding = layer.getEmbeddingForToken(token)
      const magnitude = Math.sqrt(embedding.reduce((sum: number, val: number) => sum + val * val, 0))
      expect(magnitude).toBeCloseTo(1, 4)
    }
  })

  it('should handle different dimensions consistently', () => {
    const dimensions = [8, 16, 32, 64, 128]

    for (const dim of dimensions) {
      const layer = new EmbeddingLayer(dim)
      const embedding = layer.getEmbeddingForToken(0)

      expect(embedding.length).toBe(dim)

      const magnitude = Math.sqrt(embedding.reduce((sum: number, val: number) => sum + val * val, 0))
      expect(magnitude).toBeCloseTo(1, 4)
    }
  })
})

describe('Embedding randomness', () => {
  it('should produce different embeddings across layers', () => {
    const layer1 = new EmbeddingLayer(32)
    const layer2 = new EmbeddingLayer(32)

    const emb1 = layer1.getEmbeddingForToken(0)
    const emb2 = layer2.getEmbeddingForToken(0)

    expect(emb1).not.toEqual(emb2)
  })

  it('should produce varied values within an embedding', () => {
    const layer = new EmbeddingLayer(64)
    const embedding = layer.getEmbeddingForToken(0)

    const uniqueValues = new Set(embedding.map((v: number) => v.toFixed(4)))
    expect(uniqueValues.size).toBeGreaterThan(1)
  })
})
