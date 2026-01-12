import { Token } from './vocabulary'

export type Sample = {
  context: Token[]
  next: Token
}

export function buildSamples(tokens: Token[], contextSize: number): Sample[] {
  const samples: Sample[] = []
  for (let i = 0; i + contextSize < tokens.length; i++) {
    samples.push({
      context: tokens.slice(i, i + contextSize),
      next: tokens[i + contextSize],
    })
  }
  return samples
}
