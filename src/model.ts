import { Sample } from './context'
import { Token } from './vocabulary'

export class SimpleLanguageModel {
  private table = new Map<string, Map<Token, number>>()

  getRow(context: Token[]): Map<Token, number> | undefined {
    return this.table.get(context.join(','))
  }

  train(samples: Sample[]): void {
    for (const { context, next } of samples) {
      const key = context.join(',')
      if (!this.table.has(key)) {
        this.table.set(key, new Map())
      }
      const row = this.table.get(key)!
      row.set(next, (row.get(next) ?? 0) + 1)
    }
  }
}

// Probabilistic sampling with temperature
export function sampleNextTokenWithTemperature(
  distribution: Map<Token, number>,
  temperature = 1,
): null | Token {
  const entries = [...distribution.entries()]
  if (entries.length === 0) return null

  const adjusted = entries.map(([token, count]) => ({
    token,
    weight: Math.pow(count, 1 / temperature),
  }))

  const total = adjusted.reduce((sum, e) => sum + e.weight, 0)
  let r = Math.random() * total

  for (const e of adjusted) {
    r -= e.weight
    if (r <= 0) return e.token
  }

  return adjusted[0].token // fallback
}
