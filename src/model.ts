import type { TrainingSample } from './context'
import type { TokenId } from './vocabulary'

export type NgramLanguageModel = {
  getNextToken: (contextTokens: TokenId[]) => TokenFrequencyDistribution | undefined
  train: (trainingSamples: TrainingSample[]) => void
}

export type TokenFrequencyDistribution = Map<TokenId, number>

type TemperatureAdjustedWeightEntry = {
  adjustedWeight: number
  tokenId: TokenId
}

/**
 * Creates an n-gram language model that learns token patterns from training data.
 */
export const createNgramLanguageModel = (): NgramLanguageModel => {
  const contextToNextTokenFrequencies: Map<string, TokenFrequencyDistribution> = new Map<
    string,
    TokenFrequencyDistribution
  >()

  const getNextToken = (contextTokens: TokenId[]): TokenFrequencyDistribution | undefined => {
    return contextToNextTokenFrequencies.get(contextTokens.join(','))
  }

  const train = (samples: TrainingSample[]): void => {
    for (const { contextTokens, nextToken } of samples) {
      const contextKey: string = contextTokens.join(',')

      if (!contextToNextTokenFrequencies.has(contextKey)) {
        contextToNextTokenFrequencies.set(contextKey, new Map())
      }

      const frequencyDistribution: TokenFrequencyDistribution =
        contextToNextTokenFrequencies.get(contextKey)!

      const currentCount: number = frequencyDistribution.get(nextToken) ?? 0
      frequencyDistribution.set(nextToken, currentCount + 1)
    }
  }

  return { getNextToken, train }
}

/**
 * Samples the next token from a probability distribution using either nucleus sampling or temperature-based sampling.
 */
export const sampleNextToken = (
  tokenDistribution: TokenFrequencyDistribution,
  temperature: number,
  nucleusProbabilityThreshold: number,
): null | TokenId => {
  if (nucleusProbabilityThreshold > 0 && nucleusProbabilityThreshold < 1) {
    return sampleNextTokenWithNucleusSampling(
      tokenDistribution,
      nucleusProbabilityThreshold,
      temperature,
    )
  } else {
    return sampleNextTokenWithTemperature(tokenDistribution, temperature)
  }
}

/**
 * Samples the next token from a probability distribution using nucleus (top-p) sampling.
 *
 * Source: Wikipedia, various blog posts and articles
 */
export const sampleNextTokenWithNucleusSampling = (
  tokenDistribution: TokenFrequencyDistribution,
  nucleusProbabilityThreshold: number,
  temperature: number,
): null | TokenId => {
  const distributionEntries: [TokenId, number][] = [...tokenDistribution.entries()]
  if (distributionEntries.length === 0) return null

  const temperatureAdjustedWeights: TemperatureAdjustedWeightEntry[] = distributionEntries
    .map(([tokenId, frequency]: [TokenId, number]) => ({
      adjustedWeight: Math.pow(frequency, 1 / temperature),
      tokenId,
    }))
    .sort(
      (entryA: TemperatureAdjustedWeightEntry, entryB: TemperatureAdjustedWeightEntry): number =>
        entryB.adjustedWeight - entryA.adjustedWeight,
    )

  const totalWeight: number = temperatureAdjustedWeights.reduce(
    (sum: number, entry: TemperatureAdjustedWeightEntry): number => sum + entry.adjustedWeight,
    0,
  )

  let cumulativeProbability: number = 0
  const nucleusTokens: TemperatureAdjustedWeightEntry[] = []

  for (const entry of temperatureAdjustedWeights) {
    cumulativeProbability += entry.adjustedWeight / totalWeight
    nucleusTokens.push(entry)

    if (cumulativeProbability >= nucleusProbabilityThreshold) break
  }

  const nucleusTotalWeight: number = nucleusTokens.reduce(
    (sum: number, entry: TemperatureAdjustedWeightEntry): number => sum + entry.adjustedWeight,
    0,
  )
  let randomThreshold: number = Math.random() * nucleusTotalWeight

  for (const entry of nucleusTokens) {
    randomThreshold -= entry.adjustedWeight

    if (randomThreshold <= 0) return entry.tokenId
  }

  return nucleusTokens[0]?.tokenId ?? null
}

/**
 * Selects the next token from a probability distribution using temperature-based sampling.
 *
 * Source: Wikipedia, various blog posts and articles
 */
export const sampleNextTokenWithTemperature = (
  tokenDistribution: TokenFrequencyDistribution,
  temperature: number,
): null | TokenId => {
  const distributionEntries: [TokenId, number][] = [...tokenDistribution.entries()]
  if (distributionEntries.length === 0) return null

  const temperatureAdjustedWeights: TemperatureAdjustedWeightEntry[] = distributionEntries.map(
    ([tokenId, frequency]: [TokenId, number]): TemperatureAdjustedWeightEntry => ({
      adjustedWeight: Math.pow(frequency, 1 / temperature),
      tokenId,
    }),
  )

  const totalWeight: number = temperatureAdjustedWeights.reduce(
    (sum: number, entry: TemperatureAdjustedWeightEntry): number => sum + entry.adjustedWeight,
    0,
  )

  let randomThreshold: number = Math.random() * totalWeight
  for (const entry of temperatureAdjustedWeights) {
    randomThreshold -= entry.adjustedWeight

    if (randomThreshold <= 0) return entry.tokenId
  }

  return temperatureAdjustedWeights[0]?.tokenId ?? null
}
