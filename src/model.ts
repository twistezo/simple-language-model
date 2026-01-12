import type { TrainingSample } from './context'
import type { TokenIdentifier } from './vocabulary'

export type TokenFrequencyDistribution = Map<TokenIdentifier, number>

export class NgramLanguageModel {
  private contextToNextTokenFrequencies = new Map<string, TokenFrequencyDistribution>()

  getNextTokenDistribution(
    contextTokens: TokenIdentifier[],
  ): TokenFrequencyDistribution | undefined {
    const contextKey = contextTokens.join(',')

    return this.contextToNextTokenFrequencies.get(contextKey)
  }

  trainOnSamples(trainingSamples: TrainingSample[]): void {
    for (const { contextTokens, nextToken } of trainingSamples) {
      const contextKey = contextTokens.join(',')

      if (!this.contextToNextTokenFrequencies.has(contextKey)) {
        this.contextToNextTokenFrequencies.set(contextKey, new Map())
      }

      const frequencyDistribution = this.contextToNextTokenFrequencies.get(contextKey)!
      const currentCount = frequencyDistribution.get(nextToken) ?? 0
      frequencyDistribution.set(nextToken, currentCount + 1)
    }
  }
}

export const sampleNextTokenWithTemperature = (
  tokenDistribution: TokenFrequencyDistribution,
  temperature = 1,
): TokenIdentifier | null => {
  const distributionEntries = [...tokenDistribution.entries()]
  if (distributionEntries.length === 0) return null

  const temperatureAdjustedWeights = distributionEntries.map(([tokenIdentifier, frequency]) => ({
    tokenIdentifier,
    adjustedWeight: Math.pow(frequency, 1 / temperature),
  }))

  const totalWeight = temperatureAdjustedWeights.reduce(
    (sum, entry) => sum + entry.adjustedWeight,
    0,
  )
  let randomThreshold = Math.random() * totalWeight

  for (const entry of temperatureAdjustedWeights) {
    randomThreshold -= entry.adjustedWeight
    if (randomThreshold <= 0) return entry.tokenIdentifier
  }

  return temperatureAdjustedWeights[0]?.tokenIdentifier ?? null
}

export const sampleNextTokenWithNucleusSampling = (
  tokenDistribution: TokenFrequencyDistribution,
  nucleusProbabilityThreshold: number,
  temperature = 1,
): TokenIdentifier | null => {
  const distributionEntries = [...tokenDistribution.entries()]
  if (distributionEntries.length === 0) return null

  const temperatureAdjustedWeights = distributionEntries
    .map(([tokenIdentifier, frequency]) => ({
      tokenIdentifier,
      adjustedWeight: Math.pow(frequency, 1 / temperature),
    }))
    .sort((entryA, entryB) => entryB.adjustedWeight - entryA.adjustedWeight)

  const totalWeight = temperatureAdjustedWeights.reduce(
    (sum, entry) => sum + entry.adjustedWeight,
    0,
  )

  let cumulativeProbability = 0
  const nucleusTokens: typeof temperatureAdjustedWeights = []

  for (const entry of temperatureAdjustedWeights) {
    cumulativeProbability += entry.adjustedWeight / totalWeight
    nucleusTokens.push(entry)
    if (cumulativeProbability >= nucleusProbabilityThreshold) break
  }

  const nucleusTotalWeight = nucleusTokens.reduce((sum, entry) => sum + entry.adjustedWeight, 0)
  let randomThreshold = Math.random() * nucleusTotalWeight

  for (const entry of nucleusTokens) {
    randomThreshold -= entry.adjustedWeight
    if (randomThreshold <= 0) return entry.tokenIdentifier
  }

  return nucleusTokens[0]?.tokenIdentifier ?? null
}

export const sampleNextToken = (
  tokenDistribution: TokenFrequencyDistribution,
  temperature = 1,
  nucleusProbabilityThreshold?: number,
): TokenIdentifier | null => {
  if (
    nucleusProbabilityThreshold !== undefined &&
    nucleusProbabilityThreshold > 0 &&
    nucleusProbabilityThreshold < 1
  ) {
    return sampleNextTokenWithNucleusSampling(
      tokenDistribution,
      nucleusProbabilityThreshold,
      temperature,
    )
  }

  return sampleNextTokenWithTemperature(tokenDistribution, temperature)
}
