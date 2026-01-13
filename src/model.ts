import type { TrainingSample } from './context'
import type { TokenIdentifier } from './vocabulary'

export type NgramLanguageModel = {
  getNextTokenDistribution: (
    contextTokens: TokenIdentifier[],
  ) => TokenFrequencyDistribution | undefined
  trainOnSamples: (trainingSamples: TrainingSample[]) => void
}

export type TokenFrequencyDistribution = Map<TokenIdentifier, number>

/**
 * Creates an n-gram language model that learns token patterns from training data.
 * The model stores frequency distributions of tokens that follow specific context sequences,
 * allowing it to predict the most likely next token given a context.
 */
export const createNgramLanguageModel = (): NgramLanguageModel => {
  const contextToNextTokenFrequencies = new Map<string, TokenFrequencyDistribution>()

  const getNextTokenDistribution = (
    contextTokens: TokenIdentifier[],
  ): TokenFrequencyDistribution | undefined => {
    const contextKey = contextTokens.join(',')

    return contextToNextTokenFrequencies.get(contextKey)
  }

  const trainOnSamples = (trainingSamples: TrainingSample[]): void => {
    for (const { contextTokens, nextToken } of trainingSamples) {
      const contextKey = contextTokens.join(',')

      if (!contextToNextTokenFrequencies.has(contextKey)) {
        contextToNextTokenFrequencies.set(contextKey, new Map())
      }

      const frequencyDistribution = contextToNextTokenFrequencies.get(contextKey)!
      const currentCount = frequencyDistribution.get(nextToken) ?? 0
      frequencyDistribution.set(nextToken, currentCount + 1)
    }
  }

  return { getNextTokenDistribution, trainOnSamples }
}

/**
 * Selects the next token from a probability distribution using temperature-based sampling.
 * Higher temperature values produce more random/creative outputs, while lower values
 * make the selection more deterministic by favoring higher-frequency tokens.
 */
export const sampleNextTokenWithTemperature = (
  tokenDistribution: TokenFrequencyDistribution,
  temperature = 1,
): null | TokenIdentifier => {
  const distributionEntries = [...tokenDistribution.entries()]
  if (distributionEntries.length === 0) return null

  const temperatureAdjustedWeights = distributionEntries.map(([tokenIdentifier, frequency]) => ({
    adjustedWeight: Math.pow(frequency, 1 / temperature),
    tokenIdentifier,
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

/**
 * Samples the next token from a probability distribution using nucleus (top-p) sampling.
 *
 * This function selects a token by first applying temperature scaling to adjust the randomness,
 * then filtering to keep only the most probable tokens whose cumulative probability exceeds
 * the nucleus threshold (top-p). Finally, it randomly samples from this filtered set.
 *
 * Higher temperature increases randomness, lower temperature makes selection more deterministic.
 * The nucleus threshold controls how many top tokens are considered for sampling.
 */
export const sampleNextTokenWithNucleusSampling = (
  tokenDistribution: TokenFrequencyDistribution,
  nucleusProbabilityThreshold: number,
  temperature = 1,
): null | TokenIdentifier => {
  const distributionEntries = [...tokenDistribution.entries()]
  if (distributionEntries.length === 0) return null

  const temperatureAdjustedWeights = distributionEntries
    .map(([tokenIdentifier, frequency]) => ({
      adjustedWeight: Math.pow(frequency, 1 / temperature),
      tokenIdentifier,
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

/**
 * Samples the next token from a probability distribution using either nucleus sampling or temperature-based sampling.
 * If a valid nucleus probability threshold (between 0 and 1) is provided, it uses nucleus sampling.
 * Otherwise, it falls back to temperature-based sampling.
 */
export const sampleNextToken = (
  tokenDistribution: TokenFrequencyDistribution,
  temperature = 1,
  nucleusProbabilityThreshold?: number,
): null | TokenIdentifier => {
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
