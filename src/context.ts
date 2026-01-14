import type { TokenId } from './vocabulary'

export type TrainingSample = {
  contextTokens: TokenId[]
  nextToken: TokenId
}

/**
 * Creates training samples from a sequence of tokens using a sliding window approach.
 */
export const buildTrainingSamples = (
  tokenSequence: TokenId[],
  contextWindowSize: number,
): TrainingSample[] => {
  const trainingSamples: TrainingSample[] = []

  for (let position = 0; position + contextWindowSize < tokenSequence.length; position++) {
    const nextToken: TokenId | undefined = tokenSequence[position + contextWindowSize]

    if (nextToken !== undefined) {
      trainingSamples.push({
        contextTokens: tokenSequence.slice(position, position + contextWindowSize),
        nextToken,
      })
    }
  }

  return trainingSamples
}
