import type { TokenIdentifier } from './vocabulary'

export type TrainingSample = {
  contextTokens: TokenIdentifier[]
  nextToken: TokenIdentifier
}

export const buildTrainingSamples = (
  tokenSequence: TokenIdentifier[],
  contextWindowSize: number,
): TrainingSample[] => {
  const trainingSamples: TrainingSample[] = []

  for (let position = 0; position + contextWindowSize < tokenSequence.length; position++) {
    const nextToken = tokenSequence[position + contextWindowSize]
    if (nextToken !== undefined) {
      trainingSamples.push({
        contextTokens: tokenSequence.slice(position, position + contextWindowSize),
        nextToken,
      })
    }
  }

  return trainingSamples
}
