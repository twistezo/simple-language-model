import { applyMultiLayerAttentionWithResidualConnections } from './attention'
import { buildTrainingSamples, type TrainingSample } from './context'
import {
  DEFAULT_ATTENTION_LAYERS,
  DEFAULT_EMBEDDING_DIMENSION,
  DEFAULT_GENERATION_LENGTH,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
} from './defaults'
import { createEmbeddingLayer, type EmbeddingLayer } from './embeddings'
import {
  createNgramLanguageModel,
  type NgramLanguageModel,
  sampleNextToken,
  type TokenFrequencyDistribution,
} from './model'
import { tokenizeText } from './tokenizer'
import { createVocabulary, type TokenId, type Vocabulary } from './vocabulary'

export type LanguageModel = {
  attentionLayerCount: number
  contextWindowSize: number
  embeddingLayer: EmbeddingLayer
  ngramModel: NgramLanguageModel
  vocabulary: Vocabulary
}

export const trainLanguageModel = (
  trainingTexts: string[],
  contextWindowSize: number,
  embeddingDimension = DEFAULT_EMBEDDING_DIMENSION,
  attentionLayerCount = DEFAULT_ATTENTION_LAYERS,
): LanguageModel => {
  const vocabulary: Vocabulary = createVocabulary()

  const trainingSamples: TrainingSample[] = trainingTexts.flatMap(
    (text: string): TrainingSample[] =>
      buildTrainingSamples(tokenizeText(text, vocabulary, true), contextWindowSize),
  )

  const ngramModel: NgramLanguageModel = createNgramLanguageModel()
  ngramModel.train(trainingSamples)

  const embeddingLayer: EmbeddingLayer = createEmbeddingLayer(embeddingDimension)
  for (const sample of trainingSamples) {
    for (const tokenId of sample.contextTokens) {
      embeddingLayer.initializeTokenEmbedding(tokenId)
    }
    embeddingLayer.initializeTokenEmbedding(sample.nextToken)
  }

  return {
    attentionLayerCount,
    contextWindowSize,
    embeddingLayer,
    ngramModel,
    vocabulary,
  }
}

export const generateText = ({
  model,
  prompt,
}: {
  model: LanguageModel
  prompt: string
}): string => {
  let currentContext: TokenId[] = tokenizeText(prompt, model.vocabulary, false)
  const outputWords: string[] = [...prompt.split(/\s+/)]

  for (let step = 0; step < DEFAULT_GENERATION_LENGTH; step++) {
    const contextEmbeddings = model.embeddingLayer.getEmbeddingsForTokenSequence(currentContext)

    // Note: Attention is used here for educational purposes-it illustrates
    // the underlying principles of the mechanism.
    // Without a neural network it cannot be used for prediction.
    //
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const _contextualEmbeddings = applyMultiLayerAttentionWithResidualConnections(
      contextEmbeddings,
      model.attentionLayerCount,
    )

    const nextTokenDistribution: TokenFrequencyDistribution | undefined =
      model.ngramModel.getNextToken(currentContext)
    if (!nextTokenDistribution) break

    const nextToken: null | TokenId = sampleNextToken(
      nextTokenDistribution,
      DEFAULT_TEMPERATURE,
      DEFAULT_TOP_P,
    )
    if (nextToken === null) break

    const decodedWord: string = model.vocabulary.decodeTokenToWord(nextToken)
    outputWords.push(decodedWord)

    currentContext = [...currentContext.slice(1), nextToken]
  }

  return outputWords.join(' ')
}
