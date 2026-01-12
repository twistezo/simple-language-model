import { applyMultiLayerAttentionWithResidualConnections } from './attention'
import { DEFAULT_ATTENTION_LAYERS, DEFAULT_EMBEDDING_DIMENSION, DEFAULT_TOP_P } from './constants'
import { buildTrainingSamples } from './context'
import { createEmbeddingLayer, type EmbeddingLayer } from './embeddings'
import { createNgramLanguageModel, type NgramLanguageModel, sampleNextToken } from './model'
import { tokenizeText } from './tokenizer'
import { createVocabulary, type Vocabulary } from './vocabulary'

export type LanguageModel = {
  attentionLayerCount: number
  contextWindowSize: number
  embeddingLayer: EmbeddingLayer
  ngramModel: NgramLanguageModel
  vocabulary: Vocabulary
}

export const generateText = (
  languageModel: LanguageModel,
  promptText: string,
  generationLength: number,
  temperature: number,
  nucleusProbabilityThreshold = DEFAULT_TOP_P,
): string => {
  let currentContext = tokenizeText(promptText, languageModel.vocabulary, false)
  const outputWords = [...promptText.split(/\s+/)]

  for (let step = 0; step < generationLength; step++) {
    const contextEmbeddings =
      languageModel.embeddingLayer.getEmbeddingsForTokenSequence(currentContext)

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const _contextualEmbeddings = applyMultiLayerAttentionWithResidualConnections(
      contextEmbeddings,
      languageModel.attentionLayerCount,
    )

    const nextTokenDistribution = languageModel.ngramModel.getNextTokenDistribution(currentContext)
    if (!nextTokenDistribution) break

    const nextToken = sampleNextToken(
      nextTokenDistribution,
      temperature,
      nucleusProbabilityThreshold,
    )
    if (nextToken === null) break

    const decodedWord = languageModel.vocabulary.decodeTokenToWord(nextToken)
    outputWords.push(decodedWord)

    currentContext = [...currentContext.slice(1), nextToken]
  }

  return outputWords.join(' ')
}

export const trainLanguageModel = (
  trainingTexts: string[],
  contextWindowSize: number,
  embeddingDimension = DEFAULT_EMBEDDING_DIMENSION,
  attentionLayerCount = DEFAULT_ATTENTION_LAYERS,
): LanguageModel => {
  const vocabulary = createVocabulary()

  const trainingSamples = trainingTexts.flatMap(text =>
    buildTrainingSamples(tokenizeText(text, vocabulary, true), contextWindowSize),
  )

  const ngramModel = createNgramLanguageModel()
  ngramModel.trainOnSamples(trainingSamples)

  const embeddingLayer = createEmbeddingLayer(embeddingDimension)
  for (const sample of trainingSamples) {
    for (const tokenIdentifier of sample.contextTokens) {
      embeddingLayer.initializeTokenEmbedding(tokenIdentifier)
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
