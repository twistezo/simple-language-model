import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet'

import {
  DEFAULT_ATTENTION_LAYERS,
  DEFAULT_CONTEXT_SIZE,
  DEFAULT_EMBEDDING_DIMENSION,
  DEFAULT_GENERATION_LENGTH,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
} from './constants'
import { generateText, trainLanguageModel } from './llm'

async function main() {
  console.log('LLM')

  console.group('\nDefaults:')
  console.log(`Attention layers: ${DEFAULT_ATTENTION_LAYERS}`)
  console.log(`Context size: ${DEFAULT_CONTEXT_SIZE}`)
  console.log(`Embedding dimension: ${DEFAULT_EMBEDDING_DIMENSION}`)
  console.log(`Generation length: ${DEFAULT_GENERATION_LENGTH}`)
  console.log(`Temperature: ${DEFAULT_TEMPERATURE}`)
  console.log(`Top P: ${DEFAULT_TOP_P}`)
  console.groupEnd()

  console.group('\nDataset:')
  const startTime = Date.now()

  console.log(`Loading 'simple-wikipedia.parquet'...`)
  const parquetFile = await asyncBufferFromFile('dataset/simple-wikipedia.parquet')

  console.log('Parsing records...')
  const parquetRecords = await parquetReadObjects({ file: parquetFile })
  console.log(`- Loaded ${parquetRecords.length} records`)

  console.log('Preparing training texts...')
  const trainingTexts: string[] = []
  for (const record of parquetRecords) {
    if (typeof record.text === 'string') {
      trainingTexts.push(record.text)
    }
  }
  console.log(`- Collected ${trainingTexts.length} text entries`)
  console.groupEnd()

  console.log(`\nTraining ${DEFAULT_CONTEXT_SIZE}-gram language model...`)
  const languageModel = trainLanguageModel(trainingTexts, DEFAULT_CONTEXT_SIZE)

  const trainingDurationSeconds = ((Date.now() - startTime) / 1000).toFixed(1)
  console.log(`\nModel trained in ${trainingDurationSeconds}s.\n`)

  console.log(
    `Hint: This is a ${DEFAULT_CONTEXT_SIZE}-gram model, so enter at least ${DEFAULT_CONTEXT_SIZE} words.`,
  )

  while (true) {
    const userInput = prompt('Prompt>')
    if (userInput === null) break

    const inputWords = userInput.trim().split(/\s+/)
    if (inputWords.length < DEFAULT_CONTEXT_SIZE) {
      console.log(`Please enter at least ${DEFAULT_CONTEXT_SIZE} words.`)
      continue
    }

    const generatedOutput = generateText(
      languageModel,
      userInput,
      DEFAULT_GENERATION_LENGTH,
      DEFAULT_TEMPERATURE,
    )
    console.log(generatedOutput)
  }
}

main().catch(console.error)
