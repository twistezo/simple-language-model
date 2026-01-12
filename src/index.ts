import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet'

import { DEFAULT_CONTEXT_SIZE, DEFAULT_GENERATION_LENGTH, DEFAULT_TEMPERATURE } from './constants'
import { generateText, trainLanguageModel } from './llm'

async function main() {
  console.log('Starting LLM demo...')
  console.log('Dataset: dataset/simple-wikipedia.parquet')
  console.log(`Context size: ${DEFAULT_CONTEXT_SIZE}, Temperature: ${DEFAULT_TEMPERATURE}`)

  const startTime = Date.now()

  const parquetFile = await asyncBufferFromFile('dataset/simple-wikipedia.parquet')
  console.log('File loaded into memory.')

  console.log('Parsing Parquet records...')
  const parquetRecords = await parquetReadObjects({ file: parquetFile })
  console.log(`Loaded ${parquetRecords.length} records.`)

  console.log('Preparing training texts...')
  const trainingTexts: string[] = []
  for (const record of parquetRecords) {
    if (typeof record.text === 'string') {
      trainingTexts.push(record.text)
    }
  }
  console.log(`Collected ${trainingTexts.length} text entries.`)

  console.log(`Training ${DEFAULT_CONTEXT_SIZE}-gram language model (this may take a while)...`)
  const languageModel = trainLanguageModel(trainingTexts, DEFAULT_CONTEXT_SIZE)

  const trainingDurationSeconds = ((Date.now() - startTime) / 1000).toFixed(1)
  console.log(`Model trained in ${trainingDurationSeconds}s.\n`)

  console.log('How to use:')
  console.log(
    `- This is a ${DEFAULT_CONTEXT_SIZE}-gram model, so enter at least ${DEFAULT_CONTEXT_SIZE} words`,
  )
  console.log(`- Example: "a cat" works, "cat" does not`)
  console.log('')

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
