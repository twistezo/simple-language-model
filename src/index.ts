import chalk from 'chalk'
import figlet from 'figlet'
import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet'

import {
  DEFAULT_ATTENTION_LAYERS,
  DEFAULT_CONTEXT_SIZE,
  DEFAULT_EMBEDDING_DIMENSION,
  DEFAULT_GENERATION_LENGTH,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
} from './constants'
import { selectDatasetFile } from './dataset'
import { generateText, trainLanguageModel } from './llm'

async function main() {
  const logo = await figlet.text('AI - LLM')
  console.log(chalk.green(logo))

  console.group(chalk.green('\nDefaults'))
  console.log(`Attention layers: ${DEFAULT_ATTENTION_LAYERS}`)
  console.log(`Context size: ${DEFAULT_CONTEXT_SIZE}`)
  console.log(`Embedding dimension: ${DEFAULT_EMBEDDING_DIMENSION}`)
  console.log(`Generation length: ${DEFAULT_GENERATION_LENGTH}`)
  console.log(`Temperature: ${DEFAULT_TEMPERATURE}`)
  console.log(`Top P: ${DEFAULT_TOP_P}`)
  console.groupEnd()

  const datasetPath = selectDatasetFile()
  console.group(chalk.green('\nDataset'))
  const startTime = Date.now()

  console.log(`Loading "${datasetPath}"...`)
  const parquetFile = await asyncBufferFromFile(datasetPath)

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

  console.log(chalk.green(`\nTraining ${DEFAULT_CONTEXT_SIZE}-gram language model...`))
  const languageModel = trainLanguageModel(trainingTexts, DEFAULT_CONTEXT_SIZE)

  const trainingDurationSeconds = ((Date.now() - startTime) / 1000).toFixed(1)
  console.log(`- Model trained in ${trainingDurationSeconds}s.\n`)

  console.group(chalk.green('Usage'))
  console.log(
    `This is a ${DEFAULT_CONTEXT_SIZE}-gram model, so enter at least ${DEFAULT_CONTEXT_SIZE} words.`,
  )
  console.log('Type "exit" or press enter with an empty line to quit.\n')
  console.groupEnd()

  while (true) {
    const userInput = prompt(chalk.green('Prompt>'))
    if (userInput === null || userInput.toLowerCase() === 'exit') break
    if (userInput.trim() === '') continue

    const inputWords = userInput.trim().split(/\s+/)
    if (inputWords.length < DEFAULT_CONTEXT_SIZE) {
      console.log(`Please enter at least ${DEFAULT_CONTEXT_SIZE} words.`)
      continue
    }

    try {
      const generatedOutput = generateText(
        languageModel,
        userInput,
        DEFAULT_GENERATION_LENGTH,
        DEFAULT_TEMPERATURE,
      )
      console.log(generatedOutput)
    } catch (error) {
      if (error instanceof Error) {
        console.log(chalk.red(error.message))
      }
    }
  }
}

main().catch(console.error)
