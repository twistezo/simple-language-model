import chalk from 'chalk'
import figlet from 'figlet'
import { type AsyncBuffer, asyncBufferFromFile, parquetReadObjects } from 'hyparquet'

import { prepareTrainingTexts, selectFile } from './dataset'
import { DEFAULT_CONTEXT_SIZE, printDefaults } from './defaults'
import { generateText, type LanguageModel, trainLanguageModel } from './llm'

async function main() {
  const logo: string = await figlet.text('SLM')
  console.log(chalk.green(logo))
  printDefaults()

  const datasetPath: string = selectFile()
  console.group(chalk.green('\nDataset'))
  const startTime: number = Date.now()

  console.log(`Loading "${datasetPath}"...`)
  const parquetFile: AsyncBuffer = await asyncBufferFromFile(datasetPath)

  console.log('Parsing records...')
  const parquetRecords: Record<string, unknown>[] = await parquetReadObjects({ file: parquetFile })
  console.log(`- Loaded ${parquetRecords.length.toLocaleString('pl-PL')} records`)

  console.log('Preparing training texts...')
  const trainingTexts: string[] = prepareTrainingTexts(parquetRecords)
  console.log(`- Collected ${trainingTexts.length.toLocaleString('pl-PL')} text entries`)
  console.groupEnd()

  console.log(chalk.green(`\nTraining ${DEFAULT_CONTEXT_SIZE}-gram language model...`))
  const languageModel: LanguageModel = trainLanguageModel(trainingTexts, DEFAULT_CONTEXT_SIZE)

  const trainingDurationSeconds = ((Date.now() - startTime) / 1000).toFixed(1)
  console.log(`- Model trained in ${trainingDurationSeconds}s.\n`)

  console.group(chalk.green('Usage'))
  console.log(
    `This is a ${DEFAULT_CONTEXT_SIZE}-gram model, so enter at least ${DEFAULT_CONTEXT_SIZE} words.`,
  )
  console.log('Type "exit" or press enter with an empty line to quit.\n')
  console.groupEnd()

  while (true) {
    const userInput: null | string = prompt(chalk.green('Prompt>'))
    if (userInput === null || userInput.toLowerCase() === 'exit') {
      break
    } else if (userInput.trim() === '') {
      continue
    }

    const inputWords: string[] = userInput.trim().split(/\s+/)
    if (inputWords.length < DEFAULT_CONTEXT_SIZE) {
      console.log(`Please enter at least ${DEFAULT_CONTEXT_SIZE} words.`)
      continue
    }

    try {
      const generatedOutput: string = generateText({ model: languageModel, prompt: userInput })
      console.log(generatedOutput)
    } catch (error) {
      if (error instanceof Error) {
        console.log(chalk.red(error.message))
      }
    }
  }
}

main().catch(console.error)
