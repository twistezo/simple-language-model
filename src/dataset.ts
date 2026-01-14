import { Glob } from 'bun'
import chalk from 'chalk'

const DEFAULT_DATASET = 'simple-wikipedia.parquet'
const DATASET_DIR = 'dataset'

export const selectFile = (): string => {
  const glob: Glob = new Glob('*.parquet')
  const files: string[] = Array.from(glob.scanSync(DATASET_DIR))

  if (files.length === 0) {
    console.log(chalk.red(`No .parquet files found in ${DATASET_DIR}/`))
    process.exit(1)
  }

  console.log(chalk.green('\nAvailable datasets'))
  files.forEach((file: string): void => {
    console.log(`  ${file}${file === DEFAULT_DATASET ? ' (default)' : ''}`)
  })

  const userChoice: null | string = prompt(
    chalk.green('\nEnter filename or press Enter for default>'),
  )

  if (userChoice === null || userChoice.trim() === '') {
    console.log(`- Using default: ${DEFAULT_DATASET}`)
    return `${DATASET_DIR}/${DEFAULT_DATASET}`
  }

  const selectedFile: string = userChoice.trim()
  if (files.includes(selectedFile)) {
    return `${DATASET_DIR}/${selectedFile}`
  }

  console.log(chalk.red(`File "${selectedFile}" not found.`))
  console.log(chalk.red(`Using default: ${DEFAULT_DATASET}`))

  return `${DATASET_DIR}/${DEFAULT_DATASET}`
}

export const prepareTrainingTexts = (records: Record<string, unknown>[]): string[] => {
  const trainingTexts: string[] = []

  for (const record of records) {
    const texts = extractTextFromRecord(record)
    for (const text of texts) {
      if (typeof text === 'string' && text.length > 0) {
        trainingTexts.push(text)
      }
    }
  }

  return trainingTexts
}

const extractTextFromRecord = (record: unknown): string[] => {
  if (typeof record === 'string') {
    return [record]
  } else if (Array.isArray(record)) {
    return record.flatMap(extractTextFromRecord)
  } else if (record && typeof record === 'object') {
    return Object.values(record).flatMap(extractTextFromRecord)
  } else {
    return []
  }
}
