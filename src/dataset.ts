import { Glob } from 'bun'
import chalk from 'chalk'

const DEFAULT_DATASET = 'simple-wikipedia.parquet'
const DATASET_DIR = 'dataset'

export const selectDatasetFile = (): string => {
  const glob = new Glob('*.parquet')
  const files = Array.from(glob.scanSync(DATASET_DIR))

  if (files.length === 0) {
    console.log(chalk.red(`No .parquet files found in ${DATASET_DIR}/`))
    process.exit(1)
  }

  console.log(chalk.green('\nAvailable datasets'))
  files.forEach(file => {
    const isDefault = file === DEFAULT_DATASET
    console.log(`  ${file}${isDefault ? ' (default)' : ''}`)
  })

  const userChoice = prompt('\nEnter filename or press Enter for default.')
  if (userChoice === null || userChoice.trim() === '') {
    console.log(`- Using default: ${DEFAULT_DATASET}`)

    return `${DATASET_DIR}/${DEFAULT_DATASET}`
  }

  const selectedFile = userChoice.trim()
  if (files.includes(selectedFile)) {
    return `${DATASET_DIR}/${selectedFile}`
  }

  console.log(chalk.red(`File "${selectedFile}" not found.`))
  console.log(chalk.red(`Using default: ${DEFAULT_DATASET}`))

  return `${DATASET_DIR}/${DEFAULT_DATASET}`
}
