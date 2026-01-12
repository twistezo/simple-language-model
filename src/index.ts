import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet'
import readline from 'readline'

import { DEFAULT_CONTEXT_SIZE, DEFAULT_GENERATION_LENGTH, DEFAULT_TEMPERATURE } from './constants'
import { generate, trainLLM } from './llm'

function getCliArg(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`)
  if (index === -1) return undefined
  return process.argv[index + 1]
}

const CONTEXT_SIZE = Number(getCliArg('context-size')) || DEFAULT_CONTEXT_SIZE

const TEMPERATURE = Number(getCliArg('temperature')) || DEFAULT_TEMPERATURE

async function main() {
  console.log('Starting LLM demo...')
  console.log('Dataset: dataset/simple-wikipedia.parquet')
  console.log(`Context size: ${CONTEXT_SIZE}, Temperature: ${TEMPERATURE}`)

  const t0 = Date.now()

  const file = await asyncBufferFromFile('dataset/simple-wikipedia.parquet')
  console.log('File loaded into memory.')

  console.log('Parsing Parquet records...')
  const records = await parquetReadObjects({ file })
  console.log(`Loaded ${records.length} records.`)

  console.log('Preparing training texts...')
  const texts: string[] = []
  for (const r of records) {
    if (typeof r.text === 'string') {
      texts.push(r.text)
    }
  }
  console.log(`Collected ${texts.length} text entries.`)

  console.log(`Training ${CONTEXT_SIZE}-gram language model (this may take a while)...`)
  const llm = trainLLM(texts, CONTEXT_SIZE)

  const seconds = ((Date.now() - t0) / 1000).toFixed(1)
  console.log(`Model trained in ${seconds}s.\n`)

  console.log('How to use:')
  console.log(`- This is a ${CONTEXT_SIZE}-gram model, so enter at least ${CONTEXT_SIZE} words`)
  console.log(`- Example: "a cat" works, "cat" does not`)
  console.log('')

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  })

  function ask() {
    rl.question('Prompt> ', input => {
      const words = input.trim().split(/\s+/)
      if (words.length < CONTEXT_SIZE) {
        console.log(`Please enter at least ${CONTEXT_SIZE} words.`)
        return ask()
      }

      const out = generate(llm, input, DEFAULT_GENERATION_LENGTH, TEMPERATURE)
      console.log(out)
      ask()
    })
  }

  ask()
}

main().catch(console.error)
