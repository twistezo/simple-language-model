import chalk from 'chalk'

export const DEFAULT_ATTENTION_LAYERS = 4
export const DEFAULT_CONTEXT_SIZE = 3
export const DEFAULT_EMBEDDING_DIMENSION = 64
export const DEFAULT_GENERATION_LENGTH = 15
export const DEFAULT_TEMPERATURE = 0.7
export const DEFAULT_TOP_P = 0.9

export const printDefaults = (): void => {
  console.group(chalk.green('\nDefaults'))
  console.log(`Attention layers: ${DEFAULT_ATTENTION_LAYERS}`)
  console.log(`Context size: ${DEFAULT_CONTEXT_SIZE}`)
  console.log(`Embedding dimension: ${DEFAULT_EMBEDDING_DIMENSION}`)
  console.log(`Generation length: ${DEFAULT_GENERATION_LENGTH}`)
  console.log(`Temperature: ${DEFAULT_TEMPERATURE}`)
  console.log(`Top P: ${DEFAULT_TOP_P}`)
  console.groupEnd()
}
