# Copilot Instructions

Always follow rules from /docs folder.

## Global rules:

- Use Bun as runtime, package manager, bundler, and test runner
- Use ESLint and Prettier for code formatting and linting
- Use TypeScript
- Prefer TypeScript types over interfaces
- Use arrow functions for function definitions
- Import and export TypeScript types using `import type` and `export type`
- Always add blank line before return statements
- Use clear and descriptive names for variables and functions
- Prefer functions over classes
- Do not use this keyword

## Programming principles

Use the following programming principles when applicable.

### SOLID

- S - SRP - Single Responsibility Principle

  > one responsibility

- O - OCP - Open-closed Principle

  > expandable without modification

- L - LSP - Liskov Substitution Principle

  > while extending, keep or extend the interface

- I - ISP - Interface segregation Principle

  > dedicated are better than generic

- D - DIP - Dependency inversion Principle

  > high-level things can't depend on those at low-level

### LC & HC

- Loose coupling

  > unrelated elements should have as few dependencies as possible

- High cohesion

  > related elements should be close to each other

### DI & IoC

- Depencendy Injection

  > accept instances of others rather than creating them within

- Inversion Of Control

  > do not create dependencies, accept them (DI); delegation of events instead of sequences

### Other principles

- DRY - Don't Repeat Yourself

  > divide and conquer - refactor

- KISS - Keep It Simple Stupid

  > don't overengineer - use basics

- YAGNI - You Aren't Gonna Need It

  > requirements are always changing
