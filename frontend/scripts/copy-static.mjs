import { cpSync, rmSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const dist = path.join(root, 'dist')
const target = path.join(root, '..', 'backend', 'app', 'static')

rmSync(target, { recursive: true, force: true })
cpSync(dist, target, { recursive: true })
console.log(`Copied ${dist} -> ${target}`)
