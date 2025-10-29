#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script para reemplazar emojis problemáticos en detailed_analysis.py

with open('app/routers/detailed_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Reemplazar emojis problemáticos con símbolos seguros
replacements = {
    # Emojis de árboles
    '🌳': '[TREE]',
    '\ud83c\udf33': '[TREE]',
    # Emojis de checkmarks y advertencias
    '✅': '[OK]',
    '⚠️': '[!]',
    '❌': '[X]',
}

for emoji, replacement in replacements.items():
    content = content.replace(emoji, replacement)

with open('app/routers/detailed_analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Emojis reemplazados exitosamente")
