#!/bin/sh

set -e

echo "Formatting..."
echo "--- Ruff ---"
ruff format panelbeater
echo "--- isort ---"
isort panelbeater

echo "Checking..."
echo "--- Flake8 ---"
flake8 panelbeater
echo "--- pylint ---"
pylint panelbeater
echo "--- mypy ---"
mypy panelbeater --disable-error-code=import-untyped
echo "--- Ruff ---"
ruff check panelbeater
echo "--- pyright ---"
pyright panelbeater
