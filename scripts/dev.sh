#!/bin/bash

echo "Activating environment..."
source .venv/bin/activate
echo "Environment activated."

echo "Starting PostgreSQL..."
brew services start postgresql@16

echo "You are ready to work!"
