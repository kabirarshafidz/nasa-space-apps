#!/bin/bash

# Start FastAPI with auto-reload enabled
echo "🚀 Starting NASA Training Pipeline API with auto-reload..."
echo "📍 API will be available at: http://localhost:8000"
echo "📖 API docs at: http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"

# Check if artifacts directory exists
if [ ! -d "artifacts" ]; then
    echo "⚠️  Warning: artifacts/ directory not found"
    echo "   You may need to run the training script first to generate PCA artifacts"
    echo ""
fi

# Start with uvicorn and auto-reload
python -m uvicorn train_pipeline:app --host 0.0.0.0 --port 8000 --reload --log-level info

