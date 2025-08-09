#!/bin/bash
# RSE Production Startup Script
# Starts embedding service and RSE MCP server in correct order

set -e

echo "🚀 Starting RSE Production System..."

# Check if Python environment exists
if [ ! -d "embedding_service/venv" ]; then
    echo "📦 Creating Python virtual environment..."
    cd embedding_service
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..
else
    echo "✅ Python environment found"
fi

# Start embedding service
echo "🐍 Starting Python embedding service..."
cd embedding_service
source venv/bin/activate
python embedding_service.py &
EMBEDDING_PID=$!
cd ..

echo "⏳ Waiting for embedding service to initialize..."
sleep 5

# Validate embedding service is running
echo "🔍 Validating embedding service..."
curl -f http://127.0.0.1:8001/health || {
    echo "❌ Embedding service failed to start"
    kill $EMBEDDING_PID 2>/dev/null || true
    exit 1
}

echo "✅ Embedding service running on http://127.0.0.1:8001"

# Start RSE MCP server
echo "🎯 Starting RSE MCP Server..."
npm run build && npm start

# Cleanup on exit
trap 'echo "🛑 Shutting down..."; kill $EMBEDDING_PID 2>/dev/null || true' EXIT
