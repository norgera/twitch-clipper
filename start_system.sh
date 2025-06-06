#!/bin/bash

echo "🚀 Starting Twitch ML Analytics System..."
echo ""

# Check if required dependencies exist
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not found. Please run: cd frontend && npm install"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "❌ Python virtual environment not found. Please create one first."
    exit 1
fi

echo "🔧 Starting API Backend..."
# Start the API backend in background
python run_integrated_system.py &
API_PID=$!

# Wait a moment for the API to start
sleep 3

echo "🌐 Starting React Frontend..."
# Start the React frontend
cd frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ System started successfully!"
echo "📊 Dashboard: http://localhost:3000"
echo "🔌 API: http://localhost:8000"
echo ""
echo "To stop the system, press Ctrl+C or run: kill $API_PID $FRONTEND_PID"

# Wait for user to stop
wait 