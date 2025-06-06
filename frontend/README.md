# Twitch ML Analytics Dashboard

A real-time React dashboard for monitoring and analyzing Twitch chat data using machine learning.

## Features

- **Real-time Data**: Live WebSocket connection to display chat analytics
- **Interactive Charts**: Visualize messages, sentiment, and user activity over time
- **Channel Management**: Switch between different Twitch channels
- **ML Monitoring**: Track machine learning model status and training progress
- **Responsive Design**: Modern UI with Tailwind CSS

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Running Twitch ML Analytics API backend

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The dashboard will be available at `http://localhost:3000`

## Configuration

The dashboard expects the API server to be running on `http://localhost:8000`. 

To change the API URL, modify the `API_BASE_URL` constant in `src/components/Dashboard.tsx`.

## Usage

1. **Start the API Backend**: Make sure your Twitch ML Analytics API is running
2. **Start the Frontend**: Run `npm start` to launch the dashboard
3. **Add Test Channel**: Use the "Add Test Channel" button to create sample data
4. **Select Channel**: Choose a channel from the dropdown to view its analytics
5. **Monitor Live Data**: Watch real-time charts and metrics update automatically

## Components

- **Dashboard**: Main component with real-time data visualization
- **Charts**: Interactive charts for messages, sentiment, and emote data
- **WebSocket Integration**: Real-time data streaming
- **Error Handling**: User-friendly error messages and connection status

## Development

- Built with React 18 and TypeScript
- Styled with Tailwind CSS
- Charts powered by Recharts
- HTTP requests via Axios
- Real-time data via WebSockets

## API Integration

The dashboard connects to these API endpoints:

- `GET /api/channels` - List available channels
- `GET /api/channels/{channel}/stats` - Channel statistics
- `GET /api/system/status` - System overview
- `POST /api/register` - Register test channel  
- `WebSocket /ws/{channel}` - Real-time data stream
