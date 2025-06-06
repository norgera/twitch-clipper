from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
import time
from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time data streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, channel: str):
        """Connect a new WebSocket client to a channel"""
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)
        logger.info(f"Client connected to channel {channel}. Total connections: {len(self.active_connections[channel])}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        """Disconnect a WebSocket client from a channel"""
        if channel in self.active_connections:
            try:
                self.active_connections[channel].remove(websocket)
                if not self.active_connections[channel]:
                    del self.active_connections[channel]
                logger.info(f"Client disconnected from channel {channel}")
            except ValueError:
                # WebSocket already removed
                pass
    
    async def broadcast_to_channel(self, channel: str, data: dict):
        """Broadcast data to all connected clients for a channel"""
        if channel in self.active_connections:
            disconnected = []
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logger.warning(f"Failed to send data to client: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for connection in disconnected:
                self.disconnect(connection, channel)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(conns) for conns in self.active_connections.values())

# Global instances
manager = ConnectionManager()
bot_analyzers = {}  # Will be populated by the Twitch bot

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Twitch ML Analytics API...")
    logger.info("API server ready to accept connections")
    yield
    # Shutdown
    logger.info("Shutting down Twitch ML Analytics API...")

app = FastAPI(
    title="Twitch ML Analytics API",
    description="Real-time ML analytics and monitoring for Twitch chat analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Twitch ML Analytics API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/api/channels")
async def get_channels():
    """Get list of currently monitored channels"""
    channels = list(bot_analyzers.keys())
    return {
        "channels": channels,
        "count": len(channels),
        "timestamp": time.time()
    }

@app.get("/api/channels/{channel}/stats")
async def get_channel_stats(channel: str):
    """Get current statistics for a specific channel"""
    if channel not in bot_analyzers:
        raise HTTPException(
            status_code=404, 
            detail=f"Channel '{channel}' is not being monitored"
        )
    
    try:
        analyzer = bot_analyzers[channel]
        stats = analyzer.get_window_stats()
        
        # Enhanced stats with ML metadata
        enhanced_stats = {
            "channel": channel,
            "timestamp": time.time(),
            "stats": stats,
            "ml_status": {
                "model_loaded": len(analyzer.feature_history) >= 100,  # Use constant directly
                "training_samples": len(analyzer.feature_history),  # Show ACTUAL samples, not capped
                "baseline_samples": 60,  # Target baseline samples
                "current_baseline_count": len(analyzer.baseline_velocities),  # Show ACTUAL baseline count
                "model_status": "active" if len(analyzer.feature_history) >= 100 and len(analyzer.baseline_velocities) >= 60 else "training",
                "last_prediction": time.time(),
                "samples_used_for_training": 200 if len(analyzer.feature_history) > 200 else len(analyzer.feature_history)  # NEW: Show what's actually used
            },
            "data_status": {
                "total_messages": getattr(analyzer, 'total_messages_processed', len(analyzer.messages)),  # Fallback for compatibility
                "messages_in_buffer": len(analyzer.messages),  # Add buffer size for debugging
                "emote_window_size": len(analyzer.emote_window),
                "memory_stats": analyzer.emote_spam_detector.get_memory_stats()
            }
        }
        
        return enhanced_stats
        
    except Exception as e:
        logger.error(f"Error getting stats for channel {channel}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats for channel '{channel}'"
        )

@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """WebSocket endpoint for real-time channel data streaming"""
    await manager.connect(websocket, channel)
    
    try:
        logger.info(f"Starting real-time stream for channel: {channel}")
        
        while True:
            if channel in bot_analyzers:
                try:
                    analyzer = bot_analyzers[channel]
                    stats = analyzer.get_window_stats()
                    
                    # Create enhanced real-time data payload
                    data = {
                        "channel": channel,
                        "timestamp": time.time(),
                        "stats": stats,
                        "recent_messages": [
                            {
                                "text": msg["text"][:150],  # Truncate very long messages
                                "user": msg["user_id"],
                                "timestamp": msg["timestamp"].isoformat(),
                                "sentiment": msg.get("sentiment", {}).get("compound", 0)
                            }
                            for msg in list(analyzer.messages)[-5:]  # Last 5 messages
                        ],
                        "ml_metrics": {
                            "feature_count": len(analyzer.feature_history),
                            "model_status": "active" if len(analyzer.feature_history) >= 100 else "training",
                            "baseline_samples": len(analyzer.baseline_velocities),
                            "memory_usage": analyzer.emote_spam_detector.get_memory_stats(),
                            "last_update": time.time()
                        },
                        "connection_info": {
                            "connected_clients": len(manager.active_connections.get(channel, [])),
                            "uptime": time.time()
                        }
                    }
                    
                    await websocket.send_json(data)
                    
                except Exception as e:
                    logger.error(f"Error processing data for channel {channel}: {e}")
                    # Send error notification to client
                    await websocket.send_json({
                        "error": f"Data processing error: {str(e)}",
                        "channel": channel,
                        "timestamp": time.time()
                    })
            else:
                # Channel not available
                await websocket.send_json({
                    "error": f"Channel '{channel}' is not being monitored",
                    "available_channels": list(bot_analyzers.keys()),
                    "timestamp": time.time()
                })
            
            await asyncio.sleep(0.5)  # Update every 500ms
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from channel {channel}")
        manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error for channel {channel}: {e}")
        manager.disconnect(websocket, channel)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time(),  # Will be actual uptime in production
        "services": {
            "api": "running",
            "websockets": "running",
            "ml_analyzers": "running"
        },
        "metrics": {
            "active_channels": len(bot_analyzers),
            "active_connections": manager.get_connection_count(),
            "channels_list": list(bot_analyzers.keys())
        },
        "version": "1.0.0"
    }

@app.get("/api/system/status")
async def system_status():
    """Detailed system status for debugging and monitoring"""
    total_features = sum(len(analyzer.feature_history) for analyzer in bot_analyzers.values())
    total_messages = sum(getattr(analyzer, 'total_messages_processed', len(analyzer.messages)) for analyzer in bot_analyzers.values())  # Use actual total with fallback
    
    return {
        "timestamp": time.time(),
        "channels": {
            channel: {
                "status": "active",
                "feature_count": len(analyzer.feature_history),
                "message_count": getattr(analyzer, 'total_messages_processed', len(analyzer.messages)),  # Use actual total with fallback
                "messages_in_buffer": len(analyzer.messages),  # Add buffer info
                "model_status": "active" if len(analyzer.feature_history) >= 100 else "training",
                "memory_stats": analyzer.emote_spam_detector.get_memory_stats()
            }
            for channel, analyzer in bot_analyzers.items()
        },
        "totals": {
            "channels": len(bot_analyzers),
            "total_features": total_features,
            "total_messages": total_messages,
            "connections": manager.get_connection_count()
        }
    }

# Development endpoint for testing
@app.get("/api/test")
async def test_endpoint():
    """Test endpoint for development and debugging"""
    return {
        "message": "API is working correctly",
        "timestamp": time.time(),
        "analyzers_connected": len(bot_analyzers) > 0,
        "available_endpoints": [
            "/",
            "/health",
            "/api/channels",
            "/api/channels/{channel}/stats",
            "/ws/{channel}",
            "/api/system/status"
        ]
    }

@app.post("/api/register")
async def register_external_analyzer(channel: str):
    """Register an external analyzer (for testing)"""
    # This is primarily for testing - in production the bot registers directly
    from app.chat_analyzer_ml import ChatAnalyzerML
    
    if channel not in bot_analyzers:
        # Create a test analyzer for demonstration
        test_analyzer = ChatAnalyzerML(window_size=30, channel=channel)
        
        # Add some sample data to make it realistic
        for i in range(15):
            test_analyzer.add_message(
                message=f"Test message {i} PogChamp Kappa EZ Clap",
                user_id=f"test_user_{i}",
                timestamp=datetime.now()
            )
        
        # Set a viewer count for testing
        test_analyzer.viewer_count = 250
        
        bot_analyzers[channel] = test_analyzer
        
        logger.info(f"Registered external analyzer for channel: {channel}")
        return {
            "success": True,
            "message": f"Registered analyzer for channel '{channel}' with sample data",
            "total_channels": len(bot_analyzers),
            "sample_stats": test_analyzer.get_window_stats()
        }
    else:
        return {
            "success": False,
            "message": f"Channel '{channel}' already registered",
            "total_channels": len(bot_analyzers)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 