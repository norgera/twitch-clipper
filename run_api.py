#!/usr/bin/env python3
"""
FastAPI Server Launcher for Twitch ML Analytics
Run this to start the web API server
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Launch the FastAPI server"""
    try:
        import uvicorn
        from api.main import app
        
        # Configuration
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))
        reload = os.getenv("API_RELOAD", "true").lower() == "true"
        log_level = os.getenv("API_LOG_LEVEL", "info")
        
        logger.info("🚀 Starting Twitch ML Analytics API Server")
        logger.info(f"📡 Server will be available at: http://{host}:{port}")
        logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
        logger.info(f"🏥 Health Check: http://{host}:{port}/health")
        logger.info("=" * 50)
        
        # Launch server
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install FastAPI dependencies with: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 