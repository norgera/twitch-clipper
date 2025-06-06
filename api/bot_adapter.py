"""
Bot Adapter - Connects existing Twitch bot to FastAPI service
"""
import logging
from typing import Dict
from app.chat_analyzer_ml import ChatAnalyzerML

logger = logging.getLogger(__name__)

class BotAPIAdapter:
    """Adapter class to connect Twitch bot analyzers to the FastAPI service"""
    
    def __init__(self):
        self.analyzers: Dict[str, ChatAnalyzerML] = {}
        
    def register_analyzer(self, channel: str, analyzer: ChatAnalyzerML):
        """Register a channel's analyzer with the API service"""
        self.analyzers[channel] = analyzer
        logger.info(f"Registered analyzer for channel: {channel}")
        
    def unregister_analyzer(self, channel: str):
        """Unregister a channel's analyzer from the API service"""
        if channel in self.analyzers:
            del self.analyzers[channel]
            logger.info(f"Unregistered analyzer for channel: {channel}")
    
    def get_analyzer(self, channel: str) -> ChatAnalyzerML:
        """Get analyzer for a specific channel"""
        return self.analyzers.get(channel)
    
    def get_all_analyzers(self) -> Dict[str, ChatAnalyzerML]:
        """Get all registered analyzers"""
        return self.analyzers.copy()
    
    def get_channel_list(self) -> list:
        """Get list of all registered channels"""
        return list(self.analyzers.keys())

# Global adapter instance
api_adapter = BotAPIAdapter()

def register_analyzers_with_api(analyzers: Dict[str, ChatAnalyzerML]):
    """
    Helper function to register existing bot analyzers with the API service
    Call this from your main bot file after creating analyzers
    """
    try:
        # Import here to avoid circular imports
        import api.main as api_main
        
        # Register all analyzers with both the adapter and the FastAPI app
        for channel, analyzer in analyzers.items():
            api_adapter.register_analyzer(channel, analyzer)
            api_main.bot_analyzers[channel] = analyzer
        
        logger.info(f"Registered {len(analyzers)} analyzers with API service")
        logger.info(f"API now has channels: {list(api_main.bot_analyzers.keys())}")
        
    except ImportError as e:
        logger.warning(f"Could not import API main module: {e}")
    except Exception as e:
        logger.error(f"Failed to register analyzers: {e}")

def update_api_analyzers(analyzers: Dict[str, ChatAnalyzerML]):
    """
    Update the FastAPI app's bot_analyzers reference
    Call this whenever analyzer references change
    """
    try:
        import api.main as api_main
        api_main.bot_analyzers.clear()
        api_main.bot_analyzers.update(analyzers)
        logger.info(f"Updated API with {len(analyzers)} analyzers")
    except ImportError:
        logger.warning("Could not import FastAPI main module - API not running")
    except Exception as e:
        logger.error(f"Failed to update analyzers: {e}") 