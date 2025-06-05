from collections import deque, defaultdict
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import time
import threading
import aiohttp
import json
import os
import asyncio
from dataclasses import dataclass
import pickle
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Centralized configuration for the simplified chat analyzer system."""
    
    # Emote spam detection
    EMOTE_SPAM_WINDOW_SIZE: float = 8.0
    EMOTE_SPAM_MIN_MESSAGES: int = 5
    EMOTE_SPAM_MIN_USERS: int = 4
    EMOTE_SPAM_MIN_DENSITY: float = 0.5
    EMOTE_SPAM_MAX_USERS_TRACKED: int = 10000
    EMOTE_SPAM_RATE_LIMIT_WINDOW: float = 2.0
    EMOTE_SPAM_CLEANUP_INTERVAL: float = 1.0
    EMOTE_SPAM_USER_CLEANUP_INTERVAL: float = 30.0
    
    # Emote management
    EMOTE_GLOBAL_UPDATE_INTERVAL: int = 3600  # 1 hour
    EMOTE_CHANNEL_UPDATE_INTERVAL: int = 300  # 5 minutes
    EMOTE_TOKEN_REFRESH_BUFFER: int = 300     # 5 minutes before expiry
    
    # Chat analysis (simplified)
    CHAT_WINDOW_SIZE: int = 30
    CHAT_MESSAGE_HISTORY_SIZE: int = 1000
    CHAT_BASELINE_HISTORY_SIZE: int = 600     # 10 minutes for Z-score calculation
    CHAT_EMOTE_WINDOW_SIZE: int = 8
    CHAT_EMOTE_WINDOW_HISTORY_SIZE: int = 1000
    CHAT_RESET_INTERVAL: float = 1.0
    CHAT_BASELINE_UPDATE_INTERVAL: int = 1
    
    # ML model configuration
    ML_CONTAMINATION_RATE: float = 0.03
    ML_N_ESTIMATORS: int = 100
    ML_MIN_TRAINING_SAMPLES: int = 100
    ML_MIN_BASELINE_SAMPLES: int = 60
    ML_FEATURE_HISTORY_SIZE: int = 1000
    ML_MODEL_UPDATE_INTERVAL: int = 100  # Update every N samples
    ML_MODEL_SAVE_SAMPLE_SIZE: int = 200
    ML_CLIP_THRESHOLD: float = 0.93
    
    # Viewer count scaling
    VIEWER_SMALL_THRESHOLD: int = 1000
    VIEWER_MEDIUM_THRESHOLD: int = 10000
    
    # Feature extraction limits (simplified to 8 features)
    FEATURE_MAX_Z_SCORE: float = 10.0
    FEATURE_MAX_VELOCITY_RELATIVE: float = 10.0
    FEATURE_MAX_BURST_SCORE: float = 100.0
    FEATURE_MAX_BURST_RELATIVE: float = 5.0
    FEATURE_MAX_TOTAL_MESSAGES: int = 1000
    FEATURE_MAX_VIEWER_COUNT: int = 1000000

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables with fallback to defaults."""
        return cls(
            # Allow environment overrides for key parameters
            EMOTE_SPAM_WINDOW_SIZE=float(os.getenv('EMOTE_SPAM_WINDOW_SIZE', cls.EMOTE_SPAM_WINDOW_SIZE)),
            EMOTE_SPAM_MIN_MESSAGES=int(os.getenv('EMOTE_SPAM_MIN_MESSAGES', cls.EMOTE_SPAM_MIN_MESSAGES)),
            EMOTE_SPAM_MIN_USERS=int(os.getenv('EMOTE_SPAM_MIN_USERS', cls.EMOTE_SPAM_MIN_USERS)),
            EMOTE_SPAM_MAX_USERS_TRACKED=int(os.getenv('EMOTE_SPAM_MAX_USERS_TRACKED', cls.EMOTE_SPAM_MAX_USERS_TRACKED)),
            
            EMOTE_GLOBAL_UPDATE_INTERVAL=int(os.getenv('EMOTE_GLOBAL_UPDATE_INTERVAL', cls.EMOTE_GLOBAL_UPDATE_INTERVAL)),
            EMOTE_CHANNEL_UPDATE_INTERVAL=int(os.getenv('EMOTE_CHANNEL_UPDATE_INTERVAL', cls.EMOTE_CHANNEL_UPDATE_INTERVAL)),
            
            CHAT_WINDOW_SIZE=int(os.getenv('CHAT_WINDOW_SIZE', cls.CHAT_WINDOW_SIZE)),
            ML_CONTAMINATION_RATE=float(os.getenv('ML_CONTAMINATION_RATE', cls.ML_CONTAMINATION_RATE)),
            ML_CLIP_THRESHOLD=float(os.getenv('ML_CLIP_THRESHOLD', cls.ML_CLIP_THRESHOLD)),
        )

# Global configuration instance
config = Config.from_env()

@dataclass
class EmoteSpamEvent:
    """Represents a detected emote spam event."""
    emote: str
    count: int
    unique_users: int
    time_window: float
    messages_per_second: float
    consecutive_uses: int
    score: float

@dataclass
class EmoteCache:
    """Container for cached emote data with metadata."""
    emotes: Dict[str, str]  # emote_name -> source
    last_updated: float
    etag: Optional[str] = None
    expires_at: Optional[float] = None

@dataclass 
class TokenCache:
    """Container for cached OAuth token."""
    access_token: str
    expires_at: float
    token_type: str = "Bearer"

class EmoteSpamDetector:
    """Efficient emote spam detection with per-emote tracking and memory management."""
    def __init__(self, 
                 window_size: float = None,
                 min_messages: int = None,
                 min_users: int = None,
                 min_density: float = None,
                 max_users_tracked: int = None):
        # Use config defaults if not provided
        self.window_size = window_size or config.EMOTE_SPAM_WINDOW_SIZE
        self.min_messages = min_messages or config.EMOTE_SPAM_MIN_MESSAGES
        self.min_users = min_users or config.EMOTE_SPAM_MIN_USERS
        self.min_density = min_density or config.EMOTE_SPAM_MIN_DENSITY
        self.max_users_tracked = max_users_tracked or config.EMOTE_SPAM_MAX_USERS_TRACKED
        
        # Per-emote tracking
        self.emote_events = defaultdict(lambda: deque(maxlen=config.CHAT_EMOTE_WINDOW_HISTORY_SIZE))
        self.emote_counts = defaultdict(int)
        self.user_counts = defaultdict(lambda: defaultdict(int))
        self.consecutive_counts = defaultdict(lambda: {'count': 0, 'last_user': None})
        
        # Rate limiting - track when a user last used each emote (NOW BOUNDED)
        self.user_last_used = {}
        self.rate_limit_window = config.EMOTE_SPAM_RATE_LIMIT_WINDOW
        
        # Cache last cleanup time to avoid too frequent cleanups
        self.last_cleanup = time.time()
        self.cleanup_interval = config.EMOTE_SPAM_CLEANUP_INTERVAL
        self.last_user_cleanup = time.time()
        self.user_cleanup_interval = config.EMOTE_SPAM_USER_CLEANUP_INTERVAL
        
    def add_message(self, user_id: str, emotes: Set[str], timestamp: float = None) -> Optional[List[EmoteSpamEvent]]:
        """Process a new message containing emotes. Returns spam events if detected."""
        now = timestamp or time.time()
        spam_events = []
        
        # Clean old events periodically
        if now - self.last_cleanup >= self.cleanup_interval:
            self._cleanup_old_events(now)
            self.last_cleanup = now
            
        # Clean up user tracking periodically
        if now - self.last_user_cleanup >= self.user_cleanup_interval:
            self._cleanup_user_tracking(now)
            self.last_user_cleanup = now
            
        # Process each emote in the message - only once per emote
        for emote in emotes:
            # Check rate limiting with bounded user tracking
            if self._is_rate_limited(user_id, emote, now):
                continue  # Skip this emote due to rate limiting
                
            # Update last used timestamp with LRU management
            self._update_user_last_used(user_id, emote, now)
            
            # Update tracking data structures
            self.emote_events[emote].append((now, user_id))
            self.emote_counts[emote] += 1
            self.user_counts[emote][user_id] += 1
            
            # Track consecutive uses by different users
            consecutive = self.consecutive_counts[emote]
            if consecutive['last_user'] != user_id:
                consecutive['count'] += 1
            consecutive['last_user'] = user_id
            
            # Check for spam
            spam_score = self._check_emote_spam(emote, now)
            if spam_score > 0:
                window_start = now - self.window_size
                events = self.emote_events[emote]
                recent_events = [(ts, uid) for ts, uid in events if ts > window_start]
                
                if recent_events:
                    time_span = max(recent_events[-1][0] - recent_events[0][0], 0.1)
                    msgs_per_sec = len(recent_events) / time_span
                    
                    spam_events.append(EmoteSpamEvent(
                        emote=emote,
                        count=len(recent_events),
                        unique_users=len(set(uid for _, uid in recent_events)),
                        time_window=time_span,
                        messages_per_second=msgs_per_sec,
                        consecutive_uses=consecutive['count'],
                        score=spam_score
                    ))
                    
        return spam_events if spam_events else None
    
    def _is_rate_limited(self, user_id: str, emote: str, now: float) -> bool:
        """Check if user is rate limited for this emote."""
        if user_id not in self.user_last_used:
            return False
        
        if emote not in self.user_last_used[user_id]:
            return False
            
        last_used = self.user_last_used[user_id][emote]
        return now - last_used < self.rate_limit_window
    
    def _update_user_last_used(self, user_id: str, emote: str, now: float):
        """Update user's last used timestamp with efficient memory management."""
        # Initialize user dict if needed
        if user_id not in self.user_last_used:
            # If we're at capacity, remove oldest users during cleanup instead of here
            if len(self.user_last_used) >= self.max_users_tracked:
                # Trigger immediate cleanup if we're at capacity
                self._cleanup_user_tracking(now)
            
            # If still at capacity after cleanup, skip this user (they'll be added later)
            if len(self.user_last_used) >= self.max_users_tracked:
                return
                
            self.user_last_used[user_id] = {}
        
        # Simple O(1) timestamp update
        self.user_last_used[user_id][emote] = now
        
    def _cleanup_user_tracking(self, now: float):
        """Clean up old user tracking data to prevent memory leaks."""
        cutoff = now - (self.rate_limit_window * 10)  # Keep data for 10x the rate limit window
        
        users_to_remove = []
        for user_id, emote_data in self.user_last_used.items():
            # Clean old emote data for this user
            emotes_to_remove = []
            for emote, timestamp in emote_data.items():
                if timestamp < cutoff:
                    emotes_to_remove.append(emote)
            
            # Remove old emotes
            for emote in emotes_to_remove:
                del emote_data[emote]
            
            # If user has no recent emote usage, mark for removal
            if not emote_data:
                users_to_remove.append(user_id)
        
        # Remove users with no recent activity
        for user_id in users_to_remove:
            del self.user_last_used[user_id]
        
        if users_to_remove:
            logger.debug(f"Cleaned up {len(users_to_remove)} inactive users from tracking")
        
    def _check_emote_spam(self, emote: str, now: float) -> float:
        """Calculate simplified spam score for an emote. Returns 0 if not spam, otherwise returns score."""
        window_start = now - self.window_size
        events = self.emote_events[emote]
        recent_events = [(ts, uid) for ts, uid in events if ts > window_start]
        
        if not recent_events:
            return 0.0
            
        # Get basic metrics
        total_uses = len(recent_events)
        unique_users = len(set(uid for _, uid in recent_events))
        time_span = max(recent_events[-1][0] - recent_events[0][0], 0.1)
        msgs_per_sec = total_uses / time_span
        
        # Check minimum thresholds
        if (total_uses < self.min_messages or 
            unique_users < self.min_users or 
            msgs_per_sec < self.min_density):
            return 0.0
            
        # Simplified scoring with 2 components (reduced from 6+)
        # 1. Message density score (0-50 points)
        density_score = min(msgs_per_sec * 10.0, 50.0)
        
        # 2. User participation score (0-50 points)  
        user_participation = min(unique_users / 8.0, 1.0)  # Max at 8 users
        user_score = user_participation * 50.0
        
        # Combine scores
        total_score = density_score + user_score
        
        # Return score if above threshold (reduced from 40 to 30)
        spam_threshold = 30.0
        return total_score if total_score > spam_threshold else 0.0
        
    def _cleanup_old_events(self, now: float):
        """Remove events older than the window size."""
        cutoff = now - self.window_size
        
        for emote in list(self.emote_events.keys()):
            events = self.emote_events[emote]
            while events and events[0][0] < cutoff:
                ts, uid = events.popleft()
                self.emote_counts[emote] -= 1
                self.user_counts[emote][uid] -= 1
                if self.user_counts[emote][uid] <= 0:
                    del self.user_counts[emote][uid]
                    
            # Clean up empty entries
            if not events:
                del self.emote_events[emote]
                del self.emote_counts[emote]
                del self.user_counts[emote]
                del self.consecutive_counts[emote]

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics for monitoring."""
        return {
            'tracked_users': len(self.user_last_used),
            'tracked_emotes': len(self.emote_events),
            'total_events': sum(len(events) for events in self.emote_events.values()),
            'max_users_limit': self.max_users_tracked
        }

class ImprovedEmoteManager:
    """Improved emote manager with caching and incremental updates."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Separate caches for different emote sources
        self.global_twitch_cache = EmoteCache({}, 0)
        self.channel_twitch_caches = {}
        self.global_7tv_cache = EmoteCache({}, 0)
        self.channel_7tv_caches = {}
        
        # Token cache
        self.token_cache: Optional[TokenCache] = None
        
        # Update intervals from config
        self.global_update_interval = config.EMOTE_GLOBAL_UPDATE_INTERVAL
        self.channel_update_interval = config.EMOTE_CHANNEL_UPDATE_INTERVAL
        self.token_refresh_buffer = config.EMOTE_TOKEN_REFRESH_BUFFER
        
        # Combined emote dictionary for backward compatibility
        self.emote_dict = {}
        
    async def update_emotes(self, channel_id: str):
        """Update emote dictionary with intelligent caching (optimized)."""
        try:
            current_time = time.time()
            
            # Quick check: do we need any updates?
            needs_global_twitch = current_time - self.global_twitch_cache.last_updated > self.global_update_interval
            needs_channel_twitch = channel_id and (
                channel_id not in self.channel_twitch_caches or 
                current_time - self.channel_twitch_caches[channel_id].last_updated > self.channel_update_interval
            )
            needs_global_7tv = current_time - self.global_7tv_cache.last_updated > self.global_update_interval
            needs_channel_7tv = channel_id and (
                channel_id not in self.channel_7tv_caches or 
                current_time - self.channel_7tv_caches[channel_id].last_updated > self.channel_update_interval
            )
            
            # Skip if no updates needed
            if not any([needs_global_twitch, needs_channel_twitch, needs_global_7tv, needs_channel_7tv]):
                return
            
            # Single session for all updates
            async with aiohttp.ClientSession() as session:
                # Get token only if we need Twitch API calls
                if needs_global_twitch or needs_channel_twitch:
                    token = await self._get_valid_token()
                    if not token:
                        self.logger.error("Could not obtain valid OAuth token")
                        return
                        
                    headers = {
                        "Client-ID": os.getenv('TWITCH_CLIENT_ID'),
                        "Authorization": f"Bearer {token}"
                    }
                    
                    # Update Twitch emotes
                    tasks = []
                    if needs_global_twitch:
                        tasks.append(self._update_global_twitch_emotes(session, headers))
                    if needs_channel_twitch:
                        tasks.append(self._update_channel_twitch_emotes(session, headers, channel_id))
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update 7TV emotes
                tasks = []
                if needs_global_7tv:
                    tasks.append(self._update_global_7tv_emotes(session))
                if needs_channel_7tv:
                    tasks.append(self._update_channel_7tv_emotes(session, channel_id))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            # Rebuild combined emote dictionary
            self._rebuild_emote_dict()
            
        except Exception as e:
            self.logger.error(f"Error updating emotes: {e}", exc_info=True)
    
    async def _get_valid_token(self) -> Optional[str]:
        """Get a valid OAuth token, refreshing if necessary."""
        current_time = time.time()
        
        # Check if we have a valid cached token
        if (self.token_cache and 
            current_time < self.token_cache.expires_at - self.token_refresh_buffer):
            return self.token_cache.access_token
        
        # Request new token
        client_id = os.getenv('TWITCH_CLIENT_ID')
        client_secret = os.getenv('TWITCH_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            self.logger.error("Missing Twitch API credentials")
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                auth_url = "https://id.twitch.tv/oauth2/token"
                auth_params = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "client_credentials"
                }
                
                async with session.post(auth_url, params=auth_params) as resp:
                    if resp.status == 200:
                        token_data = await resp.json()
                        access_token = token_data["access_token"]
                        expires_in = token_data.get("expires_in", 3600)
                        
                        # Cache the token
                        self.token_cache = TokenCache(
                            access_token=access_token,
                            expires_at=current_time + expires_in
                        )
                        
                        self.logger.info("Successfully obtained and cached new Twitch API token")
                        return access_token
                    else:
                        self.logger.error(f"Failed to get Twitch token: {resp.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting OAuth token: {e}", exc_info=True)
            return None
    
    async def _update_global_twitch_emotes(self, session: aiohttp.ClientSession, headers: dict):
        """Update global Twitch emotes cache."""
        try:
            self.logger.info("Updating global Twitch emotes...")
            
            async with session.get(
                "https://api.twitch.tv/helix/chat/emotes/global",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    emotes_data = data.get("data", [])
                    
                    new_emotes = {}
                    for emote in emotes_data:
                        name = emote.get("name", "").strip()
                        if name and len(name) >= 2:
                            new_emotes[name] = "twitch:global"
                    
                    # Update cache
                    self.global_twitch_cache = EmoteCache(
                        emotes=new_emotes,
                        last_updated=time.time(),
                        etag=resp.headers.get('ETag')
                    )
                    
                    self.logger.info(f"Updated {len(new_emotes)} global Twitch emotes")
                    
                elif resp.status == 304:  # Not Modified
                    self.logger.info("Global Twitch emotes not modified")
                    self.global_twitch_cache.last_updated = time.time()
                else:
                    self.logger.error(f"Failed to get global Twitch emotes: {resp.status}")
                    
        except Exception as e:
            self.logger.error(f"Error updating global Twitch emotes: {e}", exc_info=True)
    
    async def _update_channel_twitch_emotes(self, session: aiohttp.ClientSession, headers: dict, channel_id: str):
        """Update channel-specific Twitch emotes cache."""
        try:
            self.logger.info(f"Updating Twitch emotes for channel {channel_id}...")
            
            async with session.get(
                f"https://api.twitch.tv/helix/chat/emotes?broadcaster_id={channel_id}",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    emotes_data = data.get("data", [])
                    
                    new_emotes = {}
                    for emote in emotes_data:
                        name = emote.get("name", "").strip()
                        if name and len(name) >= 2:
                            new_emotes[name] = f"twitch:channel:{channel_id}"
                    
                    # Update cache
                    self.channel_twitch_caches[channel_id] = EmoteCache(
                        emotes=new_emotes,
                        last_updated=time.time(),
                        etag=resp.headers.get('ETag')
                    )
                    
                    self.logger.info(f"Updated {len(new_emotes)} Twitch emotes for channel {channel_id}")
                    
                elif resp.status == 304:  # Not Modified
                    self.logger.info(f"Twitch emotes for channel {channel_id} not modified")
                    if channel_id in self.channel_twitch_caches:
                        self.channel_twitch_caches[channel_id].last_updated = time.time()
                else:
                    self.logger.error(f"Failed to get channel Twitch emotes: {resp.status}")
                    
        except Exception as e:
            self.logger.error(f"Error updating channel Twitch emotes: {e}", exc_info=True)
    
    async def _update_global_7tv_emotes(self, session: aiohttp.ClientSession):
        """Update global 7TV emotes cache."""
        try:
            self.logger.info("Updating global 7TV emotes...")
            
            async with session.get("https://7tv.io/v3/emote-sets/global") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    new_emotes = {}
                    for emote in data.get("emotes", []):
                        name = emote.get("name", "").strip()
                        if name and len(name) >= 2:
                            new_emotes[name] = "7tv:global"
                    
                    # Update cache
                    self.global_7tv_cache = EmoteCache(
                        emotes=new_emotes,
                        last_updated=time.time()
                    )
                    
                    self.logger.info(f"Updated {len(new_emotes)} global 7TV emotes")
                else:
                    self.logger.error(f"Failed to get global 7TV emotes: {resp.status}")
                    
        except Exception as e:
            self.logger.error(f"Error updating global 7TV emotes: {e}", exc_info=True)
    
    async def _update_channel_7tv_emotes(self, session: aiohttp.ClientSession, channel_id: str):
        """Update channel-specific 7TV emotes cache."""
        try:
            self.logger.info(f"Updating 7TV emotes for channel {channel_id}...")
            
            # Get 7TV user connection
            stv_user_data = await self._get_7tv_user_connection(session, channel_id)
            if not stv_user_data:
                return
            
            # Get emote set
            emote_set = await self._get_7tv_emote_set(session, stv_user_data)
            if not emote_set:
                return
            
            new_emotes = {}
            for emote in emote_set.get("emotes", []):
                name = emote.get("name", "").strip()
                if name and len(name) >= 2:
                    new_emotes[name] = f"7tv:channel:{channel_id}"
            
            # Update cache
            self.channel_7tv_caches[channel_id] = EmoteCache(
                emotes=new_emotes,
                last_updated=time.time()
            )
            
            self.logger.info(f"Updated {len(new_emotes)} 7TV emotes for channel {channel_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating channel 7TV emotes: {e}", exc_info=True)
    
    def _rebuild_emote_dict(self):
        """Rebuild the combined emote dictionary from all caches."""
        self.emote_dict.clear()
        
        # Add global Twitch emotes
        self.emote_dict.update(self.global_twitch_cache.emotes)
        
        # Add channel Twitch emotes
        for cache in self.channel_twitch_caches.values():
            self.emote_dict.update(cache.emotes)
        
        # Add global 7TV emotes
        self.emote_dict.update(self.global_7tv_cache.emotes)
        
        # Add channel 7TV emotes
        for cache in self.channel_7tv_caches.values():
            self.emote_dict.update(cache.emotes)
        
        self.logger.info(f"Rebuilt emote dictionary with {len(self.emote_dict)} total emotes")
        
    def get_emotes(self) -> Set[str]:
        """Get the current set of all known emotes."""
        return set(self.emote_dict.keys())
        
    def get_channel_emotes(self, channel_id: str) -> Set[str]:
        """Get emotes specific to a channel."""
        return {
            emote for emote, source in self.emote_dict.items()
            if f"channel:{channel_id}" in source
        }
    
    def _get_emote_source_stats(self) -> str:
        """Get statistics about emote sources."""
        stats = {}
        for source in self.emote_dict.values():
            stats[source] = stats.get(source, 0) + 1
        return ", ".join(f"{k}: {v}" for k, v in sorted(stats.items()))

    async def _get_7tv_user_connection(self, session: aiohttp.ClientSession, twitch_id: str) -> Optional[dict]:
        """Get 7TV user connection from Twitch ID."""
        try:
            # Get user by Twitch ID
            async with session.get(f"https://7tv.io/v3/users/twitch/{twitch_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.logger.info(f"Found 7TV user data")
                    return data
                else:
                    self.logger.warning(f"Failed to get 7TV user. Status: {resp.status}")
                    
        except Exception as e:
            self.logger.error(f"Error getting 7TV user connection: {e}", exc_info=True)
        return None

    async def _get_7tv_emote_set(self, session: aiohttp.ClientSession, user_data: dict) -> Optional[dict]:
        """Get 7TV emote set for a user."""
        try:
            # Get emote set ID from user data
            emote_set_id = user_data.get("emote_set", {}).get("id")
            if not emote_set_id:
                self.logger.warning("No active emote set ID found in user data")
                return None
                
            # Get the emote set
            return await self._get_7tv_emote_set_by_id(session, emote_set_id)
                    
        except Exception as e:
            self.logger.error(f"Error getting 7TV emote set: {e}", exc_info=True)
        return None

    async def _get_7tv_emote_set_by_id(self, session: aiohttp.ClientSession, emote_set_id: str) -> Optional[dict]:
        """Get 7TV emote set directly by set ID."""
        try:
            async with session.get(f"https://7tv.io/v3/emote-sets/{emote_set_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.logger.info(f"Found 7TV emote set with {len(data.get('emotes', []))} emotes")
                    return data
                else:
                    self.logger.warning(f"Failed to get emote set. Status: {resp.status}")
                    
        except Exception as e:
            self.logger.error(f"Error getting 7TV emote set: {e}", exc_info=True)
        return None

# Keep legacy EmoteManager class for backward compatibility
EmoteManager = ImprovedEmoteManager

class ChatAnalyzerML:
    def __init__(self, window_size: int = None, channel: str = None):
        """
        Initialize the ML-enhanced chat analyzer with simplified tracking.
        
        Args:
            window_size (int): Size of the sliding window in seconds
            channel (str): Channel name
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Use config defaults
        self.window_size = window_size or config.CHAT_WINDOW_SIZE
        self.channel = channel
        
        # Core data structures (simplified)
        self.messages = deque(maxlen=config.CHAT_MESSAGE_HISTORY_SIZE)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.viewer_count = 0
        
        # Message velocity tracking
        self.msg_count = 0
        self.last_reset = time.time()
        self.current_rate = 0
        
        # Start the reset timer in a separate thread
        self.running = True
        self.reset_thread = threading.Thread(target=self._reset_counter, daemon=True)
        self.reset_thread.start()
        
        # Baseline velocity tracking (kept for Z-score calculation)
        self.baseline_velocities = deque(maxlen=config.CHAT_BASELINE_HISTORY_SIZE)
        self.last_baseline_update = datetime.now()
        self.baseline_update_interval = config.CHAT_BASELINE_UPDATE_INTERVAL
        
        # Simplified emote tracking
        self.emote_window = deque(maxlen=config.CHAT_EMOTE_WINDOW_HISTORY_SIZE)
        self.emote_window_size = config.CHAT_EMOTE_WINDOW_SIZE
        
        # ML components with config parameters
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=config.ML_CONTAMINATION_RATE,
            random_state=42,
            n_estimators=config.ML_N_ESTIMATORS,
            max_samples='auto',
            bootstrap=True
        )
        
        # Feature history for training (reduced to 8 features)
        self.feature_history = deque(maxlen=config.ML_FEATURE_HISTORY_SIZE)
        
        # Initialize emote tracking
        self.emote_manager = EmoteManager()
        
        # Load base emotes
        self.load_twitch_emotes()
        self.logger.debug(f"Loaded {len(self.base_hype_emotes)} known emotes")
        
        # Add emote spam detector with config parameters
        self.emote_spam_detector = EmoteSpamDetector()
        
        # Create model directory if needed
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Load ML model if exists
        self.load_ml_model()
        
        # Initialize emotes and base sets
        self.emotes = {}
        self.base_hype_emotes = set()
        self.channel_emotes = set()

    def _reset_counter(self):
        """Reset message counter every second in a separate thread."""
        while self.running:
            time.sleep(config.CHAT_RESET_INTERVAL)
            self.current_rate = self.msg_count
            self.msg_count = 0
            self.last_reset = time.time()

    def update_viewer_count(self, count: int) -> None:
        """Update the current viewer count."""
        self.viewer_count = max(1, count)

    def add_message(self, message: str, user_id: str, timestamp: datetime) -> None:
        """Add a new chat message with user information."""
        # Increment message counter
        self.msg_count += 1
        
        # Extract and track emotes
        found_emotes = self._extract_emotes(message)
        
        # Check for emote spam
        if found_emotes:
            spam_events = self.emote_spam_detector.add_message(
                user_id=user_id,
                emotes=found_emotes,
                timestamp=timestamp.timestamp()
            )
            
            if spam_events:
                # Log spam events
                for event in spam_events:
                    self.logger.info(
                        f"Emote spam detected: {event.emote} "
                        f"({event.count} uses by {event.unique_users} users "
                        f"at {event.messages_per_second:.1f} msg/sec, "
                        f"score: {event.score:.1f})"
                    )
                    
            # Add to emote window for general burst detection
            self.emote_window.append({
                'emotes': found_emotes,
                'timestamp': timestamp,
                'user_id': user_id,
                'spam_events': spam_events
            })
            
        # Process message with immediate sentiment calculation
        text = message.strip()
        sentiment = self.sentiment_analyzer.polarity_scores(text) if text else {'compound': 0.0}
        
        msg_data = {
            'text': message,
            'user_id': user_id,
            'timestamp': timestamp,
            'sentiment': sentiment  # Calculate once and store
        }
        self.messages.append(msg_data)
        
        # Update baseline velocities
        now = time.time()
        if now - self.last_baseline_update.timestamp() >= self.baseline_update_interval:
            if self.current_rate > 0:
                self.baseline_velocities.append(self.current_rate)
            self.last_baseline_update = datetime.fromtimestamp(now)
        
    def get_chat_velocity(self) -> float:
        """Get the current messages per second."""
        return self.current_rate
        
    def get_velocity_zscore(self) -> float:
        """Calculate Z-score of current velocity compared to historical baseline."""
        if len(self.baseline_velocities) < 60:  # Need 1 minute of baseline data
            return 0.0
            
        current_velocity = self.get_chat_velocity()
        
        # Calculate using baseline stats
        velocities = np.array(list(self.baseline_velocities))
        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)
        
        # Require some minimum variance to consider spikes meaningful
        if std_velocity < 0.1:
            std_velocity = 0.1
            
        # Calculate Z-score
        z_score = (current_velocity - mean_velocity) / std_velocity
        
        # Smooth extreme values
        return np.clip(z_score, -5.0, 5.0)  # Limit to ±5 standard deviations
        
    def get_normalized_chat_velocity(self) -> float:
        """
        Get normalized chat velocity using Z-score.
        Returns a value that represents how unusual the current velocity is.
        """
        z_score = self.get_velocity_zscore()
        
        # Convert Z-score to a more intuitive scale
        # Values will typically range from 0 to ~5, where:
        # 0-1: Normal velocity
        # 1-2: Somewhat high (1.5 std devs above mean)
        # 2-3: Very high (2 std devs above mean)
        # 3+: Extremely high (3+ std devs above mean)
        
        # Smooth the transition and cap at 5
        normalized_score = min(max(0, z_score), 5.0)
        
        # Apply non-linear scaling to emphasize larger spikes
        if normalized_score > 2.0:
            normalized_score = 2.0 + (normalized_score - 2.0) * 1.5
            
        return normalized_score
        
    def get_average_sentiment(self) -> float:
        """Calculate average sentiment in current window (using pre-calculated values)."""
        if not self.messages:
            return 0.0
            
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_size)
        
        try:
            # Get messages in window (sentiment already calculated)
            window_messages = [
                msg for msg in self.messages
                if msg['timestamp'] > window_start and 'sentiment' in msg
            ]
            
            if not window_messages:
                return 0.0
            
            # Extract pre-calculated sentiments
            sentiments = [msg['sentiment']['compound'] for msg in window_messages]
            return np.mean(sentiments)
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}", exc_info=True)
            return 0.0

    def _extract_emotes(self, message: str) -> Set[str]:
        """Extract known emotes from a message (optimized)."""
        # Split message into words and find emotes efficiently
        words = set(message.split())
        emotes = words.intersection(self.base_hype_emotes)
        
        # Only log if we find emotes and logging level is debug
        if emotes and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Found emotes: {emotes}")
            
        return emotes

    def load_twitch_emotes(self):
        """Initialize emote sets with common Twitch emotes for better detection before API loads."""
        self.logger.info("Initializing emote tracking with default emotes...")
        
        # Initialize with common Twitch emotes for better initial detection
        twitch_default_emotes = {
            # Twitch Global Emotes
            "PogChamp", "Kappa", "LUL", "KappaPride", "PrideLove", "CoolCat", "DansGame", "FailFish", 
            "Jebaited", "Kreygasm", "ResidentSleeper", "TriHard", "HeyGuys", "VoHiYo", "BibleThump",
            "WutFace", "BabyRage", "SeemsGood", "NotLikeThis", "PJSalt", "FrankerZ", "TwitchUnity",
            
            # Common Text Emotes
            ":)", ":D", ":(", ":o", ";)", ":/", ":|", "O_o", ":p", "<3", "R)", "B)", 
            
            # Popular Third-Party Emotes
            "KEKW", "OMEGALUL", "PogU", "PepeLaugh", "WeirdChamp", "Pog", "monkaS", "pepeD",
            "widepeepoHappy", "LULW", "Sadge", "PepeHands", "HYPERS", "monkaW", "5Head",
            "Pepega", "catJAM", "AYAYA", "PepoG", "PauseChamp", "Copium", "peepoGlad"
        }
        
        # Add emotes to our dictionaries with sources for debugging
        self.emotes = {emote: "default:twitch" for emote in twitch_default_emotes}
        self.base_hype_emotes = twitch_default_emotes.copy()
        self.channel_emotes = set()
        
        # Log what we've loaded
        self.logger.info(f"Initialized with {len(self.base_hype_emotes)} default emotes")
        self.logger.debug(f"Default Twitch emotes loaded: {sorted(twitch_default_emotes)[:20]} (+ {len(twitch_default_emotes)-20} more)")
        
        # Try to prevent emotes from being cleared too quickly while API loads
        self.last_update = time.time() - 200  # Set last update to 200 seconds ago

    async def update_emotes(self, channel_id: str = None, emote_set_id: str = None) -> None:
        """Update emotes from Twitch and 7TV."""
        try:
            # Delegate to emote manager
            await self.emote_manager.update_emotes(channel_id)
            
            # Update our local emote sets from the manager
            self.emotes = self.emote_manager.emote_dict
            self.base_hype_emotes = set(self.emote_manager.emote_dict.keys())
            self.channel_emotes = self.emote_manager.get_channel_emotes(channel_id) if channel_id else set()
            
            self.logger.info(f"Updated emotes from emote manager: {len(self.emotes)} total emotes")
            
        except Exception as e:
            self.logger.error(f"Error updating emotes: {e}", exc_info=True)

    def load_ml_model(self):
        """Load the ML model, scaler, and feature history from disk if available."""
        if not self.channel:
            self.logger.warning("Cannot load ML model: No channel name provided")
            return False
            
        try:
            # Create safe filename from channel name
            safe_channel = self.channel.lower().replace(' ', '_')
            base_path = self.models_dir / safe_channel
            
            # Check if model files exist
            model_path = base_path.with_suffix('.joblib')
            scaler_path = base_path.with_name(f"{safe_channel}_scaler.joblib")
            features_path = base_path.with_name(f"{safe_channel}_features.pkl")
            baseline_path = base_path.with_name(f"{safe_channel}_baseline.pkl")
            
            if not model_path.exists() or not scaler_path.exists():
                self.logger.info(f"No saved ML model found for channel {self.channel}")
                return False
                
            # Load isolation forest model
            self.isolation_forest = joblib.load(model_path)
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Load feature history if available
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    loaded_features = pickle.load(f)
                    # Convert list back to deque
                    self.feature_history = deque(loaded_features, maxlen=config.ML_FEATURE_HISTORY_SIZE)
                    
            # Load baseline velocities if available
            if baseline_path.exists():
                with open(baseline_path, 'rb') as f:
                    loaded_baseline = pickle.load(f)
                    # Convert list back to deque
                    self.baseline_velocities = deque(loaded_baseline, maxlen=config.CHAT_BASELINE_HISTORY_SIZE)
                    
            self.logger.info(
                f"ML model loaded for channel {self.channel}: "
                f"{len(self.feature_history)} features, "
                f"{len(self.baseline_velocities)} baseline samples"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}", exc_info=True)
            # Reset to new models if loading failed
            self.scaler = StandardScaler()
            self.isolation_forest = IsolationForest(
                contamination=config.ML_CONTAMINATION_RATE,
                random_state=42,
                n_estimators=config.ML_N_ESTIMATORS,
                max_samples='auto',
                bootstrap=True
            )
            return False

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.running = False
        if hasattr(self, 'reset_thread'):
            self.reset_thread.join(timeout=1.0)
            
        # Save ML model on exit if we have enough data
        if hasattr(self, 'feature_history') and len(self.feature_history) >= config.ML_MIN_TRAINING_SAMPLES:
            self.save_ml_model()

    def get_emote_burst_score(self) -> float:
        """
        Simplified emote burst score focusing on core signals.
        Reduced complexity for better performance and future extensibility.
        """
        if not self.emote_window:
            return 0.0
        
        now = datetime.now()
        window_start = now - timedelta(seconds=self.emote_window_size)
        
        # Get recent emote usage
        recent_entries = [e for e in self.emote_window if e['timestamp'] > window_start]
        
        if not recent_entries:
            return 0.0
        
        # Simplified tracking: focus on most active emote only
        emote_usage = defaultdict(lambda: defaultdict(list))
        
        for entry in recent_entries:
            user_id = entry['user_id']
            timestamp = entry['timestamp']
            for emote in entry['emotes']:
                emote_usage[emote][user_id].append(timestamp)
        
        if not emote_usage:
            return 0.0
        
        # Get the most popular emote (highest user count)
        top_emote = max(emote_usage.items(), key=lambda x: len(x[1]))
        emote_name, users = top_emote
        
        unique_users_count = len(users)
        total_users_in_window = len(set(e['user_id'] for e in recent_entries))
        
        # Core scoring components (simplified from 6 to 3)
        # 1. User participation (40 points max)
        user_participation = min(unique_users_count / max(total_users_in_window, 1), 1.0)
        user_score = user_participation * 40.0
        
        # 2. Message density (40 points max)
        all_timestamps = [ts for user_timestamps in users.values() for ts in user_timestamps]
        if len(all_timestamps) >= 2:
            time_span = (max(all_timestamps) - min(all_timestamps)).total_seconds()
            time_span = max(time_span, 0.1)
            density = len(all_timestamps) / time_span
            density_score = min(density * 8.0, 40.0)  # Scale density appropriately
        else:
            density_score = 0.0
        
        # 3. Viewer participation bonus (20 points max)
        viewer_count = max(self.viewer_count, 1)
        viewer_participation = min(unique_users_count / (viewer_count * 0.01), 1.0)
        viewer_bonus = viewer_participation * 20.0
        
        # Combine scores
        final_score = user_score + density_score + viewer_bonus
        
        # Log for debugging if significant
        if final_score > 15:
            self.logger.debug(
                f"Emote burst: {emote_name} | "
                f"Users: {unique_users_count}/{total_users_in_window} | "
                f"Score: {final_score:.1f} (user:{user_score:.1f}, density:{density_score:.1f}, viewer:{viewer_bonus:.1f})"
            )
        
        return min(final_score, config.FEATURE_MAX_BURST_SCORE)

    def get_window_stats(self) -> Dict:
        """Get simplified window statistics with ML scores."""
        try:
            # Get base stats using simplified methods
            raw_velocity = self.get_chat_velocity()
            velocity_zscore = self.get_velocity_zscore()
            burst_score = self.get_emote_burst_score()
            
            # Get viewer count with minimum of 1
            viewer_count = max(self.viewer_count, 1)
            
            # Calculate scale factor for relative metrics
            if viewer_count < config.VIEWER_SMALL_THRESHOLD:
                scale_factor = 0.3 + (0.5 * (viewer_count / config.VIEWER_SMALL_THRESHOLD))
            elif viewer_count < config.VIEWER_MEDIUM_THRESHOLD:
                scale_factor = 0.8 + (0.4 * ((viewer_count - config.VIEWER_SMALL_THRESHOLD) / 
                                              (config.VIEWER_MEDIUM_THRESHOLD - config.VIEWER_SMALL_THRESHOLD)))
            else:
                scale_factor = 1.2 + min(0.6, 0.6 * ((viewer_count - config.VIEWER_MEDIUM_THRESHOLD) / 40000))
            
            # Calculate relative metrics
            velocity_relative = velocity_zscore / max(scale_factor, 0.1)
            burst_relative = burst_score / max(scale_factor * 15.0, 1.0)
            
            # Simplified rule-based score
            rule_score = 0.0
            if velocity_zscore > 2.0:  # Strong velocity spike
                rule_score += min(velocity_zscore / 8.0, 0.6)
            if burst_score > 20.0:  # Strong burst activity
                rule_score += min(burst_score / 80.0, 0.6)
                
            # Get ML prediction if we have enough data
            if len(self.feature_history) >= config.ML_MIN_TRAINING_SAMPLES and len(self.baseline_velocities) >= config.ML_MIN_BASELINE_SAMPLES:
                is_worthy, ml_score = self.is_clip_worthy()
            else:
                ml_score = 0.0
                # Basic fallback scoring
                if velocity_relative > 3.0:
                    ml_score += min(velocity_relative / 5.0, 0.7)
                if burst_relative > 1.5:
                    ml_score += min(burst_relative / 3.0, 0.6)
            
            # Simplified hybrid score
            if ml_score > 0.8:
                clip_worthy_score = ml_score * 0.8 + rule_score * 0.2
            elif rule_score > 0.7:
                clip_worthy_score = rule_score * 0.7 + ml_score * 0.3
            else:
                clip_worthy_score = rule_score * 0.5 + ml_score * 0.5
            
            clip_worthy_score = min(max(clip_worthy_score, 0.0), 1.0)
            
            # Add sentiment calculation
            try:
                sentiment = self.get_average_sentiment()
            except Exception as e:
                self.logger.debug(f"Error calculating sentiment: {e}")
                sentiment = 0.0
            
            stats = {
                'viewer_count': self.viewer_count,
                'raw_velocity': raw_velocity,
                'velocity_zscore': velocity_zscore,
                'velocity_relative': velocity_relative,
                'burst_score': burst_score,
                'burst_relative': burst_relative,
                'rule_score': rule_score,
                'ml_score': ml_score,
                'clip_worthy_score': clip_worthy_score,
                'sentiment': sentiment
            }
            
            # Debug logging for high scores
            if clip_worthy_score > 0.5:
                logger.info(f"High clip worthiness: {clip_worthy_score:.3f} (rule: {rule_score:.2f}, ML: {ml_score:.2f}) for {viewer_count} viewers")
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting window stats: {e}", exc_info=True)
            return {
                'viewer_count': self.viewer_count,
                'raw_velocity': 0.0,
                'velocity_zscore': 0.0,
                'burst_score': 0.0,
                'clip_worthy_score': 0.0,
                'sentiment': 0.0
            }

    def save_ml_model(self):
        """Save the ML model, scaler, and sample feature history to disk."""
        if not self.channel:
            self.logger.warning("Cannot save ML model: No channel name provided")
            return False
            
        try:
            # Create safe filename from channel name
            safe_channel = self.channel.lower().replace(' ', '_')
            base_path = self.models_dir / safe_channel
            
            # Save isolation forest model
            model_path = base_path.with_suffix('.joblib')
            joblib.dump(self.isolation_forest, model_path)
            
            # Save scaler
            scaler_path = base_path.with_name(f"{safe_channel}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature history (convert deque to list for serialization) using config sample size
            if len(self.feature_history) > 0:
                features_path = base_path.with_name(f"{safe_channel}_features.pkl")
                sample_features = list(self.feature_history)[-config.ML_MODEL_SAVE_SAMPLE_SIZE:]
                with open(features_path, 'wb') as f:
                    pickle.dump(sample_features, f)
                    
            # Save baseline velocities using config sample size
            if len(self.baseline_velocities) > 0:
                baseline_path = base_path.with_name(f"{safe_channel}_baseline.pkl")
                sample_baseline = list(self.baseline_velocities)[-config.ML_MODEL_SAVE_SAMPLE_SIZE:]
                with open(baseline_path, 'wb') as f:
                    pickle.dump(sample_baseline, f)
                    
            self.logger.info(f"ML model saved for channel {self.channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ML model: {e}", exc_info=True)
            return False

    def is_clip_worthy(self) -> Tuple[bool, float]:
        """
        Determine if moment is clip-worthy using unsupervised learning.
        Returns (is_worthy, anomaly_score)
        """
        try:
            # Extract current features
            features = self.extract_features()
            
            # Add to history
            self.feature_history.append(features)
            
            # Need enough data for meaningful detection using config values
            if len(self.feature_history) < config.ML_MIN_TRAINING_SAMPLES or len(self.baseline_velocities) < config.ML_MIN_BASELINE_SAMPLES:
                # Fall back to rule-based approach if not enough data
                velocity_relative = features[3] if len(features) > 3 else 0.0  # Velocity relative to channel size
                burst_relative = features[6] if len(features) > 6 else 0.0     # Burst score relative to channel size
                
                # Simple rule-based approach until we have enough data
                rule_score = 0.0
                if velocity_relative > 3.0:
                    rule_score += min(velocity_relative / 5.0, 0.7)
                if burst_relative > 1.5:
                    rule_score += min(burst_relative / 3.0, 0.6)
                    
                return rule_score > 0.5, rule_score
                
            # Update model periodically using config interval
            if len(self.feature_history) % config.ML_MODEL_UPDATE_INTERVAL == 0:
                self.update_model()
                
            # Scale features with error handling
            try:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            except Exception:
                self.logger.error("Error scaling features, trying to rebuild scaler")
                # Try rebuilding scaler
                X = np.array(list(self.feature_history))
                self.scaler.fit(X)
                # Try again
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get anomaly score (-1 for anomalies, 1 for normal)
            try:
                score = self.isolation_forest.score_samples(features_scaled)[0]
            except Exception as e:
                self.logger.error(f"Error getting anomaly score: {e}")
                return False, 0.0
            
            # Convert to probability-like score (0 to 1, higher means more anomalous)
            prob_score = 1 - (score + 1) / 2
            
            # Boost score based on velocity and burst features
            velocity_relative = features[3] if len(features) > 3 else 0.0
            burst_relative = features[6] if len(features) > 6 else 0.0
            
            # Boost based on strong relative signals (already accounting for channel size)
            if velocity_relative > 2.0:  # Strong velocity relative to channel size
                vel_boost = min(velocity_relative / 10.0, 0.25)
                prob_score = min(prob_score + vel_boost, 1.0)
                
            if burst_relative > 1.5:  # Strong burst relative to channel size
                burst_boost = min(burst_relative / 8.0, 0.25)
                prob_score = min(prob_score + burst_boost, 1.0)
                
            # Log high-scoring moments for debugging
            if prob_score > 0.8:
                self.logger.info(
                    f"High clip score: {prob_score:.3f} | "
                    f"Velocity: {velocity_relative:.2f} | "
                    f"Burst: {burst_relative:.2f} | "
                    f"Viewers: {self.viewer_count:,}"
                )
                
            # Use config threshold
            return prob_score > config.ML_CLIP_THRESHOLD, prob_score
            
        except Exception as e:
            logger.error(f"Error in clip worthiness check: {e}", exc_info=True)
            return False, 0.0

    def update_model(self):
        """Update the isolation forest model with new data."""
        if len(self.feature_history) < config.ML_MIN_TRAINING_SAMPLES:
            return
            
        try:
            # Convert feature history to numpy array
            X = np.array(list(self.feature_history))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Configure isolation forest with config parameters
            self.isolation_forest = IsolationForest(
                contamination=config.ML_CONTAMINATION_RATE,
                random_state=42,
                n_estimators=config.ML_N_ESTIMATORS,
                max_samples='auto',
                bootstrap=True
            )
            
            # Fit isolation forest
            self.isolation_forest.fit(X_scaled)
            
            # Save model to disk
            self.save_ml_model()
            
            # Log success
            self.logger.info(f"ML model updated with {len(self.feature_history)} samples")
        except Exception as e:
            self.logger.error(f"Error updating ML model: {e}", exc_info=True)

    def extract_features(self) -> np.ndarray:
        """
        Extract simplified feature vector focusing on core predictive metrics.
        Reduced from 15 to 8 features to make room for future video/audio features.
        """
        try:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_size)
            
            # Get messages in current window
            window_messages = [
                msg for msg in self.messages
                if msg['timestamp'] > window_start
            ]
            
            if not window_messages:
                return np.zeros(8)  # Reduced feature count
                
            # Core metrics with safeguards
            try:
                velocity_zscore = self.get_velocity_zscore()  # Single velocity metric
                burst_score = self.get_emote_burst_score()
                sentiment = self.get_average_sentiment()
            except Exception:
                velocity_zscore = 0.0
                burst_score = 0.0
                sentiment = 0.0
            
            # Message analysis
            message_texts = [msg['text'].lower() for msg in window_messages]
            total_messages = len(message_texts)
            unique_users = len(set(msg['user_id'] for msg in window_messages))
            
            # Simplified user engagement ratio
            user_ratio = unique_users / max(total_messages, 1)
            
            # Viewer-scaled metrics using config thresholds
            viewer_count = max(self.viewer_count, 1)
            
            # Calculate single scale factor based on viewer count
            if viewer_count < config.VIEWER_SMALL_THRESHOLD:
                scale_factor = 0.3 + (0.5 * (viewer_count / config.VIEWER_SMALL_THRESHOLD))
            elif viewer_count < config.VIEWER_MEDIUM_THRESHOLD:
                scale_factor = 0.8 + (0.4 * ((viewer_count - config.VIEWER_SMALL_THRESHOLD) / 
                                              (config.VIEWER_MEDIUM_THRESHOLD - config.VIEWER_SMALL_THRESHOLD)))
            else:
                scale_factor = 1.2 + min(0.6, 0.6 * ((viewer_count - config.VIEWER_MEDIUM_THRESHOLD) / 40000))
                
            # Scale velocity and burst relative to channel size
            velocity_relative = velocity_zscore / max(scale_factor, 0.1)
            burst_relative = burst_score / max(scale_factor * 15.0, 1.0)
            
            # Simplified 8-feature vector focused on core metrics
            features = np.array([
                min(max(velocity_zscore, 0.0), config.FEATURE_MAX_Z_SCORE),      # 0: Chat velocity anomaly
                min(max(velocity_relative, 0.0), config.FEATURE_MAX_VELOCITY_RELATIVE), # 1: Velocity relative to channel
                min(max(burst_score, 0.0), config.FEATURE_MAX_BURST_SCORE),     # 2: Emote burst score
                min(max(burst_relative, 0.0), config.FEATURE_MAX_BURST_RELATIVE), # 3: Burst relative to channel
                min(max(sentiment, -1.0), 1.0),                                 # 4: Average sentiment
                min(max(user_ratio, 0.0), 1.0),                                 # 5: User engagement ratio
                min(total_messages, config.FEATURE_MAX_TOTAL_MESSAGES),         # 6: Message count
                np.log10(min(max(viewer_count, 1), config.FEATURE_MAX_VIEWER_COUNT)) # 7: Log viewer count
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}", exc_info=True)
            return np.zeros(8)  # Return 8-element zero vector