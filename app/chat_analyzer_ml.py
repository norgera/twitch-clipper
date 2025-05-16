from collections import deque, Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, NamedTuple
import math
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import time
import threading
from difflib import SequenceMatcher
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
class EmoteSpamEvent:
    """Represents a detected emote spam event."""
    emote: str
    count: int
    unique_users: int
    time_window: float
    messages_per_second: float
    consecutive_uses: int
    score: float

class EmoteSpamDetector:
    """Efficient emote spam detection with per-emote tracking."""
    def __init__(self, 
                 window_size: float = 8.0,
                 min_messages: int = 5,
                 min_users: int = 4,   # Increased to require at least 4 unique users
                 min_density: float = 0.5):
        self.window_size = window_size
        self.min_messages = min_messages
        self.min_users = min_users
        self.min_density = min_density  # messages per second threshold
        
        # Per-emote tracking
        self.emote_events = defaultdict(lambda: deque(maxlen=1000))  # emote -> deque[(timestamp, user_id)]
        self.emote_counts = defaultdict(int)  # emote -> count in current window
        self.user_counts = defaultdict(lambda: defaultdict(int))  # emote -> {user_id -> count}
        self.consecutive_counts = defaultdict(lambda: {'count': 0, 'last_user': None})
        
        # Rate limiting - track when a user last used each emote
        self.user_last_used = defaultdict(lambda: defaultdict(float))  # user_id -> {emote -> timestamp}
        self.rate_limit_window = 2.0  # Only count one usage of same emote per user per 2 seconds
        
        # Cache last cleanup time to avoid too frequent cleanups
        self.last_cleanup = time.time()
        self.cleanup_interval = 1.0  # Clean up every second
        
    def add_message(self, user_id: str, emotes: Set[str], timestamp: float = None) -> Optional[List[EmoteSpamEvent]]:
        """Process a new message containing emotes. Returns spam events if detected."""
        now = timestamp or time.time()
        spam_events = []
        
        # Clean old events periodically
        if now - self.last_cleanup >= self.cleanup_interval:
            self._cleanup_old_events(now)
            self.last_cleanup = now
            
        # Process each emote in the message - only once per emote
        for emote in emotes:
            # Check rate limiting (one use of same emote per user per 2 seconds)
            last_used = self.user_last_used[user_id].get(emote, 0)
            if now - last_used < self.rate_limit_window:
                continue  # Skip this emote due to rate limiting
                
            # Update last used timestamp
            self.user_last_used[user_id][emote] = now
            
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
        
    def _check_emote_spam(self, emote: str, now: float) -> float:
        """Calculate spam score for an emote. Returns 0 if not spam, otherwise returns score."""
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
        consecutive = self.consecutive_counts[emote]['count']
        
        # Check minimum thresholds
        if (total_uses < self.min_messages or 
            unique_users < self.min_users or 
            msgs_per_sec < self.min_density):
            return 0.0
            
        # Calculate spam score components
        density_score = min(msgs_per_sec / 2.0, 1.0) * 30  # Reduced to 30 points for message density (was 40)
        user_score = min(unique_users / 5.0, 1.0) * 30     # 30 points for unique users
        consecutive_score = min(consecutive / 5.0, 1.0) * 25  # Reduced to 25 points for consecutive uses (was 30)
        
        # Combine scores with weights
        total_score = density_score + user_score + consecutive_score
        
        # Lower the threshold for a spam event
        return total_score if total_score > 40 else 0.0
        
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

class EmoteManager:
    def __init__(self):
        self.emote_dict = {}  # {emote_name: source}
        self.last_update = 0
        self.update_interval = 300  # Update every 5 minutes
        self.logger = logging.getLogger(__name__)
        
    async def update_emotes(self, channel_id: str):
        """Update emote dictionary from various sources"""
        try:
            current_time = time.time()
            if current_time - self.last_update < self.update_interval:
                return
                
            # Get Twitch credentials from env
            client_id = os.getenv('TWITCH_CLIENT_ID')
            client_secret = os.getenv('TWITCH_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                self.logger.error("Missing Twitch API credentials")
                return
                
            # Clear existing emotes before updating
            old_emote_count = len(self.emote_dict)
            self.emote_dict.clear()
            self.logger.info(f"Cleared {old_emote_count} existing emotes")
            
            async with aiohttp.ClientSession() as session:
                # Get Twitch OAuth token
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
                        self.logger.info("Successfully obtained Twitch API token")
                    else:
                        self.logger.error(f"Failed to get Twitch token: {resp.status}")
                        return
                
                headers = {
                    "Client-ID": client_id,
                    "Authorization": f"Bearer {access_token}"
                }
                
                # Get Global Twitch Emotes
                self.logger.info("Fetching Twitch global emotes...")
                async with session.get(
                    "https://api.twitch.tv/helix/chat/emotes/global",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        emotes_data = data.get("data", [])
                        
                        twitch_global_emotes = []
                        for emote in emotes_data:
                            name = emote.get("name", "").strip()
                            emote_id = emote.get("id", "")
                            if name and len(name) >= 2:  # Basic validation
                                self.emote_dict[name] = "twitch:global"
                                twitch_global_emotes.append(name)
                        
                        # Log sample of emotes for debugging
                        sample_size = min(20, len(twitch_global_emotes))
                        self.logger.info(f"Loaded {len(emotes_data)} Twitch global emotes")
                        if twitch_global_emotes:
                            self.logger.debug(f"Sample Twitch global emotes: {twitch_global_emotes[:sample_size]}")
                    else:
                        self.logger.error(f"Failed to get global emotes: {resp.status}")
                
                # Get Channel Twitch Emotes
                if channel_id:
                    self.logger.info(f"Fetching Twitch channel emotes for {channel_id}...")
                    async with session.get(
                        f"https://api.twitch.tv/helix/chat/emotes?broadcaster_id={channel_id}",
                        headers=headers
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            emotes_data = data.get("data", [])
                            
                            twitch_channel_emotes = []
                            for emote in emotes_data:
                                name = emote.get("name", "").strip()
                                emote_id = emote.get("id", "")
                                emote_type = emote.get("emote_type", "")
                                
                                if name and len(name) >= 2:
                                    self.emote_dict[name] = f"twitch:channel:{channel_id}"
                                    twitch_channel_emotes.append(f"{name} ({emote_type})")
                            
                            self.logger.info(f"Loaded {len(emotes_data)} Twitch channel emotes")
                            if twitch_channel_emotes:
                                self.logger.debug(f"Twitch channel emotes: {twitch_channel_emotes}")
                        else:
                            self.logger.error(f"Failed to get channel emotes: {resp.status}, response: {await resp.text()}")
                
                # Get 7TV Global Emotes
                self.logger.info("Fetching 7TV global emotes...")
                async with session.get("https://7tv.io/v3/emote-sets/global") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        added_count = 0
                        for emote in data.get("emotes", []):
                            name = emote.get("name", "").strip()
                            if name and len(name) >= 2:
                                self.emote_dict[name] = "7tv:global"
                                added_count += 1
                        self.logger.info(f"Loaded {added_count} 7TV global emotes")
                    else:
                        self.logger.error(f"Failed to get 7TV global emotes: {resp.status}")
                
                # Get 7TV Channel Emotes
                if channel_id:
                    self.logger.info(f"Fetching 7TV channel emotes...")
                    
                    # First get the 7TV user connection
                    stv_user_id = await self._get_7tv_user_connection(session, channel_id)
                    
                    if stv_user_id:
                        # Then get their active emote set
                        emote_set = await self._get_7tv_emote_set(session, stv_user_id)
                        
                        if emote_set:
                            added_count = 0
                            stv_channel_emotes = []
                            for emote in emote_set.get("emotes", []):
                                name = emote.get("name", "").strip()
                                if name and len(name) >= 2:
                                    self.emote_dict[name] = f"7tv:channel:{channel_id}"
                                    stv_channel_emotes.append(name)
                                    added_count += 1
                                    
                            self.logger.info(f"Loaded {added_count} 7TV channel emotes")
                            if added_count > 0:
                                self.logger.debug(f"7TV channel emotes: {stv_channel_emotes}")
                        else:
                            self.logger.warning(f"No 7TV emote set found for channel {channel_id}")
                    else:
                        self.logger.warning(f"No 7TV user connection found for channel {channel_id}")
            
            # Log final emote statistics
            self.last_update = current_time
            stats = self._get_emote_source_stats()
            self.logger.info(f"Emote loading complete. Sources: {stats}")
            
            # Log a sample of emotes by source for debugging
            twitch_global = [name for name, src in self.emote_dict.items() if src == "twitch:global"]
            twitch_channel = [name for name, src in self.emote_dict.items() if "twitch:channel" in src]
            stv_global = [name for name, src in self.emote_dict.items() if src == "7tv:global"]
            stv_channel = [name for name, src in self.emote_dict.items() if "7tv:channel" in src]
            
            self.logger.debug(f"Twitch global emotes count: {len(twitch_global)}")
            self.logger.debug(f"Twitch channel emotes count: {len(twitch_channel)}")
            self.logger.debug(f"7TV global emotes count: {len(stv_global)}")
            self.logger.debug(f"7TV channel emotes count: {len(stv_channel)}")
            
        except Exception as e:
            self.logger.error(f"Error updating emotes: {e}", exc_info=True)
            
    def _get_emote_source_stats(self) -> str:
        """Get statistics about emote sources."""
        stats = {}
        for source in self.emote_dict.values():
            stats[source] = stats.get(source, 0) + 1
        return ", ".join(f"{k}: {v}" for k, v in sorted(stats.items()))
        
    def get_emotes(self) -> Set[str]:
        """Get the current set of all known emotes."""
        return set(self.emote_dict.keys())
        
    def get_channel_emotes(self, channel_id: str) -> Set[str]:
        """Get emotes specific to a channel."""
        return {
            emote for emote, source in self.emote_dict.items()
            if f"channel:{channel_id}" in source
        }

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

class ChatAnalyzerML:
    def __init__(self, window_size: int = 30, channel: str = None):
        """
        Initialize the ML-enhanced chat analyzer with unsupervised learning.
        
        Args:
            window_size (int): Size of the sliding window in seconds
            channel (str): Channel name
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize emotes dictionary
        self.emotes = {}
        
        # Initialize base emote set
        self.base_hype_emotes = set()
        self.channel_emotes = set()
        
        self.window_size = window_size
        self.channel = channel
        self.messages = deque(maxlen=1000)  # Store last 1000 messages
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.viewer_count = 0
        
        # Message counting (like IRC example)
        self.msg_count = 0
        self.last_reset = time.time()
        self.current_rate = 0
        
        # Message window for burst detection
        self.message_window = []
        
        # Start the reset timer in a separate thread
        self.running = True
        self.reset_thread = threading.Thread(target=self._reset_counter, daemon=True)
        self.reset_thread.start()
        
        # Baseline velocity tracking
        self.baseline_velocities = deque(maxlen=600)  # 10 minutes of baseline data
        self.last_baseline_update = datetime.now()
        self.baseline_update_interval = 1  # Update baseline every second
        
        # Emote tracking
        self.emote_window = deque(maxlen=1000)  # Store recent emote usage
        self.emote_window_size = 8  # Look at last 8 seconds for bursts
        
        # ML components
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.03,  # Expect about 3% of moments to be clip-worthy
            random_state=42,
            n_estimators=100,    # More trees for better accuracy
            max_samples='auto',
            bootstrap=True       # Enable bootstrapping for better diversity
        )
        
        # Feature history for training
        self.feature_history = deque(maxlen=1000)  # Store last 1000 feature vectors
        
        # Tracking components
        self.user_emote_usage = {}  # {user_id: {emote: count}}
        self.emote_frequency = Counter()
        self.recent_emotes = deque(maxlen=100)
        
        # Rolling statistics
        self.velocity_window = 300  # 5 minutes of velocity history
        self.velocity_history = deque(maxlen=self.velocity_window)
        self.last_velocity_update = datetime.now()
        self.velocity_update_interval = 1  # Update velocity every second
        self.sentiment_history = deque(maxlen=60)
        self.burst_history = deque(maxlen=60)
        
        # Emote tracking
        self.emote_pattern = re.compile(r'[A-Z][A-Za-z]*(?:[A-Z][A-Za-z]*)*')  # Basic emote pattern
        
        # Initialize emote tracking
        self.emote_manager = EmoteManager()
        
        # Load base emotes
        self.load_twitch_emotes()
        self.logger.debug(f"Loaded {len(self.base_hype_emotes)} known emotes")
        
        # Add emote spam detector
        self.emote_spam_detector = EmoteSpamDetector(
            window_size=8.0,  # 8 second window for spam detection
            min_messages=5,   # At least 5 messages
            min_users=4,      # From at least 4 users
            min_density=0.5   # At least 1 message every 2 seconds
        )
        
        # Create model directory if needed
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Load ML model if exists
        self.load_ml_model()
        
    @property
    def hype_emotes(self) -> set:
        """Get combined set of base and channel-specific emotes."""
        return self.base_hype_emotes
        
    def extract_features(self) -> np.ndarray:
        """Extract feature vector from current window."""
        try:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_size)
            
            # Get messages in current window
            window_messages = [
                msg for msg in self.messages
                if msg['timestamp'] > window_start
            ]
            
            if not window_messages:
                return np.zeros(15)  # Return zero vector if no messages
                
            # Basic chat metrics with safeguards
            try:
                velocity = self.get_chat_velocity()
                norm_velocity = self.get_normalized_chat_velocity()
                z_score = self.get_velocity_zscore()  # Use Z-score for velocity
            except Exception:
                velocity = 0.0
                norm_velocity = 0.0
                z_score = 0.0
                
            try:
                sentiment = self.get_average_sentiment()
            except Exception:
                sentiment = 0.0
                
            try:
                burst_score = self.get_emote_burst_score()
            except Exception:
                burst_score = 0.0
            
            # Message content analysis
            message_texts = [msg['text'].lower() for msg in window_messages]
            unique_messages = len(set(message_texts))
            total_messages = len(message_texts)
            
            # Calculate entropy (message diversity) with safeguard
            message_entropy = unique_messages / max(total_messages, 1)
            
            # Count laughter and clips with safeguards
            try:
                laughter_count = sum(
                    1 for text in message_texts
                    if any(re.search(pattern, text) for pattern in self.laughter_patterns)
                )
            except Exception:
                laughter_count = 0
                
            try:
                clip_urls = sum(1 for text in message_texts if 'clips.twitch.tv' in text)
            except Exception:
                clip_urls = 0
            
            # Emote density with safeguards
            try:
                emote_messages = sum(1 for text in message_texts if any(emote in text for emote in self.hype_emotes))
                emote_density = emote_messages / max(total_messages, 1)
            except Exception:
                emote_density = 0.0
            
            # User engagement
            try:
                unique_users = len(set(msg['user_id'] for msg in window_messages))
                user_ratio = unique_users / max(total_messages, 1)
            except Exception:
                unique_users = 0
                user_ratio = 0.0
            
            # Viewer-scaled metrics
            viewer_count = max(self.viewer_count, 1)
            
            # Calculate scale factors based on viewer count
            if viewer_count < 1000:
                # Small channels (0.25 to 0.8 scale factor)
                velocity_scale = 0.25 + (0.55 * (viewer_count / 1000))
                burst_scale = 0.35 + (0.45 * (viewer_count / 1000))
            elif viewer_count < 10000:
                # Medium channels (0.8 to 1.2 scale factor)
                velocity_scale = 0.8 + (0.4 * ((viewer_count - 1000) / 9000))
                burst_scale = 0.8 + (0.4 * ((viewer_count - 1000) / 9000))
            else:
                # Large channels (1.2 to 2.0 scale factor)
                velocity_scale = 1.2 + min(0.8, 0.8 * ((viewer_count - 10000) / 40000))
                burst_scale = 1.2 + min(0.6, 0.6 * ((viewer_count - 10000) / 40000))
                
            # Calculate normalized metrics (relative to channel size)
            velocity_relative = norm_velocity / max(velocity_scale, 0.1)
            burst_relative = burst_score / max(burst_scale * 20.0, 1.0)
            
            # Combine features with bounds checking
            features = np.array([
                min(max(z_score, 0.0), 10.0),  # Z-score capped at 10
                min(max(velocity, 0.0), 1000.0),  # Raw velocity 
                min(max(norm_velocity, 0.0), 10.0),  # Normalized velocity
                min(max(velocity_relative, 0.0), 10.0),  # Velocity relative to channel size
                min(max(sentiment, -1.0), 1.0),  # Sentiment between -1 and 1
                min(max(burst_score, 0.0), 100.0),  # Burst score
                min(max(burst_relative, 0.0), 5.0),  # Burst score relative to channel size
                min(max(message_entropy, 0.0), 1.0),  # Entropy between 0 and 1
                min(max(laughter_count / max(total_messages, 1), 0.0), 1.0),
                min(clip_urls, 100),  # Cap clip URLs
                min(max(emote_density, 0.0), 1.0),  # Density between 0 and 1
                min(max(user_ratio, 0.0), 1.0),  # Ratio between 0 and 1
                min(unique_messages, 1000),  # Cap unique messages
                min(total_messages, 1000),  # Cap total messages
                np.log10(min(max(viewer_count, 1), 1000000))  # Log-scaled viewer count
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}", exc_info=True)
            return np.zeros(15)  # Return zero vector on error
        
    def update_model(self):
        """Update the isolation forest model with new data."""
        if len(self.feature_history) < 100:  # Wait for enough data
            return
            
        try:
            # Convert feature history to numpy array
            X = np.array(list(self.feature_history))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Configure isolation forest with better parameters for anomaly detection
            self.isolation_forest = IsolationForest(
                contamination=0.03,  # Expect about 3% of moments to be clip-worthy
                random_state=42,
                n_estimators=100,    # More trees for better accuracy
                max_samples='auto',
                bootstrap=True       # Enable bootstrapping for better diversity
            )
            
            # Fit isolation forest
            self.isolation_forest.fit(X_scaled)
            
            # Save model to disk
            self.save_ml_model()
            
            # Log success
            self.logger.info(f"ML model updated with {len(self.feature_history)} samples")
        except Exception as e:
            self.logger.error(f"Error updating ML model: {e}", exc_info=True)
            
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
            
            # Need enough data for meaningful detection
            if len(self.feature_history) < 100 or len(self.baseline_velocities) < 60:
                # Fall back to rule-based approach if not enough data
                velocity_relative = features[3]  # Velocity relative to channel size
                burst_relative = features[6]     # Burst score relative to channel size
                
                # Simple rule-based approach until we have enough data
                rule_score = 0.0
                if velocity_relative > 3.0:
                    rule_score += min(velocity_relative / 5.0, 0.7)
                if burst_relative > 1.5:
                    rule_score += min(burst_relative / 3.0, 0.6)
                    
                return rule_score > 0.5, rule_score
                
            # Update model periodically
            if len(self.feature_history) % 100 == 0:
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
            velocity_relative = features[3]
            burst_relative = features[6]
            
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
                
            threshold = 0.93  # Slightly lower threshold for ML-based clip detection
            return prob_score > threshold, prob_score
            
        except Exception as e:
            logger.error(f"Error in clip worthiness check: {e}", exc_info=True)
            return False, 0.0
        
    def update_viewer_count(self, count: int) -> None:
        """Update the current viewer count."""
        self.viewer_count = max(1, count)  # Ensure non-zero
        
    def _reset_counter(self):
        """Reset message counter every second in a separate thread."""
        while self.running:
            time.sleep(1.0)
            self.current_rate = self.msg_count
            self.msg_count = 0
            self.last_reset = time.time()

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
            
        # Process message
        msg_data = {
            'text': message,
            'user_id': user_id,
            'timestamp': timestamp,
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
        """Calculate average sentiment in current window."""
        if not self.messages:
            return 0.0
            
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_size)
        
        try:
            # Get sentiment analyzer if needed
            if not hasattr(self, 'sentiment_analyzer'):
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Process messages without sentiment
            window_messages = [
                msg for msg in self.messages
                if msg['timestamp'] > window_start
            ]
            
            # Calculate sentiment for messages that don't have it
            for msg in window_messages:
                if 'sentiment' not in msg:
                    text = msg.get('text', '')
                    try:
                        msg['sentiment'] = self.sentiment_analyzer.polarity_scores(text)
                    except Exception:
                        msg['sentiment'] = {'compound': 0.0}
            
            # Get sentiments in window
            sentiments = [
                msg['sentiment']['compound']
                for msg in window_messages
                if 'sentiment' in msg
            ]
            
            return np.mean(sentiments) if sentiments else 0.0
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}", exc_info=True)
            return 0.0
        
    def get_emote_burst_score(self) -> float:
        """
        Calculate emote burst score based on:
        1. Consecutive usage of the same emotes by different users
        2. Number of emotes relative to viewer count
        3. Time density of emote usage
        4. Variety vs repetition of emotes
        5. Recent spam events
        """
        if not self.emote_window:
            return 0.0
        
        now = datetime.now()
        window_start = now - timedelta(seconds=self.emote_window_size)
        
        # Get recent emote usage including spam events
        recent_entries = [e for e in self.emote_window if e['timestamp'] > window_start]
        
        if not recent_entries:
            return 0.0
        
        # Organize emote usage by emote and user for better tracking
        emote_usage = defaultdict(lambda: defaultdict(list))  # emote -> {user_id -> [(timestamp)]}
        
        # Track emotes and their users
        for entry in recent_entries:
            user_id = entry['user_id']
            timestamp = entry['timestamp']
            for emote in entry['emotes']:
                emote_usage[emote][user_id].append(timestamp)
        
        # Calculate base components
        unique_users_total = len(set(e['user_id'] for e in recent_entries))
        
        # Factor in spam events
        spam_bonus = 0.0
        spam_events = [e['spam_events'] for e in recent_entries if e.get('spam_events')]
        if spam_events:
            # Flatten spam events list
            all_events = [event for events in spam_events if events for event in events]
            if all_events:
                # Get highest spam score
                max_spam_score = max(event.score for event in all_events)
                spam_bonus = min(max_spam_score / 2.0, 30.0)  # Up to 30 points from spam
        
        # Calculate metrics for all emotes (not just the best one)
        emote_scores = []
        emote_info = []
        total_users_involved = set()
        
        # Get top emotes by user count (up to top 3)
        sorted_emotes = sorted(
            [(emote, len(users)) for emote, users in emote_usage.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Process top 3 emotes (or fewer if less available)
        top_emotes = sorted_emotes[:3] if len(sorted_emotes) >= 3 else sorted_emotes
        
        # Find the overall max consecutive count across all emotes
        max_consecutive_overall = 0
        
        for emote, user_count in top_emotes:
            users = emote_usage[emote]
            total_users_involved.update(users.keys())
            
            # Check for consecutive usage patterns
            user_timestamps = [(user, min(timestamps)) for user, timestamps in users.items()]
            user_timestamps.sort(key=lambda x: x[1])  # Sort by timestamp
            
            # Count consecutive different users using the same emote
            current_consecutive = 1
            for i in range(1, len(user_timestamps)):
                time_diff = (user_timestamps[i][1] - user_timestamps[i-1][1]).total_seconds()
                if time_diff < 4.0:  # Consecutive if within 4 seconds
                    current_consecutive += 1
                else:
                    # Reset consecutive count on time gap
                    current_consecutive = 1
                    
            max_consecutive_overall = max(max_consecutive_overall, current_consecutive)
            
            # Calculate time density for this emote
            timestamps = [ts for user_timestamps in users.values() for ts in user_timestamps]
            if len(timestamps) >= 2:
                time_span = (max(timestamps) - min(timestamps)).total_seconds()
                time_span = max(time_span, 0.1)  # Avoid division by zero
                emotes_per_second = len(timestamps) / time_span
                
                # Calculate score for this emote
                emote_score = {
                    'emote': emote,
                    'user_count': len(users),
                    'user_ratio': len(users) / max(unique_users_total, 1),
                    'consecutives': current_consecutive,
                    'density': emotes_per_second,
                }
                emote_scores.append(emote_score)
                
                emote_info.append(f"{emote} (used by {len(users)} users)")
        
        # If no scores calculated, return 0
        if not emote_scores:
            return 0.0
            
        # Calculate metrics across all top emotes
        consecutive_score = min(max_consecutive_overall * 3.0, 50.0)  # Up to 50 points for consecutive uses
        
        # Calculate the average density score from top emotes (weighted by user count)
        total_weight = sum(score['user_count'] for score in emote_scores)
        density_score = 0.0
        
        if total_weight > 0:
            for score in emote_scores:
                weight = score['user_count'] / total_weight
                emote_density = min(score['density'] * 3.0, 30.0)  # Cap at 30 points per emote
                density_score += emote_density * weight
                
        # Calculate user involvement ratio across all top emotes
        user_ratio = min(len(total_users_involved) / max(unique_users_total, 1), 1.0)
        user_score = user_ratio * 20.0  # Up to 20 points for user ratio
        
        # Add a bonus for multiple popular emotes (up to 15 points)
        multi_emote_bonus = min((len(emote_scores) - 1) * 7.5, 15.0)
        
        # Factor in viewer participation
        viewer_count = max(self.viewer_count, 1)
        viewer_participation = min(unique_users_total / (viewer_count * 0.01), 1.0)
        
        # Combine all components with viewer participation factor
        base_score = (
            consecutive_score +
            density_score +
            user_score +
            multi_emote_bonus +
            spam_bonus
        ) * (0.5 + viewer_participation * 0.5)
        
        # Log components for debugging
        self.logger.debug(
            f"Emote burst components:"
            f"\n  - Consecutive ({max_consecutive_overall}): {consecutive_score:.1f}/50"
            f"\n  - Density: {density_score:.1f}/30"
            f"\n  - User ratio ({user_ratio:.2f}): {user_score:.1f}/20"
            f"\n  - Multi-emote bonus: {multi_emote_bonus:.1f}/15"
            f"\n  - Spam bonus: {spam_bonus:.1f}/30"
            f"\n  - Viewer participation ({viewer_participation:.3f})"
            f"\n  - Total unique users: {unique_users_total}"
            f"\n  - Top emotes: {', '.join(emote_info)}"
        )
        
        # Cap and return final score
        final_score = min(base_score, 100.0)
        
        if final_score > 20:
            self.logger.info(f"High emote burst score: {final_score:.1f}")
        
        return final_score
        
    def get_window_stats(self) -> Dict:
        """Get enhanced window statistics with ML scores."""
        try:
            # Get base stats
            raw_velocity = self.get_chat_velocity()
            norm_velocity = self.get_normalized_chat_velocity()
            
            # Calculate emote burst score directly
            burst_score = self.get_emote_burst_score()
            
            # Get viewer count with minimum of 1
            viewer_count = max(self.viewer_count, 1)
            
            # Calculate dynamic thresholds based on viewer count
            viewer_count = max(self.viewer_count, 1)
            
            # Base thresholds for normal-sized streams (~5000 viewers)
            base_velocity_threshold = 2.5  # Set back to a reasonable value
            base_burst_threshold = 20.0    # Set back to a reasonable value
            
            # More pronounced scaling: stronger effect for small and large channels
            if viewer_count < 1000:
                # Small channels get much lower thresholds (0.25 to 0.8 scale factor)
                velocity_scale = 0.25 + (0.55 * (viewer_count / 1000))
                burst_scale = 0.35 + (0.45 * (viewer_count / 1000))
            elif viewer_count < 10000:
                # Medium channels get moderate thresholds (0.8 to 1.2 scale factor)
                velocity_scale = 0.8 + (0.4 * ((viewer_count - 1000) / 9000))
                burst_scale = 0.8 + (0.4 * ((viewer_count - 1000) / 9000))
            else:
                # Large channels get much higher thresholds (1.2 to 2.0 scale factor)
                velocity_scale = 1.2 + min(0.8, 0.8 * ((viewer_count - 10000) / 40000))
                burst_scale = 1.2 + min(0.6, 0.6 * ((viewer_count - 10000) / 40000))
            
            # Apply scaling
            velocity_threshold = base_velocity_threshold * velocity_scale
            burst_threshold = base_burst_threshold * burst_scale
            
            # Calculate relative metrics
            velocity_relative = norm_velocity / max(velocity_scale, 0.1)
            burst_relative = burst_score / max(burst_scale * 15.0, 1.0)
            
            # Calculate rule-based score (0-1 scale)
            rule_score = 0.0
            
            # Factor in velocity using scaled threshold
            if norm_velocity > velocity_threshold * 0.15:  # Lower initial trigger for gradual scoring
                rule_score += min((norm_velocity - (velocity_threshold * 0.15)) / velocity_threshold, 0.6)
                
            # Factor in burst score using scaled threshold
            if burst_score > burst_threshold * 0.1:  # Lower initial trigger for gradual scoring
                rule_score += min(burst_score / (burst_threshold * 1.5), 0.6)
                
            # Get ML prediction if we have enough data
            if len(self.feature_history) >= 100 and len(self.baseline_velocities) >= 60:
                is_worthy, ml_score = self.is_clip_worthy()
            else:
                ml_score = 0.0
                if velocity_relative > 3.0:
                    ml_score += min(velocity_relative / 5.0, 0.7)
                if burst_relative > 1.5:
                    ml_score += min(burst_relative / 3.0, 0.6)
            
            # Hybrid score - combine rule-based and ML-based approaches
            hybrid_score = 0.0
            
            # If ML score is high, prioritize it
            if ml_score > 0.8:
                hybrid_score = ml_score * 0.8 + rule_score * 0.2
            # If rule score is high, prioritize it
            elif rule_score > 0.7:
                hybrid_score = rule_score * 0.7 + ml_score * 0.3
            # Otherwise use a balanced approach
            else:
                hybrid_score = rule_score * 0.5 + ml_score * 0.5
            
            # Ensure final score is between 0 and 1
            clip_worthy_score = min(max(hybrid_score, 0.0), 1.0)
            
            # Add sentiment calculation
            try:
                sentiment = self.get_average_sentiment()
            except Exception as e:
                self.logger.debug(f"Error calculating sentiment: {e}")
                sentiment = 0.0
            
            stats = {
                'viewer_count': self.viewer_count,
                'raw_velocity': raw_velocity,
                'norm_velocity': norm_velocity,
                'velocity_relative': velocity_relative,
                'burst_score': burst_score,
                'burst_relative': burst_relative,
                'rule_score': rule_score,
                'ml_score': ml_score,
                'clip_worthy_score': clip_worthy_score,
                'velocity_threshold': velocity_threshold,
                'burst_threshold': burst_threshold,
                'sentiment': sentiment
            }
            
            # Debug logging for non-zero scores
            if clip_worthy_score > 0.5:
                logger.info(f"High clip worthiness: {clip_worthy_score:.3f} (rule: {rule_score:.2f}, ML: {ml_score:.2f}) for {viewer_count} viewers")
            elif burst_score > 0:
                logger.debug(f"Non-zero burst score: {burst_score:.1f}")
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting window stats: {e}", exc_info=True)
            return {
                'viewer_count': self.viewer_count,
                'raw_velocity': 0.0,
                'norm_velocity': 0.0,
                'burst_score': 0.0,
                'clip_worthy_score': 0.0
            }
        
    def _calculate_burst_score(self) -> float:
        """Calculate a burst score based on emote usage patterns."""
        # Just delegate to get_emote_burst_score which is our primary calculation now
        return self.get_emote_burst_score()
    
    def _extract_emotes(self, message: str) -> Set[str]:
        """Extract known emotes from a message."""
        # Split message into words and remove duplicates
        words = set(message.split())
        
        # Check against known emotes from APIs
        emotes = words.intersection(self.base_hype_emotes)
        
        if emotes:
            # Categorize emotes by source for better debugging
            twitch_global = [emote for emote in emotes if self.emotes.get(emote) == "twitch:global"]
            twitch_channel = [emote for emote in emotes if self.emotes.get(emote, "").startswith("twitch:channel")]
            stv_global = [emote for emote in emotes if self.emotes.get(emote) == "7tv:global"]
            stv_channel = [emote for emote in emotes if self.emotes.get(emote, "").startswith("7tv:channel")]
            
            # Log emotes by source
            if twitch_global:
                self.logger.debug(f"Found Twitch global emotes: {twitch_global}")
            if twitch_channel:
                self.logger.debug(f"Found Twitch channel emotes: {twitch_channel}")
            if stv_global:
                self.logger.debug(f"Found 7TV global emotes: {stv_global}")
            if stv_channel:
                self.logger.debug(f"Found 7TV channel emotes: {stv_channel}")
            
            # Log overall emote summary
            self.logger.debug(f"Extracted emotes: {emotes}")
            
        return emotes
    
    def update_channel_emotes(self, emotes: Set[str]) -> None:
        """
        Update the set of channel-specific emotes.
        This can be called when receiving emote data from Twitch's API.
        """
        self.channel_emotes = emotes
        
        # Update emote frequency with known emotes
        self.emote_frequency.update(emotes)
        
    def learn_new_emotes(self, min_frequency: int = 5) -> None:
        """
        Learn new emotes based on usage patterns.
        This helps identify channel-specific emotes without explicit API data.
        """
        # Find frequently used potential emotes
        for word, count in self.emote_frequency.items():
            if count >= min_frequency and word not in self.base_hype_emotes:
                # Check if it's a likely emote:
                # 1. All caps (like LOL, OOOO)
                # 2. CamelCase (like PogChamp)
                # 3. At least 3 characters
                if len(word) >= 3 and (
                    word.isupper() or
                    (word[0].isupper() and any(c.islower() for c in word[1:]))
                ):
                    self.logger.info(f"Learning new emote: {word} (used {count} times)")
                    self.base_hype_emotes.add(word)
                    if word not in self.channel_emotes:
                        self.channel_emotes.add(word)
            
    def get_current_emote_stats(self) -> Dict[str, int]:
        """Get emote usage statistics for the current window."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_size)
        
        # Count emotes in current window
        window_emotes = Counter()
        for entry in self.emote_window:
            if entry['timestamp'] > window_start:
                window_emotes.update(entry['emotes'])
                
        return dict(window_emotes)
        
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.running = False
        if hasattr(self, 'reset_thread'):
            self.reset_thread.join(timeout=1.0)
            
        # Save ML model on exit if we have enough data
        if hasattr(self, 'feature_history') and len(self.feature_history) >= 100:
            self.save_ml_model()
        
    # Include all other necessary methods from original ChatAnalyzer
    # (get_normalized_chat_velocity, etc.) 

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

    def add_channel_emotes(self, channel_emotes: Set[str]):
        """Add channel-specific emotes to the known set."""
        before_count = len(self.base_hype_emotes)
        self.base_hype_emotes.update(channel_emotes)
        added_count = len(self.base_hype_emotes) - before_count
        
        self.logger.info(f"Added {added_count} channel-specific emotes")
        if added_count > 0:
            self.logger.debug(f"New emotes added: {sorted(channel_emotes)}")

    def _clean_old_messages(self, current_time: float) -> None:
        """Remove messages older than the window size."""
        window_start = current_time - self.window_size
        self.message_window = [
            (msg, ts, user) for msg, ts, user in self.message_window 
            if ts >= window_start
        ]
        if len(self.message_window) > 0:
            self.logger.debug(
                f"Message window: {len(self.message_window)} messages, "
                f"span: {current_time - min(ts for _, ts, _ in self.message_window):.1f}s"
            ) 

    def is_clip_worthy_from_metrics(self) -> bool:
        """
        Determine if current chat activity is clip-worthy based on velocity and emote metrics.
        Scales thresholds dynamically based on viewer count.
        Returns True if clip-worthy, False otherwise.
        """
        # Get current metrics
        velocity = self.get_chat_velocity()
        normalized_velocity = self.get_normalized_chat_velocity()
        burst_score = self.get_emote_burst_score()
        
        # Calculate dynamic thresholds based on viewer count
        viewer_count = max(self.viewer_count, 1)
        
        # Base thresholds for normal-sized streams (~5000 viewers)
        base_velocity_threshold = 2.5  # Set back to a reasonable value
        base_burst_threshold = 20.0    # Set back to a reasonable value
        
        # Scale thresholds based on viewer count
        # For smaller streams (< 1000 viewers): lower thresholds to make it easier to trigger
        # For larger streams (> 10000 viewers): higher thresholds to prevent too many clips
        
        # More pronounced scaling: stronger effect for small and large channels
        if viewer_count < 1000:
            # Small channels get much lower thresholds (0.25 to 0.8 scale factor)
            velocity_scale = 0.25 + (0.55 * (viewer_count / 1000))
            burst_scale = 0.35 + (0.45 * (viewer_count / 1000))
        elif viewer_count < 10000:
            # Medium channels get moderate thresholds (0.8 to 1.2 scale factor)
            velocity_scale = 0.8 + (0.4 * ((viewer_count - 1000) / 9000))
            burst_scale = 0.8 + (0.4 * ((viewer_count - 1000) / 9000))
        else:
            # Large channels get much higher thresholds (1.2 to 2.0 scale factor)
            velocity_scale = 1.2 + min(0.8, 0.8 * ((viewer_count - 10000) / 40000))
            burst_scale = 1.2 + min(0.6, 0.6 * ((viewer_count - 10000) / 40000))
        
        # Apply scaling
        velocity_threshold = base_velocity_threshold * velocity_scale
        burst_threshold = base_burst_threshold * burst_scale
        
        # Check if either metric exceeds its threshold
        high_velocity = normalized_velocity >= velocity_threshold
        high_burst = burst_score >= burst_threshold
        
        # Log potential triggers with scale info
        if high_velocity or high_burst:
            self.logger.info(
                f"Potential clip moment detected for {viewer_count} viewers: "
                f"velocity={normalized_velocity:.2f} (threshold={velocity_threshold:.2f}, scale={velocity_scale:.2f}), "
                f"burst={burst_score:.1f} (threshold={burst_threshold:.1f}, scale={burst_scale:.2f})"
            )
        
        # Return True if either metric exceeds its threshold
        return high_velocity or high_burst 

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
            
            # Save feature history (convert deque to list for serialization)
            if len(self.feature_history) > 0:
                features_path = base_path.with_name(f"{safe_channel}_features.pkl")
                # Save only up to 200 features to keep file size reasonable
                sample_features = list(self.feature_history)[-200:]
                with open(features_path, 'wb') as f:
                    pickle.dump(sample_features, f)
                    
            # Save baseline velocities
            if len(self.baseline_velocities) > 0:
                baseline_path = base_path.with_name(f"{safe_channel}_baseline.pkl")
                sample_baseline = list(self.baseline_velocities)[-200:]
                with open(baseline_path, 'wb') as f:
                    pickle.dump(sample_baseline, f)
                    
            self.logger.info(f"ML model saved for channel {self.channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ML model: {e}", exc_info=True)
            return False
            
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
                    self.feature_history = deque(loaded_features, maxlen=1000)
                    
            # Load baseline velocities if available
            if baseline_path.exists():
                with open(baseline_path, 'rb') as f:
                    loaded_baseline = pickle.load(f)
                    # Convert list back to deque
                    self.baseline_velocities = deque(loaded_baseline, maxlen=600)
                    
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
                contamination=0.03,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                bootstrap=True
            )
            return False 