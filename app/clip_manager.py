import os
import time
import logging
import threading
import requests
import re
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClipRequest:
    """Represents a request to create a clip."""
    channel: str
    start_time: float
    duration: int
    trigger_type: str
    metrics: Dict


class ClipManager:
    """Manages creating clips using the Twitch API."""
    
    def __init__(self, output_dir: str = "clips"):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Store clip URLs and timing
        self.clip_urls = []
        self.last_clip_time = 0

        # Get configuration from environment
        self.output_dir = output_dir
        self.clips_file = os.path.join(output_dir, "clip_urls.txt")
        self.cooldown_period = int(
            os.getenv('CLIP_COOLDOWN', '10'))  # Seconds between clips

        # Get credentials from environment
        self.client_id = os.getenv('TWITCH_CLIENT_ID', '')
        self.client_secret = os.getenv('TWITCH_CLIENT_SECRET', '')
        raw_token = os.getenv('TWITCH_ACCESS_TOKEN', '')
        
        # Clean up access token once at initialization
        self.access_token = self._clean_token(raw_token)
        
        # Create clips directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load existing clips if any
        self._load_clips()

        # Validate the credentials
        self.validate_credentials()

        self.logger.info("ClipManager initialized with Twitch API integration")
        self.logger.info(f"Clip cooldown: {self.cooldown_period} seconds")

    def _clean_token(self, token: str) -> str:
        """Clean up token by removing any prefixes."""
        if not token:
            return ''
        token = token.replace('Bearer ', '')
        token = token.replace('oauth:', '')
        return token

    def validate_credentials(self):
        """Validate the credentials and log helpful messages."""
        if not self.client_id:
            self.logger.error(
                "MISSING: TWITCH_CLIENT_ID - You must provide a Twitch Client ID")

        if not self.access_token:
            self.logger.error(
                "MISSING: TWITCH_ACCESS_TOKEN - You must provide a User OAuth token with clips:edit scope")
            self.logger.error(
                "Use https://twitchtokengenerator.com/ to generate a token with clips:edit scope")

        # Check for refresh token
        refresh_token = os.getenv('TWITCH_REFRESH_TOKEN', '')
        if not refresh_token:
            self.logger.warning("No TWITCH_REFRESH_TOKEN found in environment")
            self.logger.warning(
                "Without a refresh token, you'll need to manually update the access token when it expires")
        else:
            self.logger.info(
                "TWITCH_REFRESH_TOKEN is present (can be used for automatic token refresh)")

    def request_clip(self, channel: str, trigger_type: str = "manual", metrics: dict = None) -> bool:
        """Request a clip creation. Returns True if request was accepted."""
        try:
            self.logger.info(
                f"CLIP REQUEST RECEIVED for channel {channel} - trigger_type: {trigger_type}")
            self.logger.info(
                f"Credential status: CLIENT_ID={'✓' if self.client_id else '✗'}, ACCESS_TOKEN={'✓' if self.access_token else '✗'}")

            # Check cooldown unless in testing mode
            current_time = time.time()
            time_since_last = current_time - self.last_clip_time

            if trigger_type != "test" and time_since_last < self.cooldown_period:
                self.logger.info(
                    f"Clip request rejected: On cooldown ({time_since_last:.1f}s < {self.cooldown_period}s)")
                return False

            # Create clip via API
            clip_url = self._create_clip_via_api(channel)
            if clip_url:
                self.last_clip_time = current_time
                self.add_clip_url(clip_url)
                self.logger.info(f"Successfully created clip: {clip_url}")
                return True
            else:
                self.logger.error("Failed to create clip via API")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in request_clip: {e}", exc_info=True)
            return False
            
    def _create_clip_via_api(self, channel: str) -> Optional[str]:
        """
        Create a clip using the Twitch API and return the URL if successful.
        Note: Clip creation is asynchronous - the clip may take a few seconds to be ready.
        """
        try:
            # Check if required credentials are available
            if not self.client_id or not self.access_token:
                self.logger.error("Missing Twitch API credentials")
                self.logger.error(
                    f"CLIENT_ID: {'present' if self.client_id else 'MISSING'}")
                self.logger.error(
                    f"ACCESS_TOKEN: {'present' if self.access_token else 'MISSING'}")
                return None

            self.logger.info(
                f"Starting clip creation process for channel: {channel}")

            # First, get the broadcaster ID from the channel name
            self.logger.info(f"Fetching broadcaster ID for channel: {channel}")
            broadcaster_id = self._get_broadcaster_id(channel)

            if not broadcaster_id:
                self.logger.error(
                    f"Could not get broadcaster ID for channel: {channel}")
                return None

            # Create the clip using query parameters
            url = 'https://api.twitch.tv/helix/clips'
            headers = {
                'Client-ID': self.client_id,
                'Authorization': f'Bearer {self.access_token}'
            }
            params = {
                'broadcaster_id': broadcaster_id
            }

            self.logger.info(
                f"Creating clip for broadcaster ID: {broadcaster_id}")
            response = requests.post(url, headers=headers, params=params)

            if response.status_code != 202:
                self.logger.error(
                    f"Failed to create clip: Status {response.status_code}")
                self.logger.error(f"Response body: {response.text}")
                if response.status_code == 401:
                    self.logger.error(
                        "Authentication failed. Make sure you have a user token with clips:edit scope")
                    self.logger.error(
                        "Use https://twitchtokengenerator.com/ to generate a new token")
                return None

            clip_data = response.json()
            if not clip_data.get('data'):
                self.logger.error("No clip data returned from API")
                return None
                    
            clip_id = clip_data['data'][0]['id']
            edit_url = clip_data['data'][0].get('edit_url')
            clip_url = f"https://clips.twitch.tv/{clip_id}"

            self.logger.info(
                "Clip creation initiated, waiting for it to be ready...")

            # Wait for clip to be ready (up to 10 seconds)
            clip_ready = self._wait_for_clip(clip_id)
            if not clip_ready:
                self.logger.error("Clip creation timed out or failed")
                return None
    
            # Store both URLs
            clip_info = {
                'id': clip_id,
                'url': clip_url,
                'edit_url': edit_url,
                'created_at': time.time()
            }
            self.clip_urls.append(clip_info)
            self._save_clips()

            self.logger.info(f"Successfully created clip!")
            self.logger.info(f"View clip at: {clip_url}")
            if edit_url:
                self.logger.info(f"Edit clip at: {edit_url}")

            return clip_url
            
        except Exception as e:
            self.logger.error(f"Error creating clip: {e}", exc_info=True)
            return None

    def _get_broadcaster_id(self, channel_name: str) -> Optional[str]:
        """Get the broadcaster ID from a channel name using Twitch API."""
        try:
            # Ensure proper authentication
            if not self.client_id or not self.access_token:
                self.logger.error(
                    f"Cannot get broadcaster ID: Missing credentials for channel {channel_name}")
                return None

            # Set up headers for the API request
            headers = {
                'Client-ID': self.client_id,
                'Authorization': f'Bearer {self.access_token}'
            }
            
            # Make the API request
            url = f'https://api.twitch.tv/helix/users'
            params = {'login': channel_name}

            self.logger.info(
                f"Fetching broadcaster ID for channel '{channel_name}'")
            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                self.logger.error(
                    f"Failed to get broadcaster ID: Status {response.status_code}")
                self.logger.error(f"Response body: {response.text}")
                if response.status_code == 401:
                    self.logger.error(
                        "Authentication failed. Make sure you have a user token with user:read scope")
                return None
                
            data = response.json()
            if not data.get('data'):
                self.logger.error(
                    f"No user data found for channel: {channel_name}")
                return None
            
            broadcaster_id = data['data'][0]['id']
            self.logger.info(
                f"Found broadcaster ID for {channel_name}: {broadcaster_id}")
            return broadcaster_id
            
        except Exception as e:
            self.logger.error(
                f"Error getting broadcaster ID: {e}", exc_info=True)
            return None
            
    def _refresh_access_token(self) -> bool:
        """
        Refresh the Twitch API access token using the refresh token flow.
        Note: For clip creation, a user token with clips:edit scope is required.
        """
        try:
            if not self.client_id or not self.client_secret:
                self.logger.error(
                    "Missing client ID or client secret for token refresh")
                return False
            
            # Get refresh token from env - required for user OAuth tokens
            refresh_token = os.getenv('TWITCH_REFRESH_TOKEN', '')
            if not refresh_token:
                self.logger.error(
                    "Cannot refresh token: No refresh token found in environment")
                self.logger.error(
                    "Use https://twitchtokengenerator.com/ to generate a new token with refresh token")
                return False
                
            # Using refresh token flow to get a new user OAuth token
            url = 'https://id.twitch.tv/oauth2/token'
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token
            }

            self.logger.info("Attempting to refresh user OAuth token")
            response = requests.post(url, data=data)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = self._clean_token(
                    token_data['access_token'])

                # Check if we have a new refresh token to save
                if 'refresh_token' in token_data:
                    self.logger.info(
                        "Received new refresh token - update your .env file with:")
                    self.logger.info(
                        f"TWITCH_REFRESH_TOKEN={token_data['refresh_token']}")

                self.logger.info("Successfully refreshed user OAuth token")
                return True
            else:
                self.logger.error(
                    f"Failed to refresh token: Status {response.status_code}")
                self.logger.error(f"Response body: {response.text}")
                return False
                    
        except Exception as e:
            self.logger.error(
                f"Error refreshing access token: {e}", exc_info=True)
            return False
    
    def close(self):
        """Clean up resources."""
        self.logger.info("ClipManager closed")

    def export_clips(self):
        """Export clip URLs to a text file."""
        try:
            with open(self.clips_file, 'w') as f:
                for url in self.clip_urls:
                    f.write(f"{url}\n")
            self.logger.info(
                f"Exported {len(self.clip_urls)} clips to {self.clips_file}")
        except Exception as e:
            self.logger.error(f"Failed to export clips: {e}")

    def get_recent_clips(self, count: int = 5) -> list:
        """Get the most recent clip URLs."""
        clips = []
        for clip in self.clip_urls[-count:]:
            if isinstance(clip, dict):
                clips.append(clip['url'])
            else:
                clips.append(clip)  # Handle legacy format
        return clips

    def add_clip_url(self, url: str):
        """Add a clip URL to the list and save to file."""
        self.clip_urls.append(url)
        self._save_clips()

    def _save_clips(self):
        """Save clip URLs and metadata to file."""
        try:
            with open(self.clips_file, 'w') as f:
                for clip in self.clip_urls:
                    if isinstance(clip, dict):
                        f.write(
                            f"{clip['url']}\t{clip.get('edit_url', '')}\t{clip['id']}\n")
                    else:
                        f.write(f"{clip}\n")  # Handle legacy format
            self.logger.info(
                f"Saved {len(self.clip_urls)} clips to {self.clips_file}")
        except Exception as e:
            self.logger.error(f"Error saving clips to file: {e}")

    def _load_clips(self):
        """Load existing clip URLs and metadata from file."""
        try:
            if os.path.exists(self.clips_file):
                self.clip_urls = []
                with open(self.clips_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            self.clip_urls.append({
                                'url': parts[0],
                                'edit_url': parts[1],
                                'id': parts[2]
                            })
                        else:
                            # Handle legacy format
                            self.clip_urls.append(parts[0])
                self.logger.info(
                    f"Loaded {len(self.clip_urls)} clips from {self.clips_file}")
        except Exception as e:
            self.logger.error(f"Error loading clips from file: {e}")
            self.clip_urls = []

    def _wait_for_clip(self, clip_id: str, timeout: int = 10) -> bool:
        """
        Wait for a clip to be ready by polling the Get Clips endpoint.
        Returns True if clip is found within timeout period.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_clip_exists(clip_id):
                return True
            time.sleep(1)
        return False
            
    def _check_clip_exists(self, clip_id: str) -> bool:
        """Check if a clip exists using the Get Clips endpoint."""
        try:
            url = f'https://api.twitch.tv/helix/clips?id={clip_id}'
            headers = {
                'Client-ID': self.client_id,
                'Authorization': f'Bearer {self.access_token}'
            }

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return bool(data.get('data'))
            return False

        except Exception as e:
            self.logger.error(f"Error checking clip status: {e}")
            return False 
