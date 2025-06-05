import os
from datetime import datetime
import logging
import asyncio
from twitchio.ext import commands
from dotenv import load_dotenv
import threading
import tkinter as tk
import time
import signal
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from .chat_analyzer_ml import ChatAnalyzerML
from .metrics_display import MetricsDisplay, MultiChannelMetricsDisplay
from .clip_manager import ClipManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG', 'False').lower() == 'true' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set debug logging for chat analyzer
logging.getLogger('app.chat_analyzer_ml').setLevel(logging.DEBUG)

class ClipperBot(commands.Bot):
    def __init__(self):
        # Load environment variables
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path)
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Get channels from environment or default to test channel
        channels_str = os.getenv('TWITCH_CHANNELS', 'your_channel_name')
        # Also check if channels are specified as command line arguments
        if len(sys.argv) > 1 and '=' in sys.argv[1] and sys.argv[1].startswith('TWITCH_CHANNELS='):
            channels_str = sys.argv[1].split('=', 1)[1]
        
        channels = [channel.strip() for channel in channels_str.split(',')]
        self.logger.info(f"Monitoring channels: {channels}")
        
        # Get Twitch token
        token = os.getenv('TWITCH_ACCESS_TOKEN', '')
        
        # Get clip duration with proper default
        try:
            clip_duration = int(os.getenv('CLIP_DURATION', '60').split('#')[0].strip())
        except (ValueError, TypeError):
            self.logger.warning("Invalid CLIP_DURATION value, using default of 30 seconds")
            clip_duration = 30
            
        # Initialize clip manager
        self.clip_manager = ClipManager(output_dir="recordings")
        
        # Initialize analyzers for each channel
        self.analyzers = {}
        for channel in channels:
            self.analyzers[channel] = ChatAnalyzerML(
                window_size=clip_duration,
                channel=channel
            )
            
        # Initialize Bot with credentials
        super().__init__(
            token=token,
            prefix='!',
            initial_channels=channels
        )
        
        # Clip cooldown mechanism - fixed value
        self.cooldown_period = 10  # Cooldown of 10 seconds between clips
        self.clip_cooldowns = {}  # Store last clip time per channel
        
        # Connection monitoring
        self.last_message_time = time.time()
        self.connection_check_interval = 60  # Check connection every minute
        self.max_message_gap = 300  # Reconnect if no messages for 5 minutes
        
        # Initialize multi-channel metrics display
        self.metrics_display = MultiChannelMetricsDisplay()
        # Pre-initialize displays for the channels
        self.metrics_display.initialize_displays(channels)
        
        # Set running flag
        self.running = True
        
        # Pending clips queue for delayed clip creation
        self.pending_clips = []
        self.clip_delay = 7.0  # Delay clips by 7 seconds
        self.clip_scheduler_task = None
        
        # Start metrics display thread
        self.metrics_thread = threading.Thread(target=self.run_metrics_display, daemon=True)
        self.metrics_thread.start()
        
        # Load clip duration from environment if available
        if os.getenv('CLIP_DURATION'):
            try:
                self.clip_manager.clip_duration = int(os.getenv('CLIP_DURATION'))
                logger.info(f"Set clip duration to {self.clip_manager.clip_duration} seconds")
            except (ValueError, TypeError):
                logger.warning("Invalid CLIP_DURATION value")
        
        self.clip_cooldown = {}  # Prevent too frequent clips
        self.viewer_update_task = None
        self.reconnect_task = None
        self.check_connection_interval = 30  # Check connection every 30 seconds
        
        self.logger.info(f"Bot initialized, monitoring channels: {', '.join(channels)}")
        
    async def close(self):
        """Cleanup resources"""
        logger.info("Cleaning up bot resources...")
        self.running = False
        
        # Cancel running tasks
        if self.viewer_update_task:
            self.viewer_update_task.cancel()
            try:
                await self.viewer_update_task
            except asyncio.CancelledError:
                pass
                
        if self.reconnect_task:
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass
                
        if self.clip_scheduler_task:
            self.clip_scheduler_task.cancel()
            try:
                await self.clip_scheduler_task
            except asyncio.CancelledError:
                pass
                
        if hasattr(self, 'ml_saver_task') and self.ml_saver_task:
            self.ml_saver_task.cancel()
            try:
                await self.ml_saver_task
            except asyncio.CancelledError:
                pass
                
        # Save ML models one final time
        for channel, analyzer in self.analyzers.items():
            if len(analyzer.feature_history) >= 100:
                analyzer.save_ml_model()
                
        # Close clip manager
        if hasattr(self, 'clip_manager'):
            self.clip_manager.close()
                
        # Close websocket connection
        try:
            await super().close()
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")
            
        # Close metrics display
        if self.metrics_display:
            self.metrics_display.close()
            
    async def check_connection(self):
        """Periodically check if we're still receiving messages and reconnect if needed."""
        while self.running:
            try:
                await asyncio.sleep(self.check_connection_interval)
                
                # If no messages received in last minute, reconnect
                if time.time() - self.last_message_time > 60:
                    logger.warning("No messages received in 60 seconds, reconnecting...")
                    await self.close()
                    await self.connect()
                    
            except Exception as e:
                logger.error(f"Error in connection check: {e}", exc_info=True)
                
            if not self.running:
                break
                
    async def _update_viewer_counts(self):
        """Periodically update viewer counts for monitored channels."""
        retry_delay = 5  # Initial retry delay
        max_retry_delay = 30  # Maximum retry delay
        
        while self.running:
            try:
                # Get all streams in one API call
                channels = list(self.analyzers.keys())
                
                try:
                    # Directly fetch streams by login name
                    streams = await self.fetch_streams(user_logins=channels)
                    
                    # Process streams and update viewer counts
                    current_time = time.time()
                    for channel in channels:
                        if not self.running:
                            return
                            
                        # Find matching stream
                        stream = next((s for s in streams if s.user.name.lower() == channel), None)
                        
                        if stream:
                            viewer_count = stream.viewer_count
                            self.analyzers[channel].update_viewer_count(viewer_count)
                            logger.info(f"Updated viewer count for {channel}: {viewer_count}")
                        else:
                            # If channel not found, set to 0 but don't log error
                            self.analyzers[channel].update_viewer_count(0)
                            
                        # Force metrics update even if no new messages
                        if self.running:
                            stats = self.analyzers[channel].get_window_stats()
                        if hasattr(self, 'metrics_display') and self.metrics_display.ready.is_set():
                            self.metrics_display.update_metrics(channel, stats, None)
                    
                    # Reset retry delay on success
                    retry_delay = 5
                    
                except Exception as e:
                    if not self.running:
                        return
                    logger.error(f"Error fetching streams: {str(e)}", exc_info=True)
                    # Exponential backoff for retries
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                    await asyncio.sleep(retry_delay)
                    continue
                    
            except Exception as e:
                if not self.running:
                    return
                logger.error(f"Failed to update viewer counts: {str(e)}")
                # Exponential backoff for retries
                retry_delay = min(retry_delay * 2, max_retry_delay)
                await asyncio.sleep(retry_delay)
                continue
            
            # Update every 15 seconds
            await asyncio.sleep(15)
            
    async def event_ready(self):
        """Called once when the bot goes online."""
        self.logger.info(f"Bot is ready! Username: {self.nick}")
        
        # Initialize emotes for each channel
        for channel, analyzer in self.analyzers.items():
            try:
                # Get channel ID from username
                users = await self.fetch_users(names=[channel])
                if users:
                    channel_id = users[0].id
                    await analyzer.update_emotes(channel_id)
                    logger.info(f"Initialized emotes for channel {channel}")
            except Exception as e:
                logger.error(f"Failed to initialize emotes for {channel}: {e}")
        
        # Start viewer count update loop
        if self.viewer_update_task:
            self.viewer_update_task.cancel()
        self.viewer_update_task = asyncio.create_task(self._update_viewer_counts())
        
        # Start connection checker
        if self.reconnect_task:
            self.reconnect_task.cancel()
        self.reconnect_task = asyncio.create_task(self.check_connection())
        
        # Start clip scheduler task
        if self.clip_scheduler_task:
            self.clip_scheduler_task.cancel()
        self.clip_scheduler_task = asyncio.create_task(self._process_clip_queue())
        
        # Start ML model saver task
        self.ml_saver_task = asyncio.create_task(self._save_ml_models_periodically())
        
        # Initialize and start metrics display thread if not already running
        if not self.metrics_thread or not self.metrics_thread.is_alive():
            self.metrics_thread = threading.Thread(target=self.run_metrics_display, daemon=True)
            self.metrics_thread.start()
            self.logger.info("Started metrics display update thread")
        
    async def event_message(self, message):
        """Handle incoming chat messages (optimized)."""
        # Don't process messages from the bot itself
        if message.echo:
            return
            
        try:
            # Get channel name without # prefix
            channel = message.channel.name.lower()
            
            if channel in self.analyzers:
                # Update last message time for connection monitoring
                self.last_message_time = time.time()
                
                # Add message to analyzer
                self.analyzers[channel].add_message(
                    message=message.content,
                    user_id=message.author.name,
                    timestamp=datetime.now()
                )
                    
                # Periodically update emotes (every 5 minutes)
                current_time = time.time()
                if not hasattr(self, '_last_emote_update'):
                    self._last_emote_update = {}
                if channel not in self._last_emote_update or current_time - self._last_emote_update[channel] > 300:
                    try:
                        users = await self.fetch_users(names=[channel])
                        if users:
                            channel_id = users[0].id
                            await self.analyzers[channel].update_emotes(channel_id)
                            self._last_emote_update[channel] = current_time
                    except Exception as e:
                        self.logger.error(f"Failed to update emotes: {e}")
                    
                # Get stats once for all operations
                stats = self.analyzers[channel].get_window_stats()
                
                # Check if the moment is clip-worthy using our hybrid scoring system
                clip_worthy_score = stats.get('clip_worthy_score', 0)
                
                # Clip is worthy if hybrid score is high enough
                if clip_worthy_score >= 0.85:  # Higher threshold for better quality clips
                    burst_score = stats.get('burst_score', 0)
                    velocity_relative = stats.get('velocity_relative', 0)
                    burst_relative = stats.get('burst_relative', 0)
                    ml_score = stats.get('ml_score', 0)
                    
                    # Determine trigger type based on what exceeded threshold more
                    if velocity_relative > burst_relative:
                        trigger_type = "high_velocity"
                        trigger_reason = f"velocity_rel={velocity_relative:.2f}"
                    else:
                        trigger_type = "emote_burst" 
                        trigger_reason = f"burst={burst_score:.1f}"
                    
                    # Log the clip reason with scores
                    self.logger.info(
                        f"Clip triggered by {trigger_type}: {trigger_reason} | "
                        f"Hybrid: {clip_worthy_score:.3f} | ML: {ml_score:.3f} | "
                        f"Viewers: {stats.get('viewer_count', 0):,}"
                    )
                    
                    # Schedule clip creation with 7-second delay
                    await self._schedule_delayed_clip(channel, trigger_type, stats.copy())
                
                # Update metrics display if available
                if hasattr(self, 'metrics_display') and self.metrics_display and self.metrics_display.ready.is_set():
                    self.metrics_display.update_metrics(channel, stats)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
            
        # Pass message to commands system
        await self.handle_commands(message)

    async def create_clip(self, channel: str, trigger_type: str, metrics: dict):
        """Create a clip when activity is detected."""
        try:
            now = time.time()
            
            # Use fixed cooldown of 10 seconds for all channels
            cooldown_period = 10  # Fixed cooldown of 10 seconds
            
            # Check if we're still on cooldown for this channel
            if channel in self.clip_cooldowns:
                last_attempt = self.clip_cooldowns[channel]
                time_since_last = now - last_attempt
                
                if time_since_last < cooldown_period and trigger_type != "test_command":
                    self.logger.info(
                        f"Clip rejected: On cooldown for {channel} ({time_since_last:.1f}s < {cooldown_period}s)"
                    )
                    return False
                
                self.logger.info(
                    f"Cooldown passed for {channel}: {time_since_last:.1f}s > {cooldown_period}s"
                )
            
            # Update the cooldown timestamp
            self.clip_cooldowns[channel] = now
            
            # Calculate threshold values for logging
            burst_score = metrics.get('burst_score', 0)
            velocity = self.analyzers[channel].get_normalized_chat_velocity()
            viewer_count = metrics.get('viewer_count', 0)
            
            self.logger.info(
                f"Creating clip for {channel}: velocity={velocity:.2f}, burst={burst_score:.1f}, "
                f"viewers={viewer_count:,}"
            )
            
            # Request clip creation
            success = self.clip_manager.request_clip(
                channel=channel,
                trigger_type=trigger_type,
                metrics=metrics
            )
            
            if not success:
                self.logger.warning(f"Failed to request clip for {channel}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating clip: {e}", exc_info=True)
            return False

    async def event_error(self, error: Exception, data: str = None):
        """Called when the bot encounters an error."""
        logger.error(f"Bot error: {error}", exc_info=True)
        
    def reconnect(self):
        """Attempt to reconnect the bot."""
        try:
            logger.info("Attempting to reconnect...")
            # Create a new event loop for reconnection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.connect())
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}", exc_info=True)
            # Wait and try again
            time.sleep(5)
            if self.running:
                self.reconnect()
        
    async def _create_clip(self, channel: str):
        """This method is deprecated - we now use the ClipManager instead."""
        logger.warning("_create_clip method is deprecated. Using clip_manager.request_clip instead.")
        
        # Get current stats
        stats = self.analyzers[channel].get_window_stats()
        
        # Request clip using ClipManager
        self.clip_manager.request_clip(channel, "api_call", stats)

    @commands.command(name='stats')
    async def stats_command(self, ctx):
        """Command to show current chat statistics."""
        channel = ctx.channel.name
        if channel not in self.analyzers:
            return
            
        stats = self.analyzers[channel].get_window_stats()
        
        # Get metrics with defaults to avoid key errors
        velocity = stats.get('norm_velocity', 0)
        burst = stats.get('burst_score', 0)
        velocity_threshold = stats.get('velocity_threshold', 2.5)
        burst_threshold = stats.get('burst_threshold', 20.0)
        rule_score = stats.get('rule_score', 0)
        ml_score = stats.get('ml_score', 0)
        clip_score = stats.get('clip_worthy_score', 0)
        viewers = stats.get('viewer_count', 0)
        
        # Send basic stats to chat
        await ctx.send(
            f"Chat Stats | "
            f"Velocity: {velocity:.2f}/{velocity_threshold:.1f} | "
            f"Burst: {burst:.1f}/{burst_threshold:.1f} | "
            f"ML: {ml_score:.3f} | "
            f"Hybrid: {clip_score:.3f} | "
            f"Viewers: {viewers:,}"
        )

    @commands.command(name='testclip')
    async def test_clip_command(self, ctx):
        """Command to force create a test clip."""
        channel = ctx.channel.name
        self.logger.info(f"TEST CLIP command received from {ctx.author.name} in channel {channel}")
        
        if channel not in self.analyzers:
            await ctx.send("Channel not being monitored by this bot.")
            return
            
        try:
            await ctx.send("⚠️ Testing clip creation - forcing clip now...")
            stats = self.analyzers[channel].get_window_stats()
            
            # Force high scores for testing
            test_stats = stats.copy()
            test_stats['burst_score'] = 100.0
            test_stats['clip_worthy_score'] = 1.0
            
            # Create clip using our create_clip method with test_command trigger
            success = await self.create_clip(
                channel=channel,
                trigger_type="test_command",
                metrics=test_stats
            )
            
            if success:
                await ctx.send("✅ Test clip created successfully!")
            else:
                await ctx.send("❌ Failed to create test clip. Check logs for details.")
                
        except Exception as e:
            self.logger.error(f"Error in test clip command: {e}", exc_info=True)
            await ctx.send(f"❌ Error creating test clip: {str(e)}")
            
    @commands.command(name='clipinfo')
    async def clip_info_command(self, ctx):
        """Command to show clip API information."""
        try:
            client_id = os.getenv('TWITCH_CLIENT_ID', '')
            access_token = os.getenv('TWITCH_ACCESS_TOKEN', '')
            refresh_token = os.getenv('TWITCH_REFRESH_TOKEN', '')
            
            status = []
            status.append(f"CLIENT_ID: {'✓ Present' if client_id else '✗ Missing'}")
            status.append(f"ACCESS_TOKEN: {'✓ Present' if access_token else '✗ Missing'}")  
            status.append(f"REFRESH_TOKEN: {'✓ Present' if refresh_token else '✗ Missing'}")
            
            await ctx.send(f"Clip API Info: {' | '.join(status)}")
        except Exception as e:
            self.logger.error(f"Error in clip info command: {e}", exc_info=True)
            await ctx.send(f"❌ Error checking clip info: {str(e)}")

    @commands.command(name='clips')
    async def show_recent_clips(self, ctx: commands.Context):
        """Show URLs of recently created clips"""
        clips = self.clip_manager.get_recent_clips(5)  # Get last 5 clips
        if clips:
            await ctx.send(f"Recent clips: {' | '.join(clips)}")
            await ctx.send(f"All clips are saved in: {self.clip_manager.clips_file}")
        else:
            await ctx.send("No clips have been created yet.")

    @commands.command(name='mlstats')
    async def ml_stats_command(self, ctx):
        """Command to show detailed ML statistics."""
        channel = ctx.channel.name
        if channel not in self.analyzers:
            return
            
        try:
            stats = self.analyzers[channel].get_window_stats()
            analyzer = self.analyzers[channel]
            
            # Format Metrics
            feature_count = len(analyzer.feature_history)
            baseline_count = len(analyzer.baseline_velocities)
            
            # Check if model was loaded from disk
            safe_channel = channel.lower().replace(' ', '_')
            model_path = analyzer.models_dir / safe_channel
            model_path = model_path.with_suffix('.joblib')
            model_source = "Loaded from disk" if model_path.exists() else "New (in training)"
            
            # Check model status
            model_status = "Active" if feature_count >= 100 and baseline_count >= 60 else "Training"
            training_progress = min(feature_count / 100, 1.0) * 100 if feature_count < 100 else 100.0
            
            # Get metrics
            velocity_relative = stats.get('velocity_relative', 0)
            burst_relative = stats.get('burst_relative', 0)
            rule_score = stats.get('rule_score', 0)
            ml_score = stats.get('ml_score', 0)
            clip_score = stats.get('clip_worthy_score', 0)
            
            # Format message for chat
            ml_info = (
                f"ML Status: {model_status} ({training_progress:.0f}% trained) | "
                f"Model: {model_source} | "
                f"Samples: {feature_count} | "
                f"Scores: Rule={rule_score:.3f}, ML={ml_score:.3f}, Hybrid={clip_score:.3f}"
            )
            
            await ctx.send(ml_info)
            
        except Exception as e:
            self.logger.error(f"Error in ML stats command: {e}", exc_info=True)
            await ctx.send("Error retrieving ML stats. Please check logs.")

    @commands.command(name='mlsave')
    async def ml_save_command(self, ctx):
        """Command to manually save the ML model."""
        channel = ctx.channel.name
        if channel not in self.analyzers:
            await ctx.send("Channel not being monitored by this bot.")
            return
            
        try:
            analyzer = self.analyzers[channel]
            feature_count = len(analyzer.feature_history)
            
            if feature_count < 100:
                await ctx.send(f"Cannot save ML model yet: Only {feature_count}/100 required samples collected.")
                return
                
            # Save the model
            if analyzer.save_ml_model():
                await ctx.send(f"✅ ML model saved successfully with {feature_count} samples!")
            else:
                await ctx.send("❌ Failed to save ML model. Check logs for details.")
                
        except Exception as e:
            self.logger.error(f"Error in ML save command: {e}", exc_info=True)
            await ctx.send(f"❌ Error saving ML model: {str(e)}")
            
    @commands.command(name='mlreset')
    async def ml_reset_command(self, ctx):
        """Command to reset the ML model."""
        channel = ctx.channel.name
        if channel not in self.analyzers:
            await ctx.send("Channel not being monitored by this bot.")
            return
            
        try:
            analyzer = self.analyzers[channel]
            
            # Create a fresh model
            analyzer.feature_history.clear()
            analyzer.scaler = StandardScaler()
            analyzer.isolation_forest = IsolationForest(
                contamination=0.03,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                bootstrap=True
            )
            
            # Remove saved models if they exist
            safe_channel = channel.lower().replace(' ', '_')
            base_path = analyzer.models_dir / safe_channel
            
            model_path = base_path.with_suffix('.joblib')
            scaler_path = base_path.with_name(f"{safe_channel}_scaler.joblib")
            features_path = base_path.with_name(f"{safe_channel}_features.pkl")
            baseline_path = base_path.with_name(f"{safe_channel}_baseline.pkl")
            
            # Delete files if they exist
            files_deleted = 0
            for path in [model_path, scaler_path, features_path, baseline_path]:
                if path.exists():
                    path.unlink()
                    files_deleted += 1
            
            await ctx.send(f"✅ ML model reset! Deleted {files_deleted} model files. Starting fresh training.")
                
        except Exception as e:
            self.logger.error(f"Error in ML reset command: {e}", exc_info=True)
            await ctx.send(f"❌ Error resetting ML model: {str(e)}")

    async def start(self):
        """Start the bot with metrics display."""
        try:
            # Initialize metrics display
            if not self.metrics_display:
                self.metrics_display = MultiChannelMetricsDisplay()
            
            # Start the viewer count updater
            self.viewer_update_task = asyncio.create_task(self._update_viewer_counts())
            
            # Connect to Twitch
            await self.connect()
            
        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}", exc_info=True)
            # Cleanup
            if self.viewer_update_task:
                self.viewer_update_task.cancel()
            await self.close()
            raise

    def run_metrics_display(self):
        """Run the metrics display in a separate thread."""
        try:
            while self.running:
                if hasattr(self, 'metrics_display') and self.metrics_display and self.metrics_display.ready.is_set():
                    for channel in self.analyzers:
                        stats = self.analyzers[channel].get_window_stats()
                        self.metrics_display.update_metrics(channel, stats)
                time.sleep(0.1)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in metrics display thread: {e}", exc_info=True)
            else:
                print(f"Error in metrics display thread: {e}")

    async def _schedule_delayed_clip(self, channel: str, trigger_type: str, metrics: dict):
        """Schedule a clip for creation after a delay."""
        scheduled_time = time.time() + self.clip_delay
        self.logger.info(f"Scheduling clip for {channel} in {self.clip_delay:.1f} seconds (trigger: {trigger_type})")
        
        # Add to pending clips queue with scheduled time
        self.pending_clips.append({
            'channel': channel,
            'trigger_type': trigger_type,
            'metrics': metrics,
            'scheduled_time': scheduled_time,
            'detected_time': time.time()
        })
        
    async def _process_clip_queue(self):
        """Process the pending clips queue at regular intervals."""
        self.logger.info("Starting clip scheduler task")
        
        while self.running:
            try:
                # Current time
                now = time.time()
                
                # Find clips that are ready to be created
                ready_clips = [clip for clip in self.pending_clips if now >= clip['scheduled_time']]
                
                # Process ready clips
                for clip in ready_clips:
                    channel = clip['channel']
                    trigger_type = clip['trigger_type']
                    metrics = clip['metrics']
                    delay = now - clip['detected_time']
                    
                    self.logger.info(f"Creating delayed clip for {channel} after {delay:.1f}s delay (trigger: {trigger_type})")
                    
                    # Create the clip
                    success = await self.create_clip(channel, trigger_type, metrics)
                    
                    if success:
                        self.logger.info(f"Successfully created delayed clip for {channel}")
                    else:
                        self.logger.warning(f"Failed to create delayed clip for {channel}")
                
                # Remove processed clips from queue
                self.pending_clips = [clip for clip in self.pending_clips if clip not in ready_clips]
                
                # Sleep briefly to avoid high CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in clip scheduler: {e}", exc_info=True)
                await asyncio.sleep(1)  # Sleep longer on error
                
        self.logger.info("Clip scheduler task stopped")
        
    async def _save_ml_models_periodically(self):
        """Periodically save ML models to disk for all channels."""
        self.logger.info("Starting ML model saver task")
        
        # Wait initially to let models load/train
        await asyncio.sleep(60)
        
        while self.running:
            try:
                # Save models for all channels with enough data
                saved_count = 0
                for channel, analyzer in self.analyzers.items():
                    if len(analyzer.feature_history) >= 100:
                        if analyzer.save_ml_model():
                            saved_count += 1
                
                if saved_count > 0:
                    self.logger.info(f"Saved ML models for {saved_count} channels")
                
                # Save every 10 minutes
                await asyncio.sleep(600)
                
            except Exception as e:
                self.logger.error(f"Error saving ML models: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before trying again
                
        self.logger.info("ML model saver task stopped")

async def main_async():
    """Async entry point for the bot."""
    bot = None
    try:
        # Create the bot
        bot = ClipperBot()
        
        # Initialize metrics display
        bot.metrics_display = MultiChannelMetricsDisplay()
        
        # Start the bot
        await bot.start()
        
        # Keep the bot running
        while bot.running:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        if bot:
            await bot.close()
        raise
    finally:
        if bot:
            await bot.close()

def main():
    """Main entry point for the bot."""
    try:
        # Process command line arguments for channels
        if len(sys.argv) > 1 and '=' in sys.argv[1] and sys.argv[1].startswith('TWITCH_CHANNELS='):
            channels_str = sys.argv[1].split('=', 1)[1]
            channels = [channel.strip() for channel in channels_str.split(',')]
            logger.info(f"Using channels from command line: {channels}")
            os.environ['TWITCH_CHANNELS'] = channels_str
        else:
            # Load environment variables
            dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            load_dotenv(dotenv_path)
            channels_str = os.getenv('TWITCH_CHANNELS', '')
            channels = [channel.strip() for channel in channels_str.split(',') if channel.strip()]
            logger.info(f"Using channels from environment: {channels}")
        
        if not channels:
            logger.error("No channels specified! Please set TWITCH_CHANNELS environment variable")
            sys.exit(1)
        
        # Create metrics display in main thread
        metrics_display = MultiChannelMetricsDisplay()
        # Pre-initialize channel displays
        metrics_display.initialize_displays(channels)
        
        # Create shutdown event for bot thread
        bot_shutdown = threading.Event()
        
        # Create a separate thread for the bot
        def run_bot():
            try:
                # Create and set event loop for bot thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Create and start the bot
                bot = ClipperBot()
                bot.metrics_display = metrics_display
                bot.running = True  # Ensure running flag is set
                
                try:
                    # Start the bot
                    loop.run_until_complete(bot.start())
                    
                    # Run until shutdown event is set
                    while not bot_shutdown.is_set():
                        loop.run_until_complete(asyncio.sleep(0.1))
                        
                finally:
                    # Cleanup bot
                    bot.running = False  # Set running flag to False
                    loop.run_until_complete(bot.close())
                    loop.close()
            
            except Exception as e:
                logging.error(f"Error in bot thread: {e}", exc_info=True)
                
        # Set up signal handlers in main thread
        def signal_handler(signum, frame):
            logging.info("Received shutdown signal, cleaning up...")
            bot_shutdown.set()
            metrics_display.close()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start bot thread
        bot_thread = threading.Thread(target=run_bot)
        bot_thread.daemon = True
        bot_thread.start()
        
        # Run metrics display in main thread
        logging.info("Starting metrics display in main thread...")
        metrics_display.run()
        
        # If we get here, the window was closed
        bot_shutdown.set()
        
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
        if 'bot_shutdown' in locals():
            bot_shutdown.set()
    except Exception as e:
        logging.error(f"Failed to start application: {e}", exc_info=True)
        raise 