import tkinter as tk
from tkinter import ttk
import queue
from datetime import datetime
import threading
import logging
import time
import asyncio

logger = logging.getLogger(__name__)

class MetricsDisplay:
    def __init__(self, channel="Unknown"):
        """Initialize the metrics display for a specific channel."""
        self.root = None
        self.update_queue = queue.Queue()
        self.ready = threading.Event()
        self.running = True
        self.last_stats = None
        self.last_channel = channel  # Default to the assigned channel
        self.last_update = time.time()
        self.update_interval = 0.1  # Update more frequently
        
        # Initialize GUI components to None
        self.main_frame = None
        self.channel_label = None
        self.metrics_frame = None
        self.status_bar = None
        self.status_var = None
        
        # Initialize metric variables
        self.viewers_var = None
        self.msg_sec_var = None
        self.burst_var = None
        self.ml_score_var = None
        self.rule_score_var = None
        
    def initialize(self):
        """Initialize the GUI (must be called from the main thread)"""
        if self.root is not None:
            return  # Already initialized
            
        try:
            # Create the root window
            self.root = tk.Tk() if not tk._default_root else tk.Toplevel()
            self.root.title(f"Twitch Chat Metrics - {self.last_channel}")
            self.root.geometry("350x250")
            
            # Calculate position based on channel name hash to distribute windows
            channel_hash = hash(self.last_channel) % 10000
            offset_x = (channel_hash % 100) * 20
            offset_y = (channel_hash // 100) * 20
            
            # Center the window on screen with offset
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = (screen_width - 350) // 2 + offset_x
            y = (screen_height - 250) // 2 + offset_y
            self.root.geometry(f"350x250+{x}+{y}")
            
            # Set up window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Create main frame
            self.main_frame = ttk.Frame(self.root, padding="10")
            self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Channel label
            self.channel_label = ttk.Label(
                self.main_frame,
                text=f"Channel: {self.last_channel}",
                font=('Arial', 16, 'bold')
            )
            self.channel_label.grid(row=0, column=0, pady=10)
            
            # Metrics frame
            self.metrics_frame = ttk.LabelFrame(
                self.main_frame,
                text="Live Metrics",
                padding="10"
            )
            self.metrics_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Create metric labels
            self.metrics = {
                "viewers": ("👥 Viewers", "0"),
                "msg_sec": ("⚡ Msg/sec", "0.0"),
                "burst": ("🎭 Emote Burst", "0.00"),
                "ml_score": ("🤖 ML Score", "0.000"),
                "rule_score": ("🤖 Rule Score", "0.000")
            }
            
            # Create and place metric labels
            for i, (key, (label_text, value)) in enumerate(self.metrics.items()):
                ttk.Label(
                    self.metrics_frame,
                    text=label_text + ":",
                    font=('Arial', 12)
                ).grid(row=i, column=0, sticky=tk.W, pady=2)
                
                setattr(self, f"{key}_var", tk.StringVar(value=value))
                ttk.Label(
                    self.metrics_frame,
                    textvariable=getattr(self, f"{key}_var"),
                    font=('Arial', 12, 'bold')
                ).grid(row=i, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Status bar
            self.status_var = tk.StringVar(value="Initializing...")
            self.status_bar = ttk.Label(
                self.main_frame,
                textvariable=self.status_var,
                font=('Arial', 10),
                relief=tk.SUNKEN
            )
            self.status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
            
            # Configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            self.main_frame.columnconfigure(0, weight=1)
            self.main_frame.rowconfigure(1, weight=1)
            
            # Start update checker
            self.check_updates()
            
            # Start periodic updater
            self.force_update()
            
            # Final update and remove topmost
            self.root.update()
            self.root.attributes('-topmost', False)
            
            # Set ready flag
            self.ready.set()
            
        except Exception as e:
            logger.error(f"Error initializing display for {self.last_channel}: {e}", exc_info=True)
            raise
        
    def force_update(self):
        """Force a periodic update of the display."""
        if not self.running:
            return
            
        try:
            current_time = time.time()
            
            # If we have stats and it's been more than a second, force an update
            if self.last_stats is not None and (current_time - self.last_update >= 1.0):
                self.last_stats['raw_velocity'] = 0.0  # Reset velocity if no new messages
                self.update_display(self.last_channel, self.last_stats)
                self.last_update = current_time
                
        except Exception as e:
            logger.error(f"Error in force update: {e}", exc_info=True)
            
        # Schedule next force update
        if self.running and self.root:
            self.root.after(100, self.force_update)  # Check every 100ms
            
    def update_display(self, channel: str, stats: dict):
        """Update the display with new stats."""
        try:
            # Update channel name
            self.channel_label.config(text=f"Channel: {channel}")
            
            # Update metrics
            self.viewers_var.set(f"{stats.get('viewer_count', 0):,}")
            
            # Format velocity
            raw_velocity = stats.get('raw_velocity', 0.0)
            if isinstance(raw_velocity, str):
                self.msg_sec_var.set(raw_velocity)
            else:
                if raw_velocity > 100:
                    self.msg_sec_var.set(f"{raw_velocity:.0f}")
                elif raw_velocity > 10:
                    self.msg_sec_var.set(f"{raw_velocity:.1f}")
                else:
                    self.msg_sec_var.set(f"{raw_velocity:.1f}")
            
            self.burst_var.set(f"{stats.get('burst_score', 0.0):.1f}")
            self.ml_score_var.set(f"{stats.get('clip_worthy_score', 0.0):.3f}")
            self.rule_score_var.set(f"{stats.get('rule_score', 0.0):.3f}")
            
            # Update status
            status = []
            clip_worthy_score = stats.get('clip_worthy_score', 0)
            ml_score = stats.get('ml_score', 0)
            rule_score = stats.get('rule_score', 0)
            
            # Add clip worthiness indicator
            if clip_worthy_score > 0.85:
                status.append("🔥 CLIP WORTHY!")
            elif clip_worthy_score > 0.7:
                status.append("👀 Almost clip-worthy")
            
            # Add ML score indicator
            if ml_score > 0.85:
                status.append("🤖 ML: Very High")
            elif ml_score > 0.7:
                status.append("🤖 ML: High")
                
            # Add burst score indicator
            if stats.get('burst_score', 0) > 30:
                status.append("🎭 Massive emote burst!")
            elif stats.get('burst_score', 0) > 15:
                status.append("✨ High emote activity!")
                
            # Add velocity indicator
            velocity = float(stats.get('raw_velocity', 0))
            if velocity > 100:
                status.append("⚡ Chat going wild!")
            elif velocity > 50:
                status.append("💨 Fast chat!")
            elif velocity == 0 and not status:
                status.append("Monitoring chat...")
            
            self.status_var.set(" | ".join(status) if status else "Monitoring chat...")
            
        except Exception as e:
            logger.error(f"Error updating display: {e}", exc_info=True)
            
    def check_updates(self):
        """Check for and process updates from the queue."""
        if not self.running:
            return
            
        try:
            # Process all pending updates, keep only the latest
            latest_update = None
            while True:
                try:
                    latest_update = self.update_queue.get_nowait()
                except queue.Empty:
                    break
                    
            if latest_update:
                channel, stats, _ = latest_update
                self.last_stats = stats.copy()
                self.last_channel = channel
                self.last_update = time.time()
                self.update_display(channel, stats)
                
        except Exception as e:
            logger.error(f"Error in check_updates: {e}", exc_info=True)
            
        # Schedule next update check
        if self.running and self.root:
            self.root.after(16, self.check_updates)  # ~60 FPS refresh rate
            
    def update_metrics(self, channel: str, stats: dict, recent_messages: list = None):
        """Thread-safe method to update metrics."""
        try:
            # If stats is a float, convert it to a dict with burst_score
            if isinstance(stats, float):
                stats = {
                    'burst_score': stats,
                    'raw_velocity': 0.0,
                    'viewer_count': 0,
                    'clip_worthy_score': 0.0
                }
            else:
                # Make a copy of the stats dict
                stats = dict(stats)
                
            # Format velocity with more precision for high numbers
            raw_velocity = stats.get('raw_velocity', 0.0)
            if raw_velocity > 100:
                stats['raw_velocity'] = f"{raw_velocity:.0f}"
            elif raw_velocity > 10:
                stats['raw_velocity'] = f"{raw_velocity:.1f}"
            else:
                stats['raw_velocity'] = f"{raw_velocity:.2f}"
                
            self.update_queue.put((channel, stats, recent_messages))
        except Exception as e:
            logger.error(f"Error queueing metrics update: {e}")
            
    def on_closing(self):
        """Handle window close button."""
        self.close()
        
    def run(self):
        """Start the display window."""
        try:
            # Initialize in the main thread
            self.initialize()
            logger.info("Metrics display initialized, starting main loop...")
            
            if self.root:
                # Start main loop
                self.root.mainloop()
                
        except Exception as e:
            logger.error(f"Error running display: {e}", exc_info=True)
        finally:
            self.running = False
            
    def close(self):
        """Close the display window."""
        if not self.running:
            return
            
        self.running = False
        if self.root:
            try:
                # Schedule the shutdown in the main thread
                self.root.after(0, self._shutdown)
            except Exception as e:
                logger.error(f"Error scheduling display shutdown: {e}", exc_info=True)
                
    def _shutdown(self):
        """Perform actual shutdown in the main thread."""
        try:
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error in display shutdown: {e}", exc_info=True)

class MultiChannelMetricsDisplay:
    """Manages separate metrics displays for multiple channels."""
    
    def __init__(self):
        """Initialize the multi-channel metrics display."""
        self.displays = {}  # Channel -> MetricsDisplay
        self.ready = threading.Event()
        self.ready.set()  # Start ready
        self.running = True
        self.logger = logging.getLogger(__name__)
        
    def initialize_displays(self, channels):
        """Initialize displays for a list of channels."""
        if not channels:
            self.logger.warning("No channels provided to initialize displays!")
            return False
            
        self.logger.info(f"Initializing displays for channels: {channels}")
        for channel in channels:
            if channel and channel not in self.displays:
                self.displays[channel] = MetricsDisplay(channel)
                self.logger.info(f"Created display for channel: {channel}")
                
        if not self.displays:
            self.logger.warning("Failed to create any displays!")
            return False
            
        self.ready.set()
        return True
        
    def update_metrics(self, channel: str, stats: dict, recent_messages: list = None):
        """Update metrics for a specific channel."""
        if not channel:
            return
            
        # Create display for channel if it doesn't exist
        if channel not in self.displays:
            self.logger.info(f"Creating new display for channel: {channel}")
            self.displays[channel] = MetricsDisplay(channel)
            
        # Update the specific channel's display
        if self.displays[channel].ready.is_set():
            self.displays[channel].update_metrics(channel, stats, recent_messages)
        
    def run(self):
        """Run all the metric displays."""
        if not self.displays:
            self.logger.warning("No channel displays to run! Make sure to call initialize_displays() first.")
            return
            
        self.logger.info(f"Running metrics displays for {len(self.displays)} channels: {', '.join(self.displays.keys())}")
            
        # Create a root window to manage all displays
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Initialize all displays
        for channel, display in self.displays.items():
            try:
                display.initialize()
                self.logger.info(f"Initialized metrics display for channel: {channel}")
            except Exception as e:
                self.logger.error(f"Failed to initialize display for {channel}: {e}")
                
        # Set up main loop and close handler
        def on_root_close():
            self.close()
            root.quit()
            
        root.protocol("WM_DELETE_WINDOW", on_root_close)
        
        try:
            # Run the Tkinter main loop
            root.mainloop()
        except Exception as e:
            self.logger.error(f"Error in metrics display main loop: {e}", exc_info=True)
        finally:
            self.running = False
            
    def close(self):
        """Close all metrics displays."""
        self.running = False
        for channel, display in self.displays.items():
            try:
                display.close()
            except Exception as e:
                self.logger.error(f"Error closing display for {channel}: {e}")
                
        self.displays.clear()