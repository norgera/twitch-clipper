# Twitch Clipper

Automatically creates clips from Twitch streams during exciting moments by analyzing chat activity. It uses machine learning to detect clip-worthy moments based on chat velocity spikes, emote bursts and anomaly detection models that learn each channel’s unique patterns. 
Its core features include an adaptive ML model that continuously improves and persists between sessions, dynamic sensitivity thresholds that adjust to channel size, separate models for each channel and a hybrid rule-and-ML approach for reliable triggering. 
In operation, it monitors real-time chat metrics, applies ML to identify unusual activity, waits seven seconds to ensure the full context is captured, then generates the clip—while continuously refining its understanding from new data. 
Under the hood, it combines rules-based triggers with an Isolation Forest anomaly detector, automatically scales thresholds by viewer count and stores all models in a models/ directory for persistent learning.

**Storage Details**:
- **ML Models**: Stored in the `models/` directory with channel-specific filenames for each broadcaster
- **Clip Records**: URLs and metadata saved in `recordings/clip_urls.txt` 
- **Auto-save**: Models are automatically saved every 10 minutes and on shutdown

## Quick Start

1. **Install**
   ```bash
   git clone https://github.com/norgera/twitch-clipper.git
   cd twitch-clipper
   pip install -r requirements.txt
   ```

2. **Configure**  
   Create a `.env` file with your Twitch credentials:
   ```
   TWITCH_CLIENT_ID=your_client_id
   TWITCH_CLIENT_SECRET=your_client_secret
   TWITCH_ACCESS_TOKEN=your_user_oauth_token_with_clips_edit_scope
   TWITCH_REFRESH_TOKEN=your_refresh_token
   TWITCH_CHANNELS=channel1,channel2
   ```

3. **Run**
   ```bash
   python -m app
   ```

   Type `!testclip` in Twitch chat to force a test clip 

## Token Generation

First [create your Twitch app here](https://dev.twitch.tv/console)
Take note of your ID, Secret, and Redirect URL

In your terminal:
```bash
export TWITCH_CLIENT_ID=YOUR_CLIENT_ID
export TWITCH_CLIENT_SECRET=YOUR_CLIENT_SECRET
export REDIRECT_URI=YOUR_REDIRECT_URL
```

```bash
AUTH_URL="https://id.twitch.tv/oauth2/authorize?\
client_id=${TWITCH_CLIENT_ID}&\
redirect_uri=${REDIRECT_URI}&\
response_type=code&\
scope=clips:edit+chat:read+chat:edit"
echo "Open this in your browser and accept:"
echo "${AUTH_URL}"
```

after you click “Authorize,” you’ll get redirected to:
"/localhost:3000/auth/callback?code=PASTE_THIS_CODE"

exchange that code for your tokens:
```bash
curl -X POST https://id.twitch.tv/oauth2/token \
  -d client_id=${TWITCH_CLIENT_ID} \
  -d client_secret=${TWITCH_CLIENT_SECRET} \
  -d code=PASTE_THIS_CODE \
  -d grant_type=authorization_code \
  -d redirect_uri=${REDIRECT_URI}
```

You’ll get back JSON:
```
{
  "access_token":"NEW_USER_TOKEN",
  "refresh_token":"NEW_REFRESH_TOKEN",
  "expires_in":3600,
  "scope":["clips:edit","chat:read","chat:edit"],
  "token_type":"bearer"
}
```
Copy those into your .env:
TWITCH_ACCESS_TOKEN=NEW_USER_TOKEN
TWITCH_REFRESH_TOKEN=NEW_REFRESH_TOKEN


## Technologies Used

### Core Libraries
- **TwitchIO**: Powers the bot's ability to connect to Twitch chat and process messages
- **NumPy**: Handles numerical operations for data processing and analysis
- **scikit-learn**: Provides machine learning capabilities, specifically the Isolation Forest algorithm for anomaly detection
- **joblib**: Enables model persistence between sessions

### APIs & Integration
- **Twitch Helix API**: Creates clips and fetches channel/emote information via RESTful endpoints
- **Twitch Chat/PubSub**: Connects to real-time chat via WebSockets using the TwitchIO library
- **7TV API**: Retrieves custom third-party emotes for enhanced emote detection

### User Experience
- **Tkinter**: Delivers a multi-window GUI for live metrics visualization (temporary)
- **VaderSentiment**: Analyzes chat sentiment to detect emotional reactions




## Feature Wish List
1. **Web dashboard**
Provide a browser-based interface where users can log in to run the application, manage their channels, view clip analytics, and download or export their data.

2. **Integrate video & audio analysis**
Add support for volume‐spike detection, audio sentiment scoring, scene classification and change detection to complement chat-based triggers.

3. **Optimize for smaller streamers**
Adjust detection logic to rely more on visual and audio cues when chat volume is low.

4. **Real-time alerts & integrations**
Send instant notifications (via Discord, Slack, email or SMS) whenever a clip is created