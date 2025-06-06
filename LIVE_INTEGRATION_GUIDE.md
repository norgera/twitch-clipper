# Live Bot-API Integration Guide

## Overview

This system connects your Twitch bot's real-time chat data directly to your API, so the API shows actual live data instead of test/mock data.

## ⚠️ Important: Environment Setup

**All credentials should be in your `.env` file - never hardcode them!**

Make sure your `.env` file contains:
```bash
TWITCH_CLIENT_ID=your_client_id_here
TWITCH_CLIENT_SECRET=your_client_secret_here
TWITCH_ACCESS_TOKEN=your_access_token_here
TWITCH_REFRESH_TOKEN=your_refresh_token_here
TWITCH_CHANNELS=xqc
TWITCH_BOT_USERNAME=your_bot_username
```

The integration scripts will automatically load these from your `.env` file.

## The Problem (Before)

- **Bot** (`python -m app`) - Collects real live Twitch data ✅
- **API** (`python run_api.py`) - Shows only test data ❌
- **No connection** between bot and API

## The Solution (After)

- **Integrated System** - Bot and API run together and share data ✅
- **Real-time updates** - API gets live data from bot instantly ✅
- **Single command** to start everything ✅

## Quick Start

### Option 1: Integrated System (Recommended)
Run both bot and API in one command:
```bash
python run_integrated_system.py
```

This will:
1. Load credentials from your `.env` file
2. Start the API server on `http://localhost:8000`
3. Start the Twitch bot 
4. Automatically connect them to share live data

### Option 2: Separate Processes (Advanced)
If you prefer to run them separately:

Terminal 1 - Start API:
```bash
python run_api.py
```

Terminal 2 - Start Bot:
```bash
python -m app
```

The bot will automatically register with the API when it starts.

## Testing the Integration

Run the test script to verify everything is working:
```bash
python test_live_integration.py
```

This will check:
- ✅ API connectivity
- ✅ Bot registration
- ✅ Live data flow
- ✅ Real-time updates

## API Endpoints

Once running, you can access:

- **Health**: http://localhost:8000/health
- **All Channels**: http://localhost:8000/api/channels  
- **Live Stats**: http://localhost:8000/api/channels/xqc/stats
- **API Docs**: http://localhost:8000/docs

## Live Data Examples

### Before (Test Data)
```json
{
  "viewer_count": 0,
  "total_messages": 0,
  "velocity": 0.0,
  "status": "test"
}
```

### After (Real Live Data)
```json
{
  "viewer_count": 27074,
  "total_messages": 1247,
  "raw_velocity": 8.5,
  "burst_score": 15.2,
  "clip_worthy_score": 0.742,
  "ml_status": "active"
}
```

## How It Works

1. **Bot Initialization**: Bot creates chat analyzers for each channel
2. **API Registration**: Bot registers its analyzers with the API service
3. **Real-time Updates**: Every 10 messages, bot updates API with fresh data
4. **Live Data Access**: API endpoints serve the bot's real-time data

## Key Files

- `run_integrated_system.py` - **MAIN SCRIPT** to run everything (uses .env)
- `app/twitch_bot.py` - Modified to register with API
- `api/bot_adapter.py` - Handles bot-API data sharing
- `test_live_integration.py` - Verification script

**⚠️ Deprecated scripts**: Files like `live_data_bridge.py`, `start_bot_with_api.py` etc. are older versions with hardcoded credentials. Use `run_integrated_system.py` instead.

## Troubleshooting

### API shows no channels
- Make sure bot is running and connected to Twitch
- Check bot logs for API registration messages
- Verify API server is accessible on port 8000

### No live data updates
- Ensure the channel has active chat (try xqc)
- Check if bot is receiving messages
- Run test script to monitor updates

### Connection errors
- Verify environment variables are set in `.env` file:
  - `TWITCH_ACCESS_TOKEN`
  - `TWITCH_CHANNELS=xqc`
- Check network connectivity
- Ensure no firewall blocking port 8000

### Missing credentials
- Make sure your `.env` file exists in the project root
- Verify all required variables are set (see Environment Setup above)
- Never hardcode credentials in scripts

## Environment Setup

Make sure these are set in your `.env` file (not in environment or hardcoded):
```bash
TWITCH_CLIENT_ID=your_client_id_here
TWITCH_CLIENT_SECRET=your_client_secret_here  
TWITCH_ACCESS_TOKEN=your_access_token_here
TWITCH_REFRESH_TOKEN=your_refresh_token_here
TWITCH_CHANNELS=xqc
TWITCH_BOT_USERNAME=your_bot_username
```

## Success Indicators

You'll know it's working when:
- ✅ Bot logs show "Registered bot analyzers with API service"
- ✅ API `/api/channels` shows your monitored channels
- ✅ Channel stats show real viewer counts (>1000 for xqc)
- ✅ Message counts increase as chat activity happens
- ✅ Test script reports "REAL LIVE DATA!"

## Next Steps

Once integrated:
1. Monitor multiple channels by updating `TWITCH_CHANNELS` in `.env`
2. Build frontend apps using the live API data
3. Set up webhooks for real-time notifications
4. Create clips based on ML analysis scores

---

🎉 **Success!** Your API now shows real live Twitch chat data from your bot! 