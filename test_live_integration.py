#!/usr/bin/env python3
"""
Test Live Integration - Verify bot data is accessible via API
"""

import requests
import time
import json

def test_api_connection():
    """Test basic API connectivity"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_channels_endpoint():
    """Test channels endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/channels", timeout=5)
        if response.status_code == 200:
            data = response.json()
            channels = data.get('channels', [])
            print(f"✅ Channels endpoint working: {len(channels)} channels")
            print(f"   📺 Channels: {channels}")
            return len(channels) > 0
        else:
            print(f"❌ Channels endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Channels endpoint error: {e}")
        return False

def test_live_data(channel="xqc"):
    """Test live data for a specific channel"""
    try:
        response = requests.get(f"http://localhost:8000/api/channels/{channel}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            ml_status = data.get('ml_status', {})
            data_status = data.get('data_status', {})
            
            print(f"✅ Live data for {channel}:")
            print(f"   👥 Viewers: {stats.get('viewer_count', 0):,}")
            print(f"   💬 Total Messages: {data_status.get('total_messages', 0)}")
            print(f"   ⚡ Velocity: {stats.get('raw_velocity', 0):.2f} msg/sec")
            print(f"   📈 Burst Score: {stats.get('burst_score', 0):.1f}")
            print(f"   🎯 Clip Score: {stats.get('clip_worthy_score', 0):.3f}")
            print(f"   🤖 ML Status: {ml_status.get('model_status', 'unknown')}")
            print(f"   📊 Training Samples: {ml_status.get('training_samples', 0)}")
            
            # Check if this looks like real data (not test data)
            viewer_count = stats.get('viewer_count', 0)
            message_count = data_status.get('total_messages', 0)
            
            if viewer_count > 1000 and message_count > 0:
                print("🎉 This appears to be REAL LIVE DATA!")
                return True
            elif message_count > 0:  
                print("⚠️ Has message data but low viewer count - might be test data")
                return True
            else:
                print("❌ No message data - likely not connected to live bot")
                return False
        else:
            print(f"❌ Stats endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Stats endpoint error: {e}")
        return False

def monitor_live_updates(channel="xqc", duration=30):
    """Monitor live updates for a short period"""
    print(f"🔍 Monitoring live updates for {channel} for {duration} seconds...")
    
    start_time = time.time()
    previous_message_count = None
    
    while time.time() - start_time < duration:
        try:
            response = requests.get(f"http://localhost:8000/api/channels/{channel}/stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
                message_count = data.get('data_status', {}).get('total_messages', 0)
                velocity = data.get('stats', {}).get('raw_velocity', 0)
                
                if previous_message_count is not None:
                    if message_count > previous_message_count:
                        diff = message_count - previous_message_count
                        print(f"📈 Messages increased by {diff} (total: {message_count}, velocity: {velocity:.1f})")
                    elif message_count == previous_message_count:
                        print(f"📊 Messages stable at {message_count} (velocity: {velocity:.1f})")
                
                previous_message_count = message_count
            
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            break
    
    print("🏁 Monitoring complete")

def main():
    """Run integration tests"""
    print("🧪 Testing Live Bot-API Integration")
    print("=" * 40)
    
    # Test API connectivity
    if not test_api_connection():
        print("❌ API not available - make sure it's running")
        return
    
    print()
    
    # Test channels endpoint
    if not test_channels_endpoint():
        print("❌ No channels registered - bot may not be connected")
        return
    
    print()
    
    # Test live data
    if test_live_data():
        print()
        print("🎉 SUCCESS: Bot data is flowing to API!")
        
        # Optionally monitor for updates
        try:
            monitor_live_updates()
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
    else:
        print("❌ FAILED: Bot data not reaching API")
    
    print("\n📋 Integration test complete!")
    print("🌐 API docs: http://localhost:8000/docs")
    print("📊 Channel stats: http://localhost:8000/api/channels/xqc/stats")

if __name__ == "__main__":
    main() 