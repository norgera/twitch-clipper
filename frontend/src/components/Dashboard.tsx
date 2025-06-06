import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

interface SystemStatus {
  channels: Record<string, any>;
  totals: {
    channels: number;
    total_messages: number;
    connections: number;
  };
}

interface RecentMessage {
  text: string;
  user: string;
  timestamp: string;
  sentiment: number;
}

interface ChannelStats {
  channel: string;
  timestamp: number;
  stats: {
    viewer_count: number;
    raw_velocity: number;
    velocity_zscore: number;
    velocity_relative: number;
    burst_score: number;
    burst_relative: number;
    rule_score: number;
    ml_score: number;
    clip_worthy_score: number;
    sentiment: number;
  };
  ml_status: {
    model_loaded: boolean;
    training_samples: number;
    baseline_samples: number;
    model_status: string;
    last_prediction: number;
    current_baseline_count: number;
    samples_used_for_training: number;
  };
  data_status: {
    total_messages: number;
    emote_window_size: number;
    memory_stats: {
      tracked_users: number;
      tracked_emotes: number;
      total_events: number;
      max_users_limit: number;
    };
  };
}

interface RealTimeData {
  channel: string;
  timestamp: number;
  stats: any;
  recent_messages: RecentMessage[];
  ml_metrics: {
    feature_count: number;
    model_status: string;
  };
  connection_info: {
    connected_clients: number;
  };
}

const API_BASE_URL = 'http://localhost:8000';

const Dashboard: React.FC = () => {
  const [channels, setChannels] = useState<string[]>([]);
  const [selectedChannel, setSelectedChannel] = useState<string>('');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [channelStats, setChannelStats] = useState<ChannelStats | null>(null);
  const [realtimeData, setRealtimeData] = useState<RealTimeData | null>(null);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string>('');
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [chatMessages, setChatMessages] = useState<RecentMessage[]>([]);

  const fetchChannels = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/channels`);
      setChannels(response.data.channels);
      if (response.data.channels.length > 0 && !selectedChannel) {
        setSelectedChannel(response.data.channels[0]);
      }
      setError('');
    } catch (err) {
      setError('Failed to fetch channels. Make sure the API server is running.');
      console.error('Error fetching channels:', err);
    }
  }, [selectedChannel]);

  const fetchSystemStatus = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/system/status`);
      setSystemStatus(response.data);
      setError('');
    } catch (err) {
      console.error('Error fetching system status:', err);
    }
  }, []);

  const fetchChannelStats = useCallback(async (channel: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/channels/${channel}/stats`);
      const stats: ChannelStats = response.data;
      setChannelStats(stats);
      
      // Debug logging
      console.log('Channel Stats Response:', stats);
      console.log('ML Score from API:', stats.stats.ml_score);
      console.log('Clip Worthy Score from API:', stats.stats.clip_worthy_score);
      console.log('Rule Score from API:', stats.stats.rule_score);
      console.log('ML Status:', stats.ml_status);
      console.log('Model Loaded:', stats.ml_status.model_loaded);
      console.log('Training Samples:', stats.ml_status.training_samples);
      console.log('Model Status:', stats.ml_status.model_status);
      
      // Check for zero scores and log potential causes
      if (stats.stats.ml_score === 0) {
        console.warn('⚠️ ML Score is 0! Potential causes:');
        console.warn('- Model not loaded:', !stats.ml_status.model_loaded);
        console.warn('- Insufficient training samples:', stats.ml_status.training_samples < stats.ml_status.baseline_samples);
        console.warn('- Model status:', stats.ml_status.model_status);
        console.warn('- Total messages:', stats.data_status.total_messages);
      }
      
      // Add to historical data for charts
      const timestamp = new Date(stats.timestamp * 1000).toLocaleTimeString();
      setHistoricalData(prev => [
        ...prev.slice(-19), // Keep last 20 data points
        {
          time: timestamp,
          viewer_count: stats.stats.viewer_count,
          raw_velocity: stats.stats.raw_velocity,
          burst_score: stats.stats.burst_score,
          ml_score: (stats.stats.ml_score || 0) * 100, // Convert to percentage
          clip_worthy_score: (stats.stats.clip_worthy_score || 0) * 100, // Convert to percentage
          sentiment: stats.stats.sentiment,
          rule_score: (stats.stats.rule_score || 0) * 100, // Convert to percentage
        }
      ]);
      
      setError('');
    } catch (err) {
      setError(`Failed to fetch stats for channel: ${channel}`);
      console.error('Error fetching channel stats:', err);
    }
  }, []);

  // Fetch available channels and system status
  useEffect(() => {
    fetchChannels();
    fetchSystemStatus();
    
    // Set up periodic updates
    const statusInterval = setInterval(() => {
      fetchSystemStatus();
    }, 5000);

    return () => clearInterval(statusInterval);
  }, [fetchChannels, fetchSystemStatus]);

  // Set up periodic channel stats fetching
  useEffect(() => {
    if (selectedChannel) {
      // Initial fetch
      fetchChannelStats(selectedChannel);
      
      // Set up periodic updates for channel stats
      const statsInterval = setInterval(() => {
        fetchChannelStats(selectedChannel);
      }, 1000); // Update every 1 second - 10ms was too fast!

      return () => clearInterval(statsInterval);
    }
  }, [selectedChannel, fetchChannelStats]);

  const connectWebSocket = useCallback((channel: string) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${channel}`);
    
    ws.onopen = () => {
      console.log(`WebSocket connected to channel: ${channel}`);
      setIsConnected(true);
      setError('');
    };
    
    ws.onmessage = (event) => {
      const data: RealTimeData = JSON.parse(event.data);
      setRealtimeData(data);
      
      // Update chat messages with new messages
      if (data.recent_messages) {
        setChatMessages(prev => {
          const newMessages = [...data.recent_messages];
          // Keep only the latest 50 messages
          return newMessages.slice(-50);
        });
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error');
      setIsConnected(false);
    };
    
    setWebsocket(ws);
  }, []);

  // Set up WebSocket connection when channel is selected
  useEffect(() => {
    if (selectedChannel && !websocket) {
      connectWebSocket(selectedChannel);
    }
    
    return () => {
      if (websocket) {
        websocket.close();
        setWebsocket(null);
      }
    };
  }, [selectedChannel, websocket, connectWebSocket]);

  const handleChannelChange = (channel: string) => {
    // Close existing WebSocket
    if (websocket) {
      websocket.close();
      setWebsocket(null);
    }
    
    setSelectedChannel(channel);
    setChannelStats(null);
    setRealtimeData(null);
    setHistoricalData([]);
    setChatMessages([]);
  };

  const registerTestChannel = async () => {
    try {
      const testChannel = 'test_channel';
      await axios.post(`${API_BASE_URL}/api/register?channel=${testChannel}`);
      await fetchChannels();
      setSelectedChannel(testChannel);
      setError('');
    } catch (err) {
      setError('Failed to register test channel');
      console.error('Error registering test channel:', err);
    }
  };

  const formatSentiment = (sentiment: number) => {
    if (sentiment > 0.1) return { text: 'Positive', color: '#22c55e' };
    if (sentiment < -0.1) return { text: 'Negative', color: '#ef4444' };
    return { text: 'Neutral', color: '#6b7280' };
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1);
  };

  const formatBurstScore = (score: number) => {
    return score.toFixed(1);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="bg-gray-800 rounded-lg shadow-lg p-6 mb-6 border border-gray-700">
          <h1 className="text-3xl font-bold text-white mb-4">
            Twitch ML Analytics Dashboard
          </h1>
          
          {error && (
            <div className="bg-red-900 border border-red-600 text-red-300 px-4 py-3 rounded mb-4">
              {error}
            </div>
          )}
          
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <label htmlFor="channel-select" className="font-medium text-gray-300">
                Channel:
              </label>
              <select
                id="channel-select"
                value={selectedChannel}
                onChange={(e) => handleChannelChange(e.target.value)}
                className="border border-gray-600 bg-gray-700 text-white rounded-md px-3 py-2"
              >
                <option value="">Select a channel</option>
                {channels.map((channel) => (
                  <option key={channel} value={channel}>
                    {channel}
                  </option>
                ))}
              </select>
            </div>
            
            <button
              onClick={registerTestChannel}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-md font-medium"
            >
              Add Test Channel
            </button>
            
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-300">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>

        {/* System Overview - Simplified */}
        <div className="flex justify-end mb-6">
          <div className="bg-gray-800 rounded-lg shadow-lg p-4 border border-gray-700">
            <h3 className="text-lg font-semibold text-gray-300">Status</h3>
            <p className="text-3xl font-bold text-green-400">Online</p>
          </div>
        </div>

        {/* Main Content Grid */}
        {selectedChannel && channelStats && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Current Stats */}
            <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
              <h2 className="text-xl font-bold text-white mb-4">
                Live Stats - {selectedChannel}
              </h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-400">
                    {(channelStats?.stats?.viewer_count || 0).toLocaleString()}
                  </p>
                  <p className="text-sm text-gray-400">
                    Viewers{(channelStats?.stats?.viewer_count || 0) <= 1 ? ' (offline chat)' : ''}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-green-400">
                    {channelStats?.stats?.raw_velocity || 0}
                  </p>
                  <p className="text-sm text-gray-400">Raw Velocity</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold" style={{ color: formatSentiment(channelStats?.stats?.sentiment || 0).color }}>
                    {formatSentiment(channelStats?.stats?.sentiment || 0).text}
                  </p>
                  <p className="text-sm text-gray-400">Sentiment</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-purple-400">
                    {formatBurstScore(channelStats?.stats?.burst_score || 0)}
                  </p>
                  <p className="text-sm text-gray-400">Burst Score</p>
                </div>
              </div>
              
              <div className="mt-4 pt-4 border-t border-gray-600 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">ML Score:</span>
                  <span className="text-sm font-semibold text-yellow-400">{formatScore(channelStats?.stats?.ml_score || 0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Clip Worthy:</span>
                  <span className="text-sm font-semibold text-orange-400">{formatScore(channelStats?.stats?.clip_worthy_score || 0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Rule Score:</span>
                  <span className="text-sm font-semibold text-cyan-400">{formatScore(channelStats?.stats?.rule_score || 0)}%</span>
                </div>
              </div>
            </div>

            {/* ML Status & Data Info */}
            <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
              <h2 className="text-xl font-bold text-white mb-4">ML Status</h2>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Model Status:</span>
                  <span className={`text-sm font-semibold px-2 py-1 rounded ${
                    channelStats?.ml_status?.model_loaded ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'
                  }`}>
                    {channelStats?.ml_status?.model_status || 'Unknown'}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Training Samples:</span>
                  <span className="text-sm font-semibold text-white">
                    {channelStats?.ml_status?.training_samples || 0} 
                    {channelStats?.ml_status?.samples_used_for_training && 
                     (channelStats?.ml_status?.training_samples || 0) > 200 && 
                     ` (using ${channelStats.ml_status.samples_used_for_training} recent)`}
                  </span>
                </div>
                
                {(channelStats?.ml_status?.training_samples || 0) < 100 && (
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                    style={{ 
                      width: `${Math.min(100, ((channelStats?.ml_status?.training_samples || 0) / 100) * 100)}%` 
                    }}
                  ></div>
                </div>
                )}
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Baseline Samples:</span>
                  <span className="text-sm font-semibold text-white">
                    {channelStats?.ml_status?.current_baseline_count || 0}/{channelStats?.ml_status?.baseline_samples || 0}
                  </span>
                </div>
                
                {((channelStats?.ml_status?.current_baseline_count || 0) < (channelStats?.ml_status?.baseline_samples || 0)) && (
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full transition-all duration-300" 
                    style={{ 
                      width: `${Math.min(100, ((channelStats?.ml_status?.current_baseline_count || 0) / (channelStats?.ml_status?.baseline_samples || 1)) * 100)}%` 
                    }}
                  ></div>
                </div>
                )}
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Total Messages:</span>
                  <span className="text-sm font-semibold text-white">{channelStats?.data_status?.total_messages || 0}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Tracked Users:</span>
                  <span className="text-sm font-semibold text-white">{channelStats?.data_status?.memory_stats?.tracked_users || 0}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-400">Tracked Emotes:</span>
                  <span className="text-sm font-semibold text-white">{channelStats?.data_status?.memory_stats?.tracked_emotes || 0}</span>
                </div>
                
                {(((channelStats?.ml_status?.training_samples || 0) < 100) || ((channelStats?.ml_status?.current_baseline_count || 0) < (channelStats?.ml_status?.baseline_samples || 0))) && (
                  <div className="mt-3 p-2 bg-yellow-900 rounded text-yellow-300 text-xs">
                    ⚠️ ML model needs:
                    {((channelStats?.ml_status?.training_samples || 0) < 100) && (
                      <div>• {100 - (channelStats?.ml_status?.training_samples || 0)} more training samples</div>
                    )}
                    {((channelStats?.ml_status?.current_baseline_count || 0) < (channelStats?.ml_status?.baseline_samples || 0)) && (
                      <div>• {(channelStats?.ml_status?.baseline_samples || 0) - (channelStats?.ml_status?.current_baseline_count || 0)} more baseline samples</div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Live Twitch Chat with 7TV/BTTV/FFZ */}
            <div className="bg-gray-800 rounded-lg shadow-lg border border-gray-700 flex flex-col">
              <div className="p-4 border-b border-gray-600">
                <h2 className="text-xl font-bold text-white">Live Twitch Chat</h2>
                <p className="text-sm text-gray-400">Channel: {selectedChannel} • 7TV/BTTV/FFZ Enabled</p>
              </div>
              <div className="flex-1 p-4">
                <div className="h-96 rounded-lg overflow-hidden bg-black">
                  {selectedChannel ? (
                    <iframe
                      src={`https://chatis.is2511.com/v2/?channel=${selectedChannel}&size=1&font=11&shadow=1&7tv=true&bttv=true&ffz=true&hideBots=true&hideCommands=true&theme=dark`}
                      height="100%"
                      width="100%"
                      style={{ border: 'none', borderRadius: '0.5rem' }}
                      title={`${selectedChannel} ChatIS with 7TV`}
                    />
                  ) : (
                    <div className="text-center text-gray-500 py-8 h-full flex items-center justify-center">
                      <p>Select a channel to view chat</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Improved Charts */}
        {historicalData.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* Real-time Activity Monitor */}
            <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
              <h2 className="text-xl font-bold text-white mb-4">Real-time Activity</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historicalData.slice(-10)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151', 
                      borderRadius: '0.5rem',
                      color: '#F9FAFB'
                    }} 
                  />
                  <Legend />
                  <Line type="monotone" dataKey="raw_velocity" stroke="#10B981" name="Chat Velocity" strokeWidth={3} />
                  <Line type="monotone" dataKey="burst_score" stroke="#8B5CF6" name="Burst Score" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Clip Prediction Tracker */}
            <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
              <h2 className="text-xl font-bold text-white mb-4">Clip Prediction</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historicalData.slice(-10)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" stroke="#9CA3AF" />
                  <YAxis domain={[0, 100]} stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151', 
                      borderRadius: '0.5rem',
                      color: '#F9FAFB'
                    }}
                    formatter={(value) => [`${value}%`, '']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="clip_worthy_score" stroke="#F59E0B" name="Clip Worthy %" strokeWidth={3} />
                  <Line type="monotone" dataKey="ml_score" stroke="#EF4444" name="ML Score %" strokeWidth={2} />
                  <Line type="monotone" dataKey="rule_score" stroke="#06B6D4" name="Rule Score %" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Sentiment Timeline */}
        {historicalData.length > 0 && (
          <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700 mb-6">
            <h2 className="text-xl font-bold text-white mb-4">Chat Sentiment Timeline</h2>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={historicalData.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis domain={[-1, 1]} stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151', 
                    borderRadius: '0.5rem',
                    color: '#F9FAFB'
                  }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="sentiment" 
                  stroke="#84CC16" 
                  name="Sentiment" 
                  strokeWidth={3}
                  dot={{ fill: '#84CC16', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard; 