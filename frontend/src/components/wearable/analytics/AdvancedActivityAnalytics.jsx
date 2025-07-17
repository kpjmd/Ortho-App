import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, AreaChart, Area } from 'recharts';
import { useWearableData } from '../../../hooks/useWearableAnalytics';
import { useTimeSeriesConfig, useBarChartConfig } from '../../../hooks/useChartConfig';

const AdvancedActivityAnalytics = ({ patientId, data: analyticsData }) => {
  const [selectedMetric, setSelectedMetric] = useState('steps');
  const [timeRange, setTimeRange] = useState('30');
  const [showTrendLine, setShowTrendLine] = useState(true);
  
  const { data: wearableData, loading, error } = useWearableData(patientId);
  const lineConfig = useTimeSeriesConfig(['value', 'trend', 'target']);
  const barConfig = useBarChartConfig(['value']);

  const processedData = useMemo(() => {
    if (!wearableData || wearableData.length === 0) return [];
    
    return wearableData.map(item => ({
      date: new Date(item.date).toLocaleDateString(),
      steps: item.steps || 0,
      distance: item.distance || 0,
      calories: item.calories_burned || 0,
      active_minutes: item.active_minutes || 0,
      sedentary_minutes: item.sedentary_minutes || 0,
      walking_speed: item.walking_speed_ms || 0,
      elevation_gain: item.elevation_gain || 0,
    }));
  }, [wearableData]);

  const activitySummary = useMemo(() => {
    if (!processedData.length) return null;
    
    const recent = processedData.slice(-7);
    const avgSteps = recent.reduce((sum, day) => sum + day.steps, 0) / recent.length;
    const avgDistance = recent.reduce((sum, day) => sum + day.distance, 0) / recent.length;
    const avgCalories = recent.reduce((sum, day) => sum + day.calories, 0) / recent.length;
    const avgActiveMinutes = recent.reduce((sum, day) => sum + day.active_minutes, 0) / recent.length;
    
    return {
      avgSteps: Math.round(avgSteps),
      avgDistance: avgDistance.toFixed(1),
      avgCalories: Math.round(avgCalories),
      avgActiveMinutes: Math.round(avgActiveMinutes),
      trend: processedData[processedData.length - 1].steps > processedData[processedData.length - 7].steps ? 'increasing' : 'decreasing',
    };
  }, [processedData]);

  const activityPatterns = useMemo(() => {
    if (!processedData.length) return [];
    
    const dayOfWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const patterns = Array(7).fill().map((_, i) => ({
      day: dayOfWeek[i],
      steps: 0,
      active_minutes: 0,
      count: 0,
    }));
    
    processedData.forEach(item => {
      const date = new Date(item.date);
      const dayIndex = date.getDay();
      patterns[dayIndex].steps += item.steps;
      patterns[dayIndex].active_minutes += item.active_minutes;
      patterns[dayIndex].count += 1;
    });
    
    return patterns.map(pattern => ({
      ...pattern,
      steps: pattern.count > 0 ? Math.round(pattern.steps / pattern.count) : 0,
      active_minutes: pattern.count > 0 ? Math.round(pattern.active_minutes / pattern.count) : 0,
    }));
  }, [processedData]);

  const getActivityScore = () => {
    if (!activitySummary) return 0;
    
    const stepsScore = Math.min(activitySummary.avgSteps / 10000, 1) * 30;
    const activeMinutesScore = Math.min(activitySummary.avgActiveMinutes / 150, 1) * 30;
    const consistencyScore = 40; // Placeholder for consistency calculation
    
    return Math.round(stepsScore + activeMinutesScore + consistencyScore);
  };

  const getActivityLevel = (score) => {
    if (score >= 80) return { level: 'Excellent', color: 'text-green-600' };
    if (score >= 60) return { level: 'Good', color: 'text-blue-600' };
    if (score >= 40) return { level: 'Fair', color: 'text-yellow-600' };
    return { level: 'Poor', color: 'text-red-600' };
  };

  const activityScore = getActivityScore();
  const activityLevel = getActivityLevel(activityScore);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <span className="text-red-600 mr-2">⚠️</span>
          <span className="text-red-800">{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-700">Advanced Activity Analytics</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="steps">Daily Steps</option>
              <option value="distance">Distance</option>
              <option value="calories">Calories Burned</option>
              <option value="active_minutes">Active Minutes</option>
              <option value="walking_speed">Walking Speed</option>
            </select>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="7">Last 7 days</option>
              <option value="30">Last 30 days</option>
              <option value="90">Last 90 days</option>
            </select>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showTrendLine}
                onChange={(e) => setShowTrendLine(e.target.checked)}
                className="mr-2"
              />
              Show Trend
            </label>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Activity Score</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-blue-600">{activityScore}</span>
              <span className="text-sm text-gray-600 ml-1">/100</span>
              <div className={`text-sm ${activityLevel.color} mt-1`}>
                {activityLevel.level}
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Avg Daily Steps</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-green-600">
                {activitySummary?.avgSteps.toLocaleString() || 'N/A'}
              </span>
              <div className="text-sm text-gray-600 mt-1">
                {activitySummary?.trend === 'increasing' ? '↗ Increasing' : '↘ Decreasing'}
              </div>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Active Minutes</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-purple-600">
                {activitySummary?.avgActiveMinutes || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">min/day</span>
              <div className="text-sm text-gray-600 mt-1">
                Goal: 150 min/week
              </div>
            </div>
          </div>

          <div className="bg-orange-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Calories Burned</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-orange-600">
                {activitySummary?.avgCalories || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">cal/day</span>
              <div className="text-sm text-gray-600 mt-1">
                Distance: {activitySummary?.avgDistance || 'N/A'} km
              </div>
            </div>
          </div>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              
              <Line
                type="monotone"
                dataKey={selectedMetric}
                stroke={lineConfig.colors.primary}
                strokeWidth={2}
                dot={{ r: 4 }}
                name={selectedMetric.replace('_', ' ').toUpperCase()}
              />
              
              {showTrendLine && (
                <Line
                  type="monotone"
                  dataKey={selectedMetric}
                  stroke={lineConfig.colors.secondary}
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Trend"
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Weekly Activity Patterns</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={activityPatterns}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              
              <Bar dataKey="steps" fill={barConfig.colors.primary} name="Average Steps" />
              <Bar dataKey="active_minutes" fill={barConfig.colors.secondary} name="Active Minutes" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {analyticsData?.activity_insights && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Activity Insights</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">Movement Patterns</h4>
              <div className="space-y-3">
                {analyticsData.activity_insights.movement_patterns?.map((pattern, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <div>
                      <p className="text-sm text-gray-700">{pattern.observation}</p>
                      <p className="text-xs text-gray-500 mt-1">{pattern.recommendation}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">Recovery Indicators</h4>
              <div className="space-y-3">
                {analyticsData.activity_insights.recovery_indicators?.map((indicator, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                    <div>
                      <p className="text-sm text-gray-700">{indicator.metric}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Status: {indicator.status} | Trend: {indicator.trend}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Activity Goals & Recommendations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Current Goals</h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Daily Steps</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${Math.min((activitySummary?.avgSteps || 0) / 10000 * 100, 100)}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500">10,000</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Active Minutes</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${Math.min((activitySummary?.avgActiveMinutes || 0) / 150 * 100, 100)}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500">150/week</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Recommendations</h4>
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className="text-blue-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  {activitySummary?.avgSteps < 5000 
                    ? 'Gradually increase daily steps by 500-1000 steps per week'
                    : 'Maintain current activity level and focus on consistency'}
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-green-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  {activitySummary?.avgActiveMinutes < 150 
                    ? 'Add 10-15 minutes of moderate activity daily'
                    : 'Consider adding strength training exercises'}
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-purple-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  Track walking speed improvements as a recovery indicator
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedActivityAnalytics;