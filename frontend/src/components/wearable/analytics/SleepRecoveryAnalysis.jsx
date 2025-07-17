import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { useWearableData } from '../../../hooks/useWearableAnalytics';
import { useTimeSeriesConfig, usePieChartConfig } from '../../../hooks/useChartConfig';

const SleepRecoveryAnalysis = ({ patientId, data: analyticsData }) => {
  const [selectedView, setSelectedView] = useState('quality');
  const [timeRange, setTimeRange] = useState('30');
  
  const { data: wearableData, loading, error } = useWearableData(patientId);
  const lineConfig = useTimeSeriesConfig(['sleep_efficiency', 'deep_sleep', 'rem_sleep']);
  const pieConfig = usePieChartConfig();

  const sleepData = useMemo(() => {
    if (!wearableData || wearableData.length === 0) return [];
    
    return wearableData.map(item => ({
      date: new Date(item.date).toLocaleDateString(),
      total_sleep: item.total_sleep_minutes || 0,
      sleep_efficiency: item.sleep_efficiency || 0,
      deep_sleep: item.deep_sleep_minutes || 0,
      light_sleep: item.light_sleep_minutes || 0,
      rem_sleep: item.rem_sleep_minutes || 0,
      awake_time: item.awake_minutes || 0,
      sleep_score: item.sleep_score || 0,
      bedtime: item.bedtime,
      wake_time: item.wake_time,
    }));
  }, [wearableData]);

  const sleepSummary = useMemo(() => {
    if (!sleepData.length) return null;
    
    const recent = sleepData.slice(-7);
    const avgSleep = recent.reduce((sum, night) => sum + night.total_sleep, 0) / recent.length;
    const avgEfficiency = recent.reduce((sum, night) => sum + night.sleep_efficiency, 0) / recent.length;
    const avgDeepSleep = recent.reduce((sum, night) => sum + night.deep_sleep, 0) / recent.length;
    const avgScore = recent.reduce((sum, night) => sum + night.sleep_score, 0) / recent.length;
    
    return {
      avgSleep: avgSleep / 60, // Convert to hours
      avgEfficiency: avgEfficiency * 100,
      avgDeepSleep: avgDeepSleep,
      avgScore: avgScore,
      trend: sleepData[sleepData.length - 1].sleep_score > sleepData[sleepData.length - 7].sleep_score ? 'improving' : 'declining',
    };
  }, [sleepData]);

  const sleepStageDistribution = useMemo(() => {
    if (!sleepSummary) return [];
    
    const recent = sleepData.slice(-7);
    const totalSleep = recent.reduce((sum, night) => sum + night.total_sleep, 0);
    const totalDeep = recent.reduce((sum, night) => sum + night.deep_sleep, 0);
    const totalLight = recent.reduce((sum, night) => sum + night.light_sleep, 0);
    const totalRem = recent.reduce((sum, night) => sum + night.rem_sleep, 0);
    const totalAwake = recent.reduce((sum, night) => sum + night.awake_time, 0);
    
    return [
      { name: 'Deep Sleep', value: totalDeep, color: '#1e40af' },
      { name: 'Light Sleep', value: totalLight, color: '#3b82f6' },
      { name: 'REM Sleep', value: totalRem, color: '#60a5fa' },
      { name: 'Awake', value: totalAwake, color: '#f59e0b' },
    ];
  }, [sleepData, sleepSummary]);

  const getSleepQuality = (score) => {
    if (score >= 80) return { quality: 'Excellent', color: 'text-green-600' };
    if (score >= 70) return { quality: 'Good', color: 'text-blue-600' };
    if (score >= 60) return { quality: 'Fair', color: 'text-yellow-600' };
    return { quality: 'Poor', color: 'text-red-600' };
  };

  const sleepQuality = getSleepQuality(sleepSummary?.avgScore || 0);

  const getOptimalBedtime = () => {
    if (!sleepData.length) return null;
    
    const bedtimes = sleepData.filter(d => d.bedtime).map(d => {
      const time = new Date(`2000-01-01T${d.bedtime}`);
      return time.getHours() + time.getMinutes() / 60;
    });
    
    if (bedtimes.length === 0) return null;
    
    const avgBedtime = bedtimes.reduce((sum, time) => sum + time, 0) / bedtimes.length;
    const hours = Math.floor(avgBedtime);
    const minutes = Math.round((avgBedtime - hours) * 60);
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
  };

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
          <h3 className="text-lg font-semibold text-gray-700">Sleep Recovery Analysis</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedView}
              onChange={(e) => setSelectedView(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="quality">Sleep Quality</option>
              <option value="stages">Sleep Stages</option>
              <option value="efficiency">Sleep Efficiency</option>
              <option value="pattern">Sleep Pattern</option>
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
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Sleep Quality</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-blue-600">
                {sleepSummary?.avgScore?.toFixed(0) || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">/100</span>
              <div className={`text-sm ${sleepQuality.color} mt-1`}>
                {sleepQuality.quality}
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Avg Sleep Duration</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-green-600">
                {sleepSummary?.avgSleep?.toFixed(1) || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">hours</span>
              <div className="text-sm text-gray-600 mt-1">
                Target: 7-9 hours
              </div>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Sleep Efficiency</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-purple-600">
                {sleepSummary?.avgEfficiency?.toFixed(0) || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">%</span>
              <div className="text-sm text-gray-600 mt-1">
                Target: >85%
              </div>
            </div>
          </div>

          <div className="bg-indigo-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Deep Sleep</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-indigo-600">
                {sleepSummary?.avgDeepSleep?.toFixed(0) || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">min</span>
              <div className="text-sm text-gray-600 mt-1">
                {sleepSummary?.trend === 'improving' ? '↗ Improving' : '↘ Declining'}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              {selectedView === 'quality' && (
                <LineChart data={sleepData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  
                  <Line
                    type="monotone"
                    dataKey="sleep_score"
                    stroke={lineConfig.colors.primary}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="Sleep Score"
                  />
                </LineChart>
              )}
              
              {selectedView === 'stages' && (
                <BarChart data={sleepData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  
                  <Bar dataKey="deep_sleep" stackId="a" fill="#1e40af" name="Deep Sleep" />
                  <Bar dataKey="light_sleep" stackId="a" fill="#3b82f6" name="Light Sleep" />
                  <Bar dataKey="rem_sleep" stackId="a" fill="#60a5fa" name="REM Sleep" />
                  <Bar dataKey="awake_time" stackId="a" fill="#f59e0b" name="Awake Time" />
                </BarChart>
              )}
              
              {selectedView === 'efficiency' && (
                <LineChart data={sleepData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  
                  <Line
                    type="monotone"
                    dataKey="sleep_efficiency"
                    stroke={lineConfig.colors.secondary}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="Sleep Efficiency"
                  />
                </LineChart>
              )}
              
              {selectedView === 'pattern' && (
                <LineChart data={sleepData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  
                  <Line
                    type="monotone"
                    dataKey="total_sleep"
                    stroke={lineConfig.colors.primary}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="Total Sleep (minutes)"
                  />
                </LineChart>
              )}
            </ResponsiveContainer>
          </div>

          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sleepStageDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {sleepStageDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Sleep Pattern Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Sleep Timing</h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Optimal Bedtime</span>
                <span className="text-sm font-medium text-blue-600">
                  {getOptimalBedtime() || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Sleep Consistency</span>
                <span className="text-sm font-medium text-green-600">
                  {sleepSummary ? 'Good' : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Recovery Quality</span>
                <span className="text-sm font-medium text-purple-600">
                  {sleepQuality.quality}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Sleep Recommendations</h4>
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className="text-blue-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  {sleepSummary?.avgSleep < 7 
                    ? 'Increase sleep duration by going to bed 30 minutes earlier'
                    : 'Maintain current sleep duration'}
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-green-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  {sleepSummary?.avgEfficiency < 85 
                    ? 'Improve sleep efficiency by reducing time in bed awake'
                    : 'Excellent sleep efficiency - keep it up!'}
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-purple-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  Focus on consistent sleep schedule for optimal recovery
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {analyticsData?.sleep_insights && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Sleep & Recovery Insights</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">Recovery Correlation</h4>
              <div className="space-y-3">
                {analyticsData.sleep_insights.recovery_correlation?.map((insight, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <div>
                      <p className="text-sm text-gray-700">{insight.finding}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Correlation: {insight.correlation_strength}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">Sleep Quality Factors</h4>
              <div className="space-y-3">
                {analyticsData.sleep_insights.quality_factors?.map((factor, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                    <div>
                      <p className="text-sm text-gray-700">{factor.factor}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Impact: {factor.impact} | Recommendation: {factor.recommendation}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SleepRecoveryAnalysis;