import React, { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useTimeSeriesConfig } from '../../../hooks/useChartConfig';

const RecoveryVelocityChart = ({ patientId, data }) => {
  const [selectedMetric, setSelectedMetric] = useState('overall');
  const [timeRange, setTimeRange] = useState('30');
  const [showConfidenceInterval, setShowConfidenceInterval] = useState(true);
  
  const chartConfig = useTimeSeriesConfig(['velocity', 'predicted', 'confidence_upper', 'confidence_lower']);

  const velocityData = useMemo(() => {
    if (!data || !data.velocity_trends) return [];
    
    return data.velocity_trends.map(item => ({
      date: new Date(item.date).toLocaleDateString(),
      velocity: item.velocity_score,
      predicted: item.predicted_velocity,
      confidence_upper: item.confidence_upper,
      confidence_lower: item.confidence_lower,
    }));
  }, [data]);

  const metricData = useMemo(() => {
    if (!data || !data.metric_velocities) return [];
    
    const metrics = selectedMetric === 'overall' ? 
      Object.keys(data.metric_velocities) :
      [selectedMetric];
    
    return data.velocity_trends.map(item => {
      const dataPoint = { date: new Date(item.date).toLocaleDateString() };
      
      metrics.forEach(metric => {
        if (data.metric_velocities[metric]) {
          const metricData = data.metric_velocities[metric].find(m => m.date === item.date);
          if (metricData) {
            dataPoint[metric] = metricData.velocity;
          }
        }
      });
      
      return dataPoint;
    });
  }, [data, selectedMetric]);

  const getVelocityStatus = (velocity) => {
    if (velocity >= 0.8) return { status: 'Excellent', color: 'text-green-600' };
    if (velocity >= 0.6) return { status: 'Good', color: 'text-blue-600' };
    if (velocity >= 0.4) return { status: 'Fair', color: 'text-yellow-600' };
    return { status: 'Poor', color: 'text-red-600' };
  };

  const currentVelocity = data?.overall_velocity_score || 0;
  const velocityStatus = getVelocityStatus(currentVelocity);

  if (!data) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">No velocity data available</div>
        </div>
      </div>
    );
  }

  const availableMetrics = data.metric_velocities ? Object.keys(data.metric_velocities) : [];

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-700">Recovery Velocity Analysis</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="overall">Overall Velocity</option>
              {availableMetrics.map(metric => (
                <option key={metric} value={metric}>{metric.replace('_', ' ').toUpperCase()}</option>
              ))}
            </select>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="7">Last 7 days</option>
              <option value="30">Last 30 days</option>
              <option value="90">Last 90 days</option>
              <option value="all">All time</option>
            </select>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showConfidenceInterval}
                onChange={(e) => setShowConfidenceInterval(e.target.checked)}
                className="mr-2"
              />
              Show Confidence Bands
            </label>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Current Velocity</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-blue-600">{(currentVelocity * 100).toFixed(1)}%</span>
              <div className={`text-sm ${velocityStatus.color} mt-1`}>
                {velocityStatus.status}
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Velocity Trend</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-green-600">
                {data.velocity_trend_direction === 'increasing' ? '↗' : 
                 data.velocity_trend_direction === 'decreasing' ? '↘' : '→'}
              </span>
              <div className="text-sm text-gray-600 mt-1">
                {data.velocity_trend_direction || 'Stable'}
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Progress Rate</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-purple-600">
                {data.progress_rate ? `${(data.progress_rate * 100).toFixed(1)}%` : 'N/A'}
              </span>
              <div className="text-sm text-gray-600 mt-1">per week</div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Plateau Risk</h4>
            <div className="mt-2">
              <span className={`text-2xl font-bold ${
                data.plateau_risk === 'high' ? 'text-red-600' :
                data.plateau_risk === 'medium' ? 'text-yellow-600' :
                'text-green-600'
              }`}>
                {data.plateau_risk ? data.plateau_risk.toUpperCase() : 'LOW'}
              </span>
            </div>
          </div>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            {selectedMetric === 'overall' ? (
              <AreaChart data={velocityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                
                {showConfidenceInterval && (
                  <Area
                    dataKey="confidence_upper"
                    stackId="1"
                    stroke="none"
                    fill={chartConfig.colors.primary}
                    fillOpacity={0.1}
                  />
                )}
                
                {showConfidenceInterval && (
                  <Area
                    dataKey="confidence_lower"
                    stackId="1"
                    stroke="none"
                    fill="white"
                    fillOpacity={1}
                  />
                )}
                
                <Line
                  type="monotone"
                  dataKey="velocity"
                  stroke={chartConfig.colors.primary}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Actual Velocity"
                />
                
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke={chartConfig.colors.secondary}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ r: 4 }}
                  name="Predicted Velocity"
                />
              </AreaChart>
            ) : (
              <LineChart data={metricData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                
                <Line
                  type="monotone"
                  dataKey={selectedMetric}
                  stroke={chartConfig.colors.primary}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name={selectedMetric.replace('_', ' ').toUpperCase()}
                />
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>

      {data.velocity_insights && data.velocity_insights.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Velocity Insights</h3>
          <div className="space-y-4">
            {data.velocity_insights.map((insight, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                <div>
                  <p className="text-sm text-gray-700">{insight.message}</p>
                  {insight.recommendation && (
                    <p className="text-xs text-gray-500 mt-1">
                      Recommendation: {insight.recommendation}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.velocity_milestones && data.velocity_milestones.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Recovery Milestones</h3>
          <div className="space-y-3">
            {data.velocity_milestones.map((milestone, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    milestone.achieved ? 'bg-green-500' : 'bg-gray-300'
                  }`}></div>
                  <span className="text-sm font-medium text-gray-700">{milestone.milestone}</span>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-600">
                    {milestone.achieved ? 'Achieved' : `${milestone.progress}% complete`}
                  </div>
                  {milestone.expected_date && (
                    <div className="text-xs text-gray-500">
                      Expected: {new Date(milestone.expected_date).toLocaleDateString()}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default RecoveryVelocityChart;