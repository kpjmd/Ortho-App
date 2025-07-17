import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart } from 'recharts';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const RecoveryTrajectoryChart = ({ patientId }) => {
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSubscale, setSelectedSubscale] = useState('total_score');

  useEffect(() => {
    const fetchTrajectoryData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await axios.get(`${API}/recovery-trajectory/${patientId}`);
        setTrajectoryData(response.data);
        
        // Set default subscale based on body part
        if (response.data.body_part === 'KNEE') {
          setSelectedSubscale('total_score');
        } else if (response.data.body_part === 'SHOULDER') {
          setSelectedSubscale('total_score');
        }
        
      } catch (err) {
        console.error('Failed to fetch trajectory data:', err);
        setError('Failed to load recovery trajectory data');
      } finally {
        setLoading(false);
      }
    };

    if (patientId) {
      fetchTrajectoryData();
    }
  }, [patientId]);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-red-600 text-center">
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!trajectoryData) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-gray-600 text-center">
          <p>No trajectory data available</p>
        </div>
      </div>
    );
  }

  // Prepare chart data by combining trajectory and actual scores
  const prepareChartData = () => {
    const chartData = [];
    const expectedData = trajectoryData.trajectory_data[selectedSubscale] || [];
    const actualData = trajectoryData.actual_scores || [];
    
    // Create a map of actual scores by week
    const actualScoresByWeek = {};
    actualData.forEach(score => {
      actualScoresByWeek[score.week] = score[selectedSubscale] || 0;
    });
    
    // Combine expected and actual data
    const allWeeks = new Set([
      ...expectedData.map(d => d.week),
      ...actualData.map(d => d.week)
    ]);
    
    Array.from(allWeeks).sort((a, b) => a - b).forEach(week => {
      const expectedPoint = expectedData.find(d => d.week === week);
      const actualScore = actualScoresByWeek[week];
      
      if (expectedPoint || actualScore !== undefined) {
        chartData.push({
          week,
          expected: expectedPoint ? expectedPoint.expected_score : null,
          upperBound: expectedPoint ? expectedPoint.upper_bound : null,
          lowerBound: expectedPoint ? expectedPoint.lower_bound : null,
          actual: actualScore !== undefined ? actualScore : null
        });
      }
    });
    
    return chartData;
  };

  const chartData = prepareChartData();
  
  // Get available subscales based on body part
  const getAvailableSubscales = () => {
    if (trajectoryData.body_part === 'KNEE') {
      return [
        { value: 'total_score', label: 'Total Score' },
        { value: 'symptoms_score', label: 'Symptoms' },
        { value: 'pain_score', label: 'Pain' },
        { value: 'adl_score', label: 'Daily Activities' },
        { value: 'sport_score', label: 'Sports/Recreation' },
        { value: 'qol_score', label: 'Quality of Life' }
      ];
    } else {
      return [
        { value: 'total_score', label: 'Total Score' },
        { value: 'pain_component', label: 'Pain Component' },
        { value: 'function_component', label: 'Function Component' }
      ];
    }
  };

  const subscaleOptions = getAvailableSubscales();
  const currentSubscaleLabel = subscaleOptions.find(s => s.value === selectedSubscale)?.label || 'Score';

  // Get milestones for current subscale
  const relevantMilestones = trajectoryData.milestones.filter(
    milestone => milestone.subscale === selectedSubscale || milestone.subscale === 'total_score'
  );

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold">{`Week ${label}`}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.dataKey === 'actual' ? 'Actual' : 'Expected'}: ${entry.value?.toFixed(1) || 'N/A'}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Get status color based on trajectory
  const getStatusColor = (actual, expected, upperBound, lowerBound) => {
    if (!actual || !expected) return 'text-gray-500';
    
    if (actual >= upperBound) return 'text-green-600';
    if (actual >= expected) return 'text-blue-600';
    if (actual >= lowerBound) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getStatusText = (actual, expected, upperBound, lowerBound) => {
    if (!actual || !expected) return 'No data';
    
    if (actual >= upperBound) return 'Ahead of Schedule';
    if (actual >= expected) return 'On Track';
    if (actual >= lowerBound) return 'Slightly Behind';
    return 'Behind Schedule';
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-xl font-semibold text-gray-800">Recovery Trajectory</h3>
        <select
          value={selectedSubscale}
          onChange={(e) => setSelectedSubscale(e.target.value)}
          className="px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {subscaleOptions.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {/* Status indicator */}
      <div className="mb-4 p-3 bg-gray-50 rounded">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Current Status:</span>
          <span className={`font-semibold ${getStatusColor(
            chartData[chartData.length - 1]?.actual,
            chartData[chartData.length - 1]?.expected,
            chartData[chartData.length - 1]?.upperBound,
            chartData[chartData.length - 1]?.lowerBound
          )}`}>
            {getStatusText(
              chartData[chartData.length - 1]?.actual,
              chartData[chartData.length - 1]?.expected,
              chartData[chartData.length - 1]?.upperBound,
              chartData[chartData.length - 1]?.lowerBound
            )}
          </span>
        </div>
        <div className="text-sm text-gray-600 mt-1">
          {trajectoryData.weeks_post_surgery} weeks post-surgery • {trajectoryData.diagnosis}
        </div>
      </div>

      {/* Chart */}
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="week" 
              label={{ value: 'Weeks Post-Surgery', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              label={{ value: currentSubscaleLabel, angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {/* Expected recovery corridor (shaded area) */}
            <defs>
              <linearGradient id="corridorGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.1}/>
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.05}/>
              </linearGradient>
            </defs>
            
            {/* Upper bound line */}
            <Line 
              type="monotone" 
              dataKey="upperBound" 
              stroke="#93C5FD" 
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
              name="Upper Bound"
            />
            
            {/* Lower bound line */}
            <Line 
              type="monotone" 
              dataKey="lowerBound" 
              stroke="#93C5FD" 
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
              name="Lower Bound"
            />
            
            {/* Expected trajectory line */}
            <Line 
              type="monotone" 
              dataKey="expected" 
              stroke="#3B82F6" 
              strokeWidth={2}
              dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
              name="Expected Recovery"
            />
            
            {/* Actual patient scores */}
            <Line 
              type="monotone" 
              dataKey="actual" 
              stroke="#EF4444" 
              strokeWidth={3}
              dot={{ fill: '#EF4444', strokeWidth: 2, r: 6 }}
              name="Actual Progress"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Milestones */}
      {relevantMilestones.length > 0 && (
        <div className="mt-6">
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Recovery Milestones</h4>
          <div className="space-y-2">
            {relevantMilestones.map((milestone, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    milestone.achieved ? 'bg-green-500' : 
                    milestone.week <= trajectoryData.weeks_post_surgery ? 'bg-red-500' : 'bg-gray-300'
                  }`}></div>
                  <div>
                    <span className="font-medium text-gray-800">Week {milestone.week}</span>
                    {milestone.critical && (
                      <span className="ml-2 px-2 py-1 bg-red-100 text-red-800 text-xs rounded">
                        Critical
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  {milestone.description}
                </div>
                <div className="text-sm font-medium">
                  {milestone.achieved ? (
                    <span className="text-green-600">✓ Achieved</span>
                  ) : milestone.week <= trajectoryData.weeks_post_surgery ? (
                    <span className="text-red-600">✗ Missed</span>
                  ) : (
                    <span className="text-gray-500">Pending</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="mt-4 p-3 bg-gray-50 rounded">
        <div className="text-sm text-gray-600">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <strong>Blue shaded area:</strong> Expected recovery corridor
            </div>
            <div>
              <strong>Red line:</strong> Your actual progress
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecoveryTrajectoryChart;