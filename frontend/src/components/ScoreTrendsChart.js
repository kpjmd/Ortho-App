import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import { 
  formatChartDate, 
  getKOOSColors, 
  getASESColors, 
  getKOOSDisplayNames, 
  getASESDisplayNames,
  SkeletonLoader 
} from '../utils/proScoreHelpers';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ScoreTrendsChart = ({ patientId, patientType }) => {
  const [trendsData, setTrendsData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchTrendsData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const endpoint = patientType === 'knee' ? 
          `${API}/koos/${patientId}/trends` : 
          `${API}/ases/${patientId}/trends`;
        
        const response = await axios.get(endpoint);
        const trends = response.data.trends || [];
        
        // Format data for chart
        const formattedData = trends.map(item => ({
          ...item,
          date: formatChartDate(item.date)
        }));
        
        setTrendsData(formattedData);
      } catch (err) {
        console.error('Failed to fetch trends data:', err);
        setError('Failed to load trends data');
      } finally {
        setLoading(false);
      }
    };

    if (patientId && patientType) {
      fetchTrendsData();
    }
  }, [patientId, patientType]);

  const colors = patientType === 'knee' ? getKOOSColors() : getASESColors();
  const displayNames = patientType === 'knee' ? getKOOSDisplayNames() : getASESDisplayNames();

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-700 mb-2">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {displayNames[entry.dataKey]}: {Math.round(entry.value)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-700">Score Trends</h3>
        </div>
        <div className="space-y-4">
          <SkeletonLoader width={100} height={300} />
          <div className="flex justify-center space-x-6">
            <SkeletonLoader width={15} height={20} />
            <SkeletonLoader width={15} height={20} />
            <SkeletonLoader width={15} height={20} />
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-700">Score Trends</h3>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
          <p className="text-red-700">{error}</p>
        </div>
      </div>
    );
  }

  if (trendsData.length < 2) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-700">Score Trends</h3>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
          <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
          </svg>
          <p className="text-gray-600 text-lg font-medium">No trend data available</p>
          <p className="text-gray-500 text-sm mt-2">
            Complete at least 2 PRO assessments to see trend charts
          </p>
        </div>
      </div>
    );
  }

  // Render lines based on patient type
  const renderLines = () => {
    if (patientType === 'knee') {
      return (
        <>
          <Line
            type="monotone"
            dataKey="symptoms_score"
            stroke={colors.symptoms_score}
            strokeWidth={2}
            dot={{ fill: colors.symptoms_score, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
            name={displayNames.symptoms_score}
          />
          <Line
            type="monotone"
            dataKey="pain_score"
            stroke={colors.pain_score}
            strokeWidth={2}
            dot={{ fill: colors.pain_score, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
            name={displayNames.pain_score}
          />
          <Line
            type="monotone"
            dataKey="adl_score"
            stroke={colors.adl_score}
            strokeWidth={2}
            dot={{ fill: colors.adl_score, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
            name={displayNames.adl_score}
          />
          <Line
            type="monotone"
            dataKey="sport_score"
            stroke={colors.sport_score}
            strokeWidth={2}
            dot={{ fill: colors.sport_score, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
            name={displayNames.sport_score}
          />
          <Line
            type="monotone"
            dataKey="qol_score"
            stroke={colors.qol_score}
            strokeWidth={2}
            dot={{ fill: colors.qol_score, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
            name={displayNames.qol_score}
          />
        </>
      );
    } else {
      return (
        <>
          <Line
            type="monotone"
            dataKey="total_score"
            stroke={colors.total_score}
            strokeWidth={3}
            dot={{ fill: colors.total_score, strokeWidth: 2, r: 5 }}
            activeDot={{ r: 7 }}
            name={displayNames.total_score}
          />
          <Line
            type="monotone"
            dataKey="pain_component"
            stroke={colors.pain_component}
            strokeWidth={2}
            dot={{ fill: colors.pain_component, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
            name={displayNames.pain_component}
          />
          <Line
            type="monotone"
            dataKey="function_component"
            stroke={colors.function_component}
            strokeWidth={2}
            dot={{ fill: colors.function_component, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
            name={displayNames.function_component}
          />
        </>
      );
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-700">Score Trends</h3>
        <span className="text-sm text-gray-500">
          {trendsData.length} assessment{trendsData.length !== 1 ? 's' : ''}
        </span>
      </div>
      
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={trendsData}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="date" 
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              domain={[0, 100]}
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              wrapperStyle={{
                paddingTop: '20px',
                fontSize: '12px'
              }}
            />
            {renderLines()}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ScoreTrendsChart;