import React, { useState, useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, BarChart, Bar, Cell } from 'recharts';
import { useCorrelationAnalysis } from '../../../hooks/useWearableAnalytics';
import { useChartConfig } from '../../../hooks/useChartConfig';

const CorrelationAnalysisView = ({ patientId, data: initialData }) => {
  const [selectedCorrelation, setSelectedCorrelation] = useState('activity_pro');
  const [selectedPROType, setSelectedPROType] = useState('total_score');
  const [timeWindow, setTimeWindow] = useState('7');
  
  const { data: fetchedData, loading, error } = useCorrelationAnalysis(patientId);
  const data = initialData || fetchedData;
  const chartConfig = useChartConfig();

  const correlationData = useMemo(() => {
    if (!data || !data.correlations) return [];
    
    const correlations = data.correlations[selectedCorrelation];
    if (!correlations) return [];
    
    return correlations.map(item => ({
      wearable_value: item.wearable_metric_value,
      pro_score: item.pro_score_value,
      date: new Date(item.date).toLocaleDateString(),
      correlation_strength: item.correlation_coefficient,
    }));
  }, [data, selectedCorrelation]);

  const correlationStrength = useMemo(() => {
    if (!data || !data.correlation_summary) return {};
    
    return data.correlation_summary.reduce((acc, item) => {
      acc[item.correlation_type] = {
        strength: item.correlation_coefficient,
        significance: item.p_value < 0.05 ? 'significant' : 'not_significant',
        direction: item.correlation_coefficient > 0 ? 'positive' : 'negative',
      };
      return acc;
    }, {});
  }, [data]);

  const timeSeriesCorrelation = useMemo(() => {
    if (!data || !data.time_series_correlations) return [];
    
    return data.time_series_correlations.map(item => ({
      date: new Date(item.date).toLocaleDateString(),
      correlation: item.correlation_coefficient,
      significance: item.p_value < 0.05 ? 'significant' : 'not_significant',
    }));
  }, [data]);

  const getCorrelationColor = (strength) => {
    const absStrength = Math.abs(strength);
    if (absStrength >= 0.7) return '#dc2626'; // Strong correlation - red
    if (absStrength >= 0.5) return '#f59e0b'; // Moderate correlation - yellow
    if (absStrength >= 0.3) return '#3b82f6'; // Weak correlation - blue
    return '#6b7280'; // Very weak correlation - gray
  };

  const getCorrelationInterpretation = (strength) => {
    const absStrength = Math.abs(strength);
    if (absStrength >= 0.7) return 'Strong';
    if (absStrength >= 0.5) return 'Moderate';
    if (absStrength >= 0.3) return 'Weak';
    return 'Very Weak';
  };

  const availableCorrelations = data?.correlations ? Object.keys(data.correlations) : [];

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

  if (!data) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">No correlation data available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-700">Correlation Analysis</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedCorrelation}
              onChange={(e) => setSelectedCorrelation(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {availableCorrelations.map(correlation => (
                <option key={correlation} value={correlation}>
                  {correlation.replace('_', ' ').toUpperCase()}
                </option>
              ))}\n            </select>
            <select
              value={selectedPROType}
              onChange={(e) => setSelectedPROType(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="total_score">Total Score</option>
              <option value="pain_score">Pain Score</option>
              <option value="function_score">Function Score</option>
              <option value="symptoms_score">Symptoms Score</option>
            </select>
            <select
              value={timeWindow}
              onChange={(e) => setTimeWindow(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="7">7-day window</option>
              <option value="14">14-day window</option>
              <option value="30">30-day window</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Correlation Strength</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-blue-600">
                {correlationStrength[selectedCorrelation]?.strength?.toFixed(2) || 'N/A'}
              </span>
              <div className="text-sm text-gray-600 mt-1">
                {correlationStrength[selectedCorrelation] 
                  ? getCorrelationInterpretation(correlationStrength[selectedCorrelation].strength)
                  : 'Unknown'}\n              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Direction</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-green-600">
                {correlationStrength[selectedCorrelation]?.direction === 'positive' ? '↗' : '↘'}\n              </span>
              <div className="text-sm text-gray-600 mt-1">
                {correlationStrength[selectedCorrelation]?.direction || 'Unknown'}\n              </div>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Significance</h4>
            <div className="mt-2">
              <span className={`text-2xl font-bold ${
                correlationStrength[selectedCorrelation]?.significance === 'significant' 
                  ? 'text-green-600' : 'text-red-600'
              }`}>
                {correlationStrength[selectedCorrelation]?.significance === 'significant' ? '✓' : '✗'}\n              </span>
              <div className="text-sm text-gray-600 mt-1">
                {correlationStrength[selectedCorrelation]?.significance || 'Unknown'}\n              </div>
            </div>
          </div>

          <div className="bg-orange-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Data Points</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-orange-600">
                {correlationData.length}\n              </span>
              <div className="text-sm text-gray-600 mt-1">
                Observations\n              </div>
            </div>
          </div>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart data={correlationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="wearable_value" 
                name="Wearable Metric"
                type="number"
                domain={['dataMin', 'dataMax']}\n              />
              <YAxis 
                dataKey="pro_score" 
                name="PRO Score"
                type="number"
                domain={['dataMin', 'dataMax']}\n              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded shadow-lg">
                        <p className="text-sm font-medium">{data.date}</p>
                        <p className="text-sm text-blue-600">
                          Wearable: {data.wearable_value}
                        </p>
                        <p className="text-sm text-green-600">
                          PRO Score: {data.pro_score}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter 
                dataKey="pro_score" 
                fill={getCorrelationColor(correlationStrength[selectedCorrelation]?.strength || 0)}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Time-Series Correlation</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeSeriesCorrelation}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[-1, 1]} />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="correlation" 
                stroke={chartConfig.colors.primary}
                strokeWidth={2}
                dot={{ r: 4 }}
                name="Correlation Coefficient"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Correlation Summary</h3>
        <div className="space-y-3">
          {Object.entries(correlationStrength).map(([type, stats]) => (
            <div key={type} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div 
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: getCorrelationColor(stats.strength) }}
                ></div>
                <span className="text-sm font-medium text-gray-700">
                  {type.replace('_', ' ').toUpperCase()}
                </span>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-gray-900">
                  {stats.strength.toFixed(2)} ({getCorrelationInterpretation(stats.strength)})
                </div>
                <div className="text-xs text-gray-500">
                  {stats.direction} • {stats.significance}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {data.insights && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Correlation Insights</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">Key Findings</h4>
              <div className="space-y-3">
                {data.insights.key_findings?.map((finding, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <div>
                      <p className="text-sm text-gray-700">{finding.observation}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Strength: {finding.strength} | Clinical relevance: {finding.clinical_relevance}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">Clinical Implications</h4>
              <div className="space-y-3">
                {data.insights.clinical_implications?.map((implication, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                    <div>
                      <p className="text-sm text-gray-700">{implication.finding}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Recommendation: {implication.recommendation}
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
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Correlation Interpretation Guide</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Correlation Strength</h4>
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-sm text-gray-600">Strong (±0.7 to ±1.0)</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span className="text-sm text-gray-600">Moderate (±0.5 to ±0.7)</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-sm text-gray-600">Weak (±0.3 to ±0.5)</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                <span className="text-sm text-gray-600">Very Weak (0 to ±0.3)</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Clinical Significance</h4>
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className="text-green-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  Strong positive correlations suggest wearable metrics can predict PRO improvements
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-blue-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  Negative correlations may indicate compensatory behaviors or measurement issues
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-purple-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  Time-lagged correlations help identify delayed effects of interventions
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CorrelationAnalysisView;