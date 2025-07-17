import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { usePredictiveInsights } from '../../../hooks/useWearableAnalytics';
import { useTimeSeriesConfig, usePieChartConfig } from '../../../hooks/useChartConfig';

const PredictiveInsightsDashboard = ({ patientId, data: initialData }) => {
  const [selectedView, setSelectedView] = useState('timeline');
  const [confidenceLevel, setConfidenceLevel] = useState(0.8);
  
  const { data: fetchedData, loading, error } = usePredictiveInsights(patientId);
  const data = initialData || fetchedData;
  
  const timelineConfig = useTimeSeriesConfig(['predicted', 'lower_bound', 'upper_bound']);
  const pieConfig = usePieChartConfig();

  const timelineData = useMemo(() => {
    if (!data || !data.timeline_prediction) return [];
    
    const timeline = data.timeline_prediction;
    const weeks = Array.from({ length: timeline.total_weeks || 26 }, (_, i) => i + 1);
    
    return weeks.map(week => ({
      week,
      predicted: timeline.weekly_predictions?.[week - 1] || 0,
      lower_bound: timeline.confidence_intervals?.[week - 1]?.lower || 0,
      upper_bound: timeline.confidence_intervals?.[week - 1]?.upper || 0,
    }));
  }, [data]);

  const complicationRiskData = useMemo(() => {
    if (!data || !data.complication_risk) return [];
    
    const risks = data.complication_risk;
    return Object.entries(risks).map(([risk, probability]) => ({
      name: risk.replace('_', ' ').toUpperCase(),
      probability: probability * 100,
      color: probability > 0.7 ? '#ef4444' : probability > 0.4 ? '#f59e0b' : '#10b981',
    }));
  }, [data]);

  const milestoneData = useMemo(() => {
    if (!data || !data.recovery_milestones) return [];
    
    return data.recovery_milestones.map(milestone => ({
      milestone: milestone.name,
      expected_week: milestone.expected_week,
      probability: milestone.probability * 100,
      confidence: milestone.confidence * 100,
    }));
  }, [data]);

  const getRecoveryTimelineStatus = () => {
    if (!data || !data.timeline_prediction) return { status: 'Unknown', color: 'gray' };
    
    const timeline = data.timeline_prediction;
    const expectedWeeks = timeline.total_weeks;
    
    if (expectedWeeks <= 12) return { status: 'Ahead of Schedule', color: 'green' };
    if (expectedWeeks <= 24) return { status: 'On Track', color: 'blue' };
    if (expectedWeeks <= 36) return { status: 'Slightly Delayed', color: 'yellow' };
    return { status: 'Significantly Delayed', color: 'red' };
  };

  const getOverallRiskLevel = () => {
    if (!data || !data.complication_risk) return { level: 'Unknown', color: 'gray' };
    
    const risks = Object.values(data.complication_risk);
    const maxRisk = Math.max(...risks);
    
    if (maxRisk < 0.3) return { level: 'Low', color: 'green' };
    if (maxRisk < 0.6) return { level: 'Moderate', color: 'yellow' };
    return { level: 'High', color: 'red' };
  };

  const timelineStatus = getRecoveryTimelineStatus();
  const riskLevel = getOverallRiskLevel();

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
          <div className="text-gray-500">No predictive data available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-700">Predictive Insights</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedView}
              onChange={(e) => setSelectedView(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="timeline">Recovery Timeline</option>
              <option value="complications">Complication Risk</option>
              <option value="milestones">Recovery Milestones</option>
            </select>
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-600">Confidence:</label>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={confidenceLevel}
                onChange={(e) => setConfidenceLevel(parseFloat(e.target.value))}
                className="w-20"
              />
              <span className="text-sm text-gray-600">{(confidenceLevel * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Expected Recovery</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-blue-600">
                {data.timeline_prediction?.total_weeks || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">weeks</span>
              <div className={`text-sm text-${timelineStatus.color}-600 mt-1`}>
                {timelineStatus.status}
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Confidence Level</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-green-600">
                {data.timeline_prediction?.confidence 
                  ? `${(data.timeline_prediction.confidence * 100).toFixed(0)}%`
                  : 'N/A'}
              </span>
              <div className="text-sm text-gray-600 mt-1">Prediction accuracy</div>
            </div>
          </div>

          <div className="bg-yellow-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Risk Level</h4>
            <div className="mt-2">
              <span className={`text-2xl font-bold text-${riskLevel.color}-600`}>
                {riskLevel.level}
              </span>
              <div className="text-sm text-gray-600 mt-1">Overall risk</div>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Next Milestone</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-purple-600">
                {data.recovery_milestones?.[0]?.expected_week || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">weeks</span>
              <div className="text-sm text-gray-600 mt-1">
                {data.recovery_milestones?.[0]?.name || 'No milestone'}
              </div>
            </div>
          </div>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            {selectedView === 'timeline' && (
              <AreaChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="week" />
                <YAxis />
                <Tooltip />
                <Legend />
                
                <Area
                  dataKey="upper_bound"
                  stackId="1"
                  stroke="none"
                  fill={timelineConfig.colors.primary}
                  fillOpacity={0.2}
                />
                
                <Area
                  dataKey="lower_bound"
                  stackId="1"
                  stroke="none"
                  fill="white"
                  fillOpacity={1}
                />
                
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke={timelineConfig.colors.primary}
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  name="Predicted Recovery"
                />
              </AreaChart>
            )}
            
            {selectedView === 'complications' && (
              <BarChart data={complicationRiskData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                
                <Bar dataKey="probability" name="Risk Probability (%)">
                  {complicationRiskData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            )}
            
            {selectedView === 'milestones' && (
              <BarChart data={milestoneData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="milestone" />
                <YAxis />
                <Tooltip />
                <Legend />
                
                <Bar dataKey="expected_week" fill={timelineConfig.colors.primary} name="Expected Week" />
                <Bar dataKey="probability" fill={timelineConfig.colors.secondary} name="Probability (%)" />
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>

      {data.plateau_risk && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Plateau Risk Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-600">Plateau Risk</h4>
              <div className="mt-2">
                <span className={`text-2xl font-bold ${
                  data.plateau_risk.risk_level === 'high' ? 'text-red-600' :
                  data.plateau_risk.risk_level === 'medium' ? 'text-yellow-600' :
                  'text-green-600'
                }`}>
                  {data.plateau_risk.risk_level?.toUpperCase() || 'LOW'}
                </span>
                <div className="text-sm text-gray-600 mt-1">
                  {data.plateau_risk.risk_percentage 
                    ? `${(data.plateau_risk.risk_percentage * 100).toFixed(0)}% probability`
                    : 'Low probability'}
                </div>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-600">Expected Plateau</h4>
              <div className="mt-2">
                <span className="text-2xl font-bold text-blue-600">
                  {data.plateau_risk.expected_plateau_week || 'N/A'}
                </span>
                <span className="text-sm text-gray-600 ml-1">weeks</span>
                <div className="text-sm text-gray-600 mt-1">If plateau occurs</div>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-600">Prevention Score</h4>
              <div className="mt-2">
                <span className="text-2xl font-bold text-green-600">
                  {data.plateau_risk.prevention_score 
                    ? `${(data.plateau_risk.prevention_score * 100).toFixed(0)}%`
                    : 'N/A'}
                </span>
                <div className="text-sm text-gray-600 mt-1">Preventability</div>
              </div>
            </div>
          </div>

          {data.plateau_risk.prevention_strategies && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Prevention Strategies</h4>
              <div className="space-y-2">
                {data.plateau_risk.prevention_strategies.map((strategy, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <span className="text-blue-500 mt-1">•</span>
                    <span className="text-sm text-gray-600">{strategy}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {data.personalized_recommendations && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Personalized Recommendations</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(data.personalized_recommendations).map(([category, recommendations]) => (
              <div key={category} className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-700 mb-3">
                  {category.replace('_', ' ').toUpperCase()}
                </h4>
                <div className="space-y-2">
                  {recommendations.map((rec, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span className="text-sm text-gray-600">{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.outcome_prediction && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Outcome Prediction</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-green-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-600">Excellent Recovery</h4>
              <div className="mt-2">
                <span className="text-2xl font-bold text-green-600">
                  {data.outcome_prediction.excellent_probability 
                    ? `${(data.outcome_prediction.excellent_probability * 100).toFixed(0)}%`
                    : 'N/A'}
                </span>
                <div className="text-sm text-gray-600 mt-1">Probability</div>
              </div>
            </div>

            <div className="bg-blue-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-600">Good Recovery</h4>
              <div className="mt-2">
                <span className="text-2xl font-bold text-blue-600">
                  {data.outcome_prediction.good_probability 
                    ? `${(data.outcome_prediction.good_probability * 100).toFixed(0)}%`
                    : 'N/A'}
                </span>
                <div className="text-sm text-gray-600 mt-1">Probability</div>
              </div>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-600">Complications</h4>
              <div className="mt-2">
                <span className="text-2xl font-bold text-yellow-600">
                  {data.outcome_prediction.complication_probability 
                    ? `${(data.outcome_prediction.complication_probability * 100).toFixed(0)}%`
                    : 'N/A'}
                </span>
                <div className="text-sm text-gray-600 mt-1">Probability</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictiveInsightsDashboard;