import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { useBarChartConfig, usePieChartConfig } from '../../../hooks/useChartConfig';

const ProviderDashboard = ({ patientId, data }) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('week');
  const [focusArea, setFocusArea] = useState('overview');
  
  const barConfig = useBarChartConfig(['value']);
  const pieConfig = usePieChartConfig();

  const getProgressData = () => {
    if (!data || !data.progress_metrics) return [];
    
    return Object.entries(data.progress_metrics).map(([metric, value]) => ({
      name: metric.replace('_', ' ').toUpperCase(),
      value: value,
      target: 100,
      status: value >= 80 ? 'excellent' : value >= 60 ? 'good' : value >= 40 ? 'fair' : 'poor',
    }));
  };

  const getRiskFactors = () => {
    if (!data || !data.risk_assessment) return [];
    
    return data.risk_assessment.risk_factors?.map(factor => ({
      name: factor.factor,
      severity: factor.severity,
      impact: factor.impact_score,
      color: factor.severity === 'high' ? '#ef4444' : factor.severity === 'medium' ? '#f59e0b' : '#10b981',
    })) || [];
  };

  const getRecommendations = () => {
    if (!data || !data.clinical_recommendations) return [];
    
    return data.clinical_recommendations.map(rec => ({
      category: rec.category,
      priority: rec.priority,
      text: rec.recommendation,
      evidence: rec.evidence_level,
      timeline: rec.expected_timeline,
    }));
  };

  const getAlertsSummary = () => {
    if (!data || !data.alerts_summary) return { critical: 0, high: 0, medium: 0, low: 0 };
    
    return data.alerts_summary.reduce((acc, alert) => {
      acc[alert.severity] = (acc[alert.severity] || 0) + 1;
      return acc;
    }, { critical: 0, high: 0, medium: 0, low: 0 });
  };

  const getOutcomeProjections = () => {
    if (!data || !data.outcome_projections) return [];
    
    return Object.entries(data.outcome_projections).map(([outcome, probability]) => ({
      name: outcome.replace('_', ' ').toUpperCase(),
      probability: probability * 100,
      color: probability > 0.7 ? '#10b981' : probability > 0.4 ? '#f59e0b' : '#ef4444',
    }));
  };

  const progressData = getProgressData();
  const riskFactors = getRiskFactors();
  const recommendations = getRecommendations();
  const alertsSummary = getAlertsSummary();
  const outcomeProjections = getOutcomeProjections();

  if (!data) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">No provider dashboard data available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-700">Provider Dashboard</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="week">This Week</option>
              <option value="month">This Month</option>
              <option value="quarter">This Quarter</option>
            </select>
            <select
              value={focusArea}
              onChange={(e) => setFocusArea(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="overview">Overview</option>
              <option value="progress">Progress Tracking</option>
              <option value="risk">Risk Assessment</option>
              <option value="recommendations">Clinical Recommendations</option>
            </select>
          </div>
        </div>

        {/* Key Metrics Summary */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Overall Progress</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-blue-600">
                {data.overall_progress_score?.toFixed(0) || 'N/A'}%
              </span>
              <div className="text-sm text-gray-600 mt-1">
                {data.progress_trend === 'improving' ? '↗ Improving' : '↘ Declining'}
              </div>
            </div>
          </div>

          <div className="bg-red-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Active Alerts</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-red-600">
                {alertsSummary.critical + alertsSummary.high}
              </span>
              <div className="text-sm text-gray-600 mt-1">
                {alertsSummary.critical} Critical, {alertsSummary.high} High
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Compliance Score</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-green-600">
                {data.compliance_score?.toFixed(0) || 'N/A'}%
              </span>
              <div className="text-sm text-gray-600 mt-1">
                Exercise & Monitoring
              </div>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Next Appointment</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-purple-600">
                {data.next_appointment_days || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">days</span>
              <div className="text-sm text-gray-600 mt-1">
                Recommended follow-up
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Progress Tracking */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Recovery Progress</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={progressData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill={barConfig.colors.primary} />
                  <Bar dataKey="target" fill={barConfig.colors.secondary} fillOpacity={0.3} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Outcome Projections */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Outcome Projections</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={outcomeProjections}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, probability }) => `${name} ${probability.toFixed(0)}%`}
                    outerRadius={60}
                    fill="#8884d8"
                    dataKey="probability"
                  >
                    {outcomeProjections.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Assessment */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Risk Assessment</h3>
        <div className="space-y-3">
          {riskFactors.map((factor, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div 
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: factor.color }}
                ></div>
                <div>
                  <span className="text-sm font-medium text-gray-700">{factor.name}</span>
                  <div className="text-xs text-gray-500">
                    Impact Score: {factor.impact}/10
                  </div>
                </div>
              </div>
              <div className="text-right">
                <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                  factor.severity === 'high' ? 'bg-red-100 text-red-800' :
                  factor.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {factor.severity.toUpperCase()}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Clinical Recommendations */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Clinical Recommendations</h3>
        <div className="space-y-4">
          {recommendations.map((rec, index) => (
            <div key={index} className={`p-4 rounded-lg border-l-4 ${
              rec.priority === 'high' ? 'border-red-500 bg-red-50' :
              rec.priority === 'medium' ? 'border-yellow-500 bg-yellow-50' :
              'border-green-500 bg-green-50'
            }`}>
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="text-sm font-medium text-gray-700">{rec.category}</span>
                    <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                      rec.priority === 'high' ? 'bg-red-100 text-red-800' :
                      rec.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {rec.priority.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{rec.text}</p>
                  <div className="text-xs text-gray-500">
                    Evidence Level: {rec.evidence} | Timeline: {rec.timeline}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Patient Summary */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Executive Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-gray-600 mb-3">Key Achievements</h4>
            <div className="space-y-2">
              {data.key_achievements?.map((achievement, index) => (
                <div key={index} className="flex items-start space-x-2">
                  <span className="text-green-500 mt-1">✓</span>
                  <span className="text-sm text-gray-600">{achievement}</span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium text-gray-600 mb-3">Areas of Concern</h4>
            <div className="space-y-2">
              {data.areas_of_concern?.map((concern, index) => (
                <div key={index} className="flex items-start space-x-2">
                  <span className="text-red-500 mt-1">!</span>
                  <span className="text-sm text-gray-600">{concern}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {data.clinical_summary && (
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Clinical Summary</h4>
            <p className="text-sm text-gray-600">{data.clinical_summary}</p>
          </div>
        )}
      </div>

      {/* Export Options */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Export & Sharing</h3>
        <div className="flex flex-wrap gap-3">
          <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
            📊 Generate Report
          </button>
          <button className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors">
            📧 Email Summary
          </button>
          <button className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors">
            📋 Copy to Clipboard
          </button>
          <button className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors">
            🖨️ Print Summary
          </button>
        </div>
      </div>
    </div>
  );
};

export default ProviderDashboard;