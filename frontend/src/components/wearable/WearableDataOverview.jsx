import React, { useState, useEffect } from 'react';
import { useWearableAnalytics } from '../../hooks/useWearableAnalytics';
import { useRealTimeUpdates } from '../../hooks/useRealTimeUpdates';
import RecoveryVelocityChart from './analytics/RecoveryVelocityChart';
import ClinicalAlertsPanel from './analytics/ClinicalAlertsPanel';
import PredictiveInsightsDashboard from './analytics/PredictiveInsightsDashboard';
import AdvancedActivityAnalytics from './analytics/AdvancedActivityAnalytics';
import SleepRecoveryAnalysis from './analytics/SleepRecoveryAnalysis';
import CardiovascularRecoveryMonitor from './analytics/CardiovascularRecoveryMonitor';
import CorrelationAnalysisView from './analytics/CorrelationAnalysisView';
import ProviderDashboard from './analytics/ProviderDashboard';
import DataManagementInterface from './management/DataManagementInterface';

const WearableDataOverview = ({ patientId, patient }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(true);
  const { data, loading, error, refetch } = useWearableAnalytics(patientId);

  useRealTimeUpdates(refetch, 30000, isRealTimeEnabled);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'velocity', label: 'Recovery Velocity', icon: '🚀' },
    { id: 'alerts', label: 'Clinical Alerts', icon: '⚠️' },
    { id: 'predictions', label: 'Predictions', icon: '🔮' },
    { id: 'activity', label: 'Activity', icon: '🏃' },
    { id: 'sleep', label: 'Sleep', icon: '😴' },
    { id: 'cardiovascular', label: 'Heart Rate', icon: '❤️' },
    { id: 'correlations', label: 'Correlations', icon: '📈' },
    { id: 'provider', label: 'Provider View', icon: '👩‍⚕️' },
    { id: 'data', label: 'Data Management', icon: '📝' },
  ];

  const getAlertCount = () => {
    if (!data.alerts) return 0;
    return data.alerts.filter(alert => alert.severity === 'critical' || alert.severity === 'high').length;
  };

  const renderOverview = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Recovery Status</h3>
        {data.velocity && (
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Overall Recovery Velocity</span>
              <span className="text-lg font-bold text-blue-600">
                {data.velocity.overall_velocity_score?.toFixed(1) || 'N/A'}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Progress Status</span>
              <span className={`px-2 py-1 text-xs rounded-full ${
                data.velocity.velocity_category === 'excellent' ? 'bg-green-100 text-green-800' :
                data.velocity.velocity_category === 'good' ? 'bg-blue-100 text-blue-800' :
                data.velocity.velocity_category === 'fair' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {data.velocity.velocity_category || 'Unknown'}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Clinical Alerts</h3>
        {data.alerts && data.alerts.length > 0 ? (
          <div className="space-y-2">
            {data.alerts.slice(0, 3).map((alert, index) => (
              <div key={index} className={`p-3 rounded-lg ${
                alert.severity === 'critical' ? 'bg-red-50 border border-red-200' :
                alert.severity === 'high' ? 'bg-yellow-50 border border-yellow-200' :
                'bg-blue-50 border border-blue-200'
              }`}>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{alert.title}</span>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    alert.severity === 'critical' ? 'bg-red-100 text-red-800' :
                    alert.severity === 'high' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-blue-100 text-blue-800'
                  }`}>
                    {alert.severity}
                  </span>
                </div>
                <p className="text-xs text-gray-600 mt-1">{alert.description}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No active alerts</p>
        )}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Predictive Insights</h3>
        {data.predictions && (
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Recovery Timeline</span>
              <span className="text-sm font-medium text-blue-600">
                {data.predictions.timeline_prediction?.estimated_weeks || 'N/A'} weeks
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Complication Risk</span>
              <span className={`px-2 py-1 text-xs rounded-full ${
                data.predictions.risk_assessment?.risk_level === 'high' ? 'bg-red-100 text-red-800' :
                data.predictions.risk_assessment?.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                'bg-green-100 text-green-800'
              }`}>
                {data.predictions.risk_assessment?.risk_level || 'Low'}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Recent Activity</h3>
        {data.insights && data.insights.activity_insights ? (
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Daily Steps</span>
              <span className="text-lg font-bold text-green-600">
                {data.insights.activity_insights.avg_daily_steps?.toFixed(0) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Activity Score</span>
              <span className="text-lg font-bold text-green-600">
                {data.insights.activity_insights.activity_score?.toFixed(1) || 'N/A'}%
              </span>
            </div>
          </div>
        ) : (
          <p className="text-gray-500">No activity data available</p>
        )}
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'velocity':
        return <RecoveryVelocityChart patientId={patientId} data={data.velocity} />;
      case 'alerts':
        return <ClinicalAlertsPanel patientId={patientId} alerts={data.alerts} />;
      case 'predictions':
        return <PredictiveInsightsDashboard patientId={patientId} data={data.predictions} />;
      case 'activity':
        return <AdvancedActivityAnalytics patientId={patientId} data={data.insights} />;
      case 'sleep':
        return <SleepRecoveryAnalysis patientId={patientId} data={data.insights} />;
      case 'cardiovascular':
        return <CardiovascularRecoveryMonitor patientId={patientId} data={data.insights} />;
      case 'correlations':
        return <CorrelationAnalysisView patientId={patientId} data={data.correlations} />;
      case 'provider':
        return <ProviderDashboard patientId={patientId} data={data.providerDashboard} />;
      case 'data':
        return <DataManagementInterface patientId={patientId} patient={patient} />;
      default:
        return renderOverview();
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
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
        <button
          onClick={refetch}
          className="mt-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Wearable Data Analytics</h2>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setIsRealTimeEnabled(!isRealTimeEnabled)}
            className={`px-3 py-1 text-sm rounded-md ${
              isRealTimeEnabled
                ? 'bg-green-100 text-green-800'
                : 'bg-gray-100 text-gray-800'
            }`}
          >
            {isRealTimeEnabled ? '🟢 Real-time' : '⚫ Manual'}
          </button>
          <button
            onClick={refetch}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md">
        <div className="border-b border-gray-200">
          <div className="flex flex-wrap">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium border-b-2 ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
                {tab.id === 'alerts' && getAlertCount() > 0 && (
                  <span className="ml-2 bg-red-500 text-white text-xs rounded-full px-2 py-1">
                    {getAlertCount()}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        <div className="p-6">
          {renderTabContent()}
        </div>
      </div>
    </div>
  );
};

export default WearableDataOverview;