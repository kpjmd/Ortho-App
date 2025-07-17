import React, { useState, useMemo } from 'react';
import { useClinicalAlerts } from '../../../hooks/useWearableAnalytics';
import { useAlertColors } from '../../../hooks/useChartConfig';

const ClinicalAlertsPanel = ({ patientId, alerts: initialAlerts }) => {
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [filterCategory, setFilterCategory] = useState('all');
  const [showResolved, setShowResolved] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState(null);

  const { alerts: fetchedAlerts, loading, error, refetch } = useClinicalAlerts(patientId);
  const alerts = initialAlerts || fetchedAlerts || [];
  const alertColors = useAlertColors();

  const filteredAlerts = useMemo(() => {
    return alerts.filter(alert => {
      if (filterSeverity !== 'all' && alert.severity !== filterSeverity) return false;
      if (filterCategory !== 'all' && alert.category !== filterCategory) return false;
      if (!showResolved && alert.resolved) return false;
      return true;
    });
  }, [alerts, filterSeverity, filterCategory, showResolved]);

  const alertCounts = useMemo(() => {
    return alerts.reduce((counts, alert) => {
      counts[alert.severity] = (counts[alert.severity] || 0) + 1;
      return counts;
    }, {});
  }, [alerts]);

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical': return '🚨';
      case 'high': return '⚠️';
      case 'medium': return '📢';
      case 'low': return 'ℹ️';
      default: return '📋';
    }
  };

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'activity': return '🏃';
      case 'sleep': return '😴';
      case 'heart_rate': return '❤️';
      case 'recovery': return '🔄';
      case 'pain': return '🩹';
      case 'mobility': return '🚶';
      default: return '📊';
    }
  };

  const getTimeSinceAlert = (timestamp) => {
    const now = new Date();
    const alertTime = new Date(timestamp);
    const diff = now - alertTime;
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    return 'Just now';
  };

  const handleAlertClick = (alert) => {
    setSelectedAlert(alert);
  };

  const handleCloseModal = () => {
    setSelectedAlert(null);
  };

  const availableCategories = [...new Set(alerts.map(alert => alert.category))];

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
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-700">Clinical Alerts</h3>
          <button
            onClick={refetch}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-red-50 rounded-lg p-4">
            <div className="flex items-center">
              <span className="text-2xl mr-3">🚨</span>
              <div>
                <h4 className="text-sm font-medium text-gray-600">Critical</h4>
                <span className="text-2xl font-bold text-red-600">
                  {alertCounts.critical || 0}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 rounded-lg p-4">
            <div className="flex items-center">
              <span className="text-2xl mr-3">⚠️</span>
              <div>
                <h4 className="text-sm font-medium text-gray-600">High</h4>
                <span className="text-2xl font-bold text-yellow-600">
                  {alertCounts.high || 0}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center">
              <span className="text-2xl mr-3">📢</span>
              <div>
                <h4 className="text-sm font-medium text-gray-600">Medium</h4>
                <span className="text-2xl font-bold text-blue-600">
                  {alertCounts.medium || 0}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center">
              <span className="text-2xl mr-3">ℹ️</span>
              <div>
                <h4 className="text-sm font-medium text-gray-600">Low</h4>
                <span className="text-2xl font-bold text-green-600">
                  {alertCounts.low || 0}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-4 mb-6">
          <select
            value={filterSeverity}
            onChange={(e) => setFilterSeverity(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>

          <select
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Categories</option>
            {availableCategories.map(category => (
              <option key={category} value={category}>
                {category.replace('_', ' ').toUpperCase()}
              </option>
            ))}
          </select>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={showResolved}
              onChange={(e) => setShowResolved(e.target.checked)}
              className="mr-2"
            />
            Show Resolved
          </label>
        </div>

        <div className="space-y-3">
          {filteredAlerts.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No alerts found matching the current filters.
            </div>
          ) : (
            filteredAlerts.map((alert, index) => (
              <div
                key={index}
                onClick={() => handleAlertClick(alert)}
                className={`p-4 rounded-lg border cursor-pointer transition-colors hover:bg-gray-50 ${
                  alert.severity === 'critical' ? 'border-red-200 bg-red-50' :
                  alert.severity === 'high' ? 'border-yellow-200 bg-yellow-50' :
                  alert.severity === 'medium' ? 'border-blue-200 bg-blue-50' :
                  'border-green-200 bg-green-50'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0 text-2xl">
                      {getSeverityIcon(alert.severity)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <h4 className="text-sm font-medium text-gray-900">{alert.title}</h4>
                        <span className="text-xs text-gray-500">
                          {getCategoryIcon(alert.category)} {alert.category}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{alert.description}</p>
                      {alert.recommendations && alert.recommendations.length > 0 && (
                        <div className="mt-2">
                          <p className="text-xs text-gray-500">Top recommendation:</p>
                          <p className="text-xs text-blue-600">{alert.recommendations[0]}</p>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex-shrink-0 text-right">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      alert.severity === 'critical' ? 'bg-red-100 text-red-800' :
                      alert.severity === 'high' ? 'bg-yellow-100 text-yellow-800' :
                      alert.severity === 'medium' ? 'bg-blue-100 text-blue-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {alert.severity}
                    </span>
                    <div className="text-xs text-gray-500 mt-1">
                      {getTimeSinceAlert(alert.timestamp)}
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-screen overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-700">Alert Details</h3>
              <button
                onClick={handleCloseModal}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <span className="text-3xl">{getSeverityIcon(selectedAlert.severity)}</span>
                <div>
                  <h4 className="text-lg font-medium text-gray-900">{selectedAlert.title}</h4>
                  <div className="flex items-center space-x-2 mt-1">
                    <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                      selectedAlert.severity === 'critical' ? 'bg-red-100 text-red-800' :
                      selectedAlert.severity === 'high' ? 'bg-yellow-100 text-yellow-800' :
                      selectedAlert.severity === 'medium' ? 'bg-blue-100 text-blue-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {selectedAlert.severity}
                    </span>
                    <span className="text-xs text-gray-500">
                      {getCategoryIcon(selectedAlert.category)} {selectedAlert.category}
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <h5 className="text-sm font-medium text-gray-700">Description</h5>
                <p className="text-sm text-gray-600 mt-1">{selectedAlert.description}</p>
              </div>

              {selectedAlert.recommendations && selectedAlert.recommendations.length > 0 && (
                <div>
                  <h5 className="text-sm font-medium text-gray-700">Recommendations</h5>
                  <ul className="mt-2 space-y-1">
                    {selectedAlert.recommendations.map((rec, index) => (
                      <li key={index} className="text-sm text-gray-600 flex items-start">
                        <span className="text-blue-500 mr-2">•</span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedAlert.data_source && (
                <div>
                  <h5 className="text-sm font-medium text-gray-700">Data Source</h5>
                  <p className="text-sm text-gray-600 mt-1">{selectedAlert.data_source}</p>
                </div>
              )}

              <div className="flex justify-between items-center text-xs text-gray-500">
                <span>Alert triggered: {new Date(selectedAlert.timestamp).toLocaleString()}</span>
                {selectedAlert.resolved && (
                  <span className="text-green-600">✓ Resolved</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ClinicalAlertsPanel;