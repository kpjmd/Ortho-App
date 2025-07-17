import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const RiskIndicator = ({ patientId }) => {
  const [riskData, setRiskData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRiskData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await axios.get(`${API}/risk-assessment/${patientId}`);
        setRiskData(response.data);
        
      } catch (err) {
        console.error('Failed to fetch risk assessment:', err);
        setError('Failed to load risk assessment');
      } finally {
        setLoading(false);
      }
    };

    if (patientId) {
      fetchRiskData();
    }
  }, [patientId]);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-16 bg-gray-200 rounded mb-4"></div>
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          </div>
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

  if (!riskData) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-gray-600 text-center">
          <p>No risk data available</p>
        </div>
      </div>
    );
  }

  // Get risk level styling
  const getRiskLevelStyling = (category) => {
    switch (category) {
      case 'Low':
        return {
          color: 'text-green-800',
          bg: 'bg-green-100',
          border: 'border-green-200',
          progressBar: 'bg-green-500',
          icon: '✓'
        };
      case 'Moderate':
        return {
          color: 'text-yellow-800',
          bg: 'bg-yellow-100',
          border: 'border-yellow-200',
          progressBar: 'bg-yellow-500',
          icon: '⚠'
        };
      case 'High':
        return {
          color: 'text-orange-800',
          bg: 'bg-orange-100',
          border: 'border-orange-200',
          progressBar: 'bg-orange-500',
          icon: '⚠'
        };
      case 'Very High':
        return {
          color: 'text-red-800',
          bg: 'bg-red-100',
          border: 'border-red-200',
          progressBar: 'bg-red-500',
          icon: '⚠'
        };
      default:
        return {
          color: 'text-gray-800',
          bg: 'bg-gray-100',
          border: 'border-gray-200',
          progressBar: 'bg-gray-500',
          icon: '?'
        };
    }
  };

  const styling = getRiskLevelStyling(riskData.risk_category);

  // Get risk level description
  const getRiskDescription = (category) => {
    switch (category) {
      case 'Low':
        return 'Recovery is progressing well with minimal risk factors.';
      case 'Moderate':
        return 'Some risk factors present but manageable with proper care.';
      case 'High':
        return 'Multiple risk factors require attention and intervention.';
      case 'Very High':
        return 'Significant risk factors present. Immediate medical attention recommended.';
      default:
        return 'Risk level could not be determined.';
    }
  };

  // Get recommendations for risk level
  const getRiskRecommendations = (category) => {
    switch (category) {
      case 'Low':
        return [
          'Continue current treatment plan',
          'Maintain regular follow-up schedule',
          'Keep up with prescribed exercises'
        ];
      case 'Moderate':
        return [
          'Increase monitoring frequency',
          'Consider adjusting treatment plan',
          'Focus on addressing identified risk factors'
        ];
      case 'High':
        return [
          'Schedule immediate clinical review',
          'Consider intensive intervention',
          'Implement risk mitigation strategies'
        ];
      case 'Very High':
        return [
          'Urgent clinical evaluation required',
          'Immediate intervention needed',
          'Close monitoring essential'
        ];
      default:
        return [];
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-xl font-semibold text-gray-800 mb-6">Risk Assessment</h3>

      {/* Risk Score Display */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${styling.bg} ${styling.border} border-2`}>
              <span className={`text-xl ${styling.color}`}>{styling.icon}</span>
            </div>
            <div>
              <div className={`text-2xl font-bold ${styling.color}`}>
                {riskData.risk_score.toFixed(1)}/100
              </div>
              <div className={`text-sm font-medium ${styling.color}`}>
                {riskData.risk_category} Risk
              </div>
            </div>
          </div>
        </div>

        {/* Risk Score Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
          <div 
            className={`h-3 rounded-full transition-all duration-500 ${styling.progressBar}`}
            style={{ width: `${riskData.risk_score}%` }}
          ></div>
        </div>

        {/* Risk Scale Labels */}
        <div className="flex justify-between text-xs text-gray-600 mb-4">
          <span>0 - Low</span>
          <span>25 - Moderate</span>
          <span>50 - High</span>
          <span>75 - Very High</span>
          <span>100</span>
        </div>

        {/* Risk Description */}
        <div className={`p-4 rounded-lg ${styling.bg} ${styling.border} border`}>
          <p className={`text-sm ${styling.color}`}>
            {getRiskDescription(riskData.risk_category)}
          </p>
        </div>
      </div>

      {/* Risk Factors */}
      {riskData.risk_factors.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Risk Factors</h4>
          <div className="space-y-2">
            {riskData.risk_factors.map((factor, index) => (
              <div key={index} className="flex items-start space-x-2 p-3 bg-red-50 rounded-md">
                <span className="text-red-600 mt-1">⚠</span>
                <span className="text-red-800 text-sm">{factor}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Concerning Patterns */}
      {riskData.concerning_patterns.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Concerning Patterns</h4>
          <div className="space-y-2">
            {riskData.concerning_patterns.map((pattern, index) => (
              <div key={index} className="flex items-start space-x-2 p-3 bg-orange-50 rounded-md">
                <span className="text-orange-600 mt-1">⚠</span>
                <span className="text-orange-800 text-sm">{pattern}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Protective Factors */}
      {riskData.protective_factors.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Protective Factors</h4>
          <div className="space-y-2">
            {riskData.protective_factors.map((factor, index) => (
              <div key={index} className="flex items-start space-x-2 p-3 bg-green-50 rounded-md">
                <span className="text-green-600 mt-1">✓</span>
                <span className="text-green-800 text-sm">{factor}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Positive Trends */}
      {riskData.positive_trends.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Positive Trends</h4>
          <div className="space-y-2">
            {riskData.positive_trends.map((trend, index) => (
              <div key={index} className="flex items-start space-x-2 p-3 bg-blue-50 rounded-md">
                <span className="text-blue-600 mt-1">↗</span>
                <span className="text-blue-800 text-sm">{trend}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk-Based Recommendations */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-3">Recommendations</h4>
        <div className="space-y-2">
          {getRiskRecommendations(riskData.risk_category).map((rec, index) => (
            <div key={index} className="flex items-start space-x-2 p-3 bg-gray-50 rounded-md">
              <span className="text-blue-600 mt-1">→</span>
              <span className="text-gray-800 text-sm">{rec}</span>
            </div>
          ))}
        </div>
      </div>

      {/* High Risk Alert */}
      {(riskData.risk_category === 'High' || riskData.risk_category === 'Very High') && (
        <div className="mb-6">
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-md">
            <div className="flex items-center">
              <span className="text-red-600 text-xl mr-2">⚠</span>
              <div>
                <h4 className="text-red-800 font-semibold">High Risk Alert</h4>
                <p className="text-red-700 text-sm">
                  This patient has been flagged as high risk for poor recovery outcomes. 
                  Consider scheduling an immediate clinical review and implementing 
                  additional monitoring protocols.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Risk Trend Indicator */}
      <div className="border-t pt-4">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <span>Risk assessment updated continuously</span>
          <span className="flex items-center">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-1"></span>
            Real-time monitoring
          </span>
        </div>
      </div>
    </div>
  );
};

export default RiskIndicator;