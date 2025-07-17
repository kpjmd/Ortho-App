import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useWearableData } from '../../../hooks/useWearableAnalytics';
import { useTimeSeriesConfig } from '../../../hooks/useChartConfig';

const CardiovascularRecoveryMonitor = ({ patientId, data: analyticsData }) => {
  const [selectedMetric, setSelectedMetric] = useState('resting_hr');
  const [timeRange, setTimeRange] = useState('30');
  const [showZones, setShowZones] = useState(true);
  
  const { data: wearableData, loading, error } = useWearableData(patientId);
  const lineConfig = useTimeSeriesConfig(['resting_hr', 'max_hr', 'avg_hr', 'hrv']);

  const heartRateData = useMemo(() => {
    if (!wearableData || wearableData.length === 0) return [];
    
    return wearableData.map(item => ({
      date: new Date(item.date).toLocaleDateString(),
      resting_hr: item.resting_hr || 0,
      max_hr: item.max_hr || 0,
      avg_hr: item.avg_hr || 0,
      hrv: item.hrv || 0,
      recovery_hr: item.recovery_hr || 0,
      exercise_hr: item.exercise_hr || 0,
    }));
  }, [wearableData]);

  const hrSummary = useMemo(() => {
    if (!heartRateData.length) return null;
    
    const recent = heartRateData.slice(-7);
    const avgRestingHR = recent.reduce((sum, day) => sum + day.resting_hr, 0) / recent.length;
    const avgMaxHR = recent.reduce((sum, day) => sum + day.max_hr, 0) / recent.length;
    const avgHRV = recent.reduce((sum, day) => sum + day.hrv, 0) / recent.length;
    const avgRecoveryHR = recent.reduce((sum, day) => sum + day.recovery_hr, 0) / recent.length;
    
    return {
      avgRestingHR: Math.round(avgRestingHR),
      avgMaxHR: Math.round(avgMaxHR),
      avgHRV: Math.round(avgHRV),
      avgRecoveryHR: Math.round(avgRecoveryHR),
      trend: heartRateData[heartRateData.length - 1].resting_hr < heartRateData[heartRateData.length - 7].resting_hr ? 'improving' : 'declining',
    };
  }, [heartRateData]);

  const getHRZones = (maxHR) => {
    if (!maxHR) return null;
    
    return {
      zone1: { min: 0, max: maxHR * 0.6, name: 'Recovery', color: '#10b981' },
      zone2: { min: maxHR * 0.6, max: maxHR * 0.7, name: 'Aerobic Base', color: '#3b82f6' },
      zone3: { min: maxHR * 0.7, max: maxHR * 0.8, name: 'Aerobic', color: '#f59e0b' },
      zone4: { min: maxHR * 0.8, max: maxHR * 0.9, name: 'Anaerobic', color: '#ef4444' },
      zone5: { min: maxHR * 0.9, max: maxHR, name: 'Neuromuscular', color: '#7c3aed' },
    };
  };

  const getCardiovascularFitness = (restingHR, maxHR, age = 40) => {
    if (!restingHR || !maxHR) return { fitness: 'Unknown', color: 'gray' };
    
    const hrReserve = maxHR - restingHR;
    const expectedMax = 220 - age;
    const fitnessScore = (hrReserve / expectedMax) * 100;
    
    if (fitnessScore >= 80) return { fitness: 'Excellent', color: 'green' };
    if (fitnessScore >= 60) return { fitness: 'Good', color: 'blue' };
    if (fitnessScore >= 40) return { fitness: 'Fair', color: 'yellow' };
    return { fitness: 'Poor', color: 'red' };
  };

  const getRecoveryReadiness = (restingHR, hrv) => {
    if (!restingHR || !hrv) return { readiness: 'Unknown', color: 'gray' };
    
    const baselineHR = hrSummary?.avgRestingHR || 60;
    const hrDeviation = (restingHR - baselineHR) / baselineHR;
    
    if (hrDeviation < -0.05 && hrv > 45) return { readiness: 'Excellent', color: 'green' };
    if (hrDeviation < 0.05 && hrv > 30) return { readiness: 'Good', color: 'blue' };
    if (hrDeviation < 0.1 && hrv > 20) return { readiness: 'Fair', color: 'yellow' };
    return { readiness: 'Poor', color: 'red' };
  };

  const currentFitness = getCardiovascularFitness(hrSummary?.avgRestingHR, hrSummary?.avgMaxHR);
  const recoveryReadiness = getRecoveryReadiness(hrSummary?.avgRestingHR, hrSummary?.avgHRV);
  const hrZones = getHRZones(hrSummary?.avgMaxHR);

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
          <h3 className="text-lg font-semibold text-gray-700">Cardiovascular Recovery Monitor</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="resting_hr">Resting Heart Rate</option>
              <option value="max_hr">Max Heart Rate</option>
              <option value="avg_hr">Average Heart Rate</option>
              <option value="hrv">Heart Rate Variability</option>
              <option value="recovery_hr">Recovery Heart Rate</option>
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
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showZones}
                onChange={(e) => setShowZones(e.target.checked)}
                className="mr-2"
              />
              Show HR Zones
            </label>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-red-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Resting Heart Rate</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-red-600">
                {hrSummary?.avgRestingHR || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">bpm</span>
              <div className="text-sm text-gray-600 mt-1">
                {hrSummary?.trend === 'improving' ? '↗ Improving' : '↘ Declining'}
              </div>
            </div>
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Heart Rate Variability</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-blue-600">
                {hrSummary?.avgHRV || 'N/A'}
              </span>
              <span className="text-sm text-gray-600 ml-1">ms</span>
              <div className="text-sm text-gray-600 mt-1">
                Recovery indicator
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Cardiovascular Fitness</h4>
            <div className="mt-2">
              <span className={`text-2xl font-bold text-${currentFitness.color}-600`}>
                {currentFitness.fitness}
              </span>
              <div className="text-sm text-gray-600 mt-1">
                Based on HR reserve
              </div>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Recovery Readiness</h4>
            <div className="mt-2">
              <span className={`text-2xl font-bold text-${recoveryReadiness.color}-600`}>
                {recoveryReadiness.readiness}
              </span>
              <div className="text-sm text-gray-600 mt-1">
                Training readiness
              </div>
            </div>
          </div>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={heartRateData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              
              {showZones && hrZones && selectedMetric === 'resting_hr' && (
                <>
                  <Area
                    dataKey={() => hrZones.zone1.max}
                    fill={hrZones.zone1.color}
                    fillOpacity={0.1}
                    stroke="none"
                  />
                  <Area
                    dataKey={() => hrZones.zone2.max}
                    fill={hrZones.zone2.color}
                    fillOpacity={0.1}
                    stroke="none"
                  />
                </>
              )}
              
              <Line
                type="monotone"
                dataKey={selectedMetric}
                stroke={lineConfig.colors.primary}
                strokeWidth={2}
                dot={{ r: 4 }}
                name={selectedMetric.replace('_', ' ').toUpperCase()}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {hrZones && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Heart Rate Training Zones</h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {Object.entries(hrZones).map(([key, zone]) => (
              <div key={key} className="border rounded-lg p-4" style={{ borderColor: zone.color }}>
                <div className="flex items-center mb-2">
                  <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: zone.color }}></div>
                  <span className="text-sm font-medium text-gray-700">{zone.name}</span>
                </div>
                <div className="text-sm text-gray-600">
                  {Math.round(zone.min)} - {Math.round(zone.max)} bpm
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Cardiovascular Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Recovery Indicators</h4>
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className="text-green-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  {hrSummary?.avgRestingHR < 60 
                    ? 'Excellent resting heart rate indicates good cardiovascular fitness'
                    : 'Resting heart rate within normal range'}
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-blue-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  {hrSummary?.avgHRV > 40 
                    ? 'High HRV suggests good recovery capacity'
                    : 'HRV indicates need for more recovery time'}
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-purple-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  Monitor heart rate trends for signs of overtraining or illness
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Training Recommendations</h4>
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className="text-green-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  {recoveryReadiness.readiness === 'Excellent' 
                    ? 'Ready for high-intensity training'
                    : 'Focus on low-intensity recovery activities'}
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-blue-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  {currentFitness.fitness === 'Poor' 
                    ? 'Gradually increase cardiovascular training intensity'
                    : 'Maintain current fitness level with consistent training'}
                </span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-orange-500 mt-1">•</span>
                <span className="text-sm text-gray-600">
                  Use heart rate zones to optimize training effectiveness
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {analyticsData?.cardiovascular_insights && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Advanced Cardiovascular Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">Recovery Patterns</h4>
              <div className="space-y-3">
                {analyticsData.cardiovascular_insights.recovery_patterns?.map((pattern, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                    <div>
                      <p className="text-sm text-gray-700">{pattern.observation}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Impact: {pattern.impact} | Recommendation: {pattern.recommendation}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-3">Fitness Progression</h4>
              <div className="space-y-3">
                {analyticsData.cardiovascular_insights.fitness_progression?.map((progress, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <div>
                      <p className="text-sm text-gray-700">{progress.metric}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Change: {progress.change} | Trend: {progress.trend}
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

export default CardiovascularRecoveryMonitor;