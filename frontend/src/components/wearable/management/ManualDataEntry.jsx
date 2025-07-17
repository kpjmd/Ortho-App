import React, { useState } from 'react';
import axios from 'axios';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const ManualDataEntry = ({ patientId, patient }) => {
  const [formData, setFormData] = useState({
    date: new Date().toISOString().split('T')[0],
    steps: '',
    distance: '',
    calories_burned: '',
    active_minutes: '',
    sedentary_minutes: '',
    resting_hr: '',
    max_hr: '',
    avg_hr: '',
    hrv: '',
    total_sleep_minutes: '',
    deep_sleep_minutes: '',
    light_sleep_minutes: '',
    rem_sleep_minutes: '',
    awake_minutes: '',
    sleep_efficiency: '',
    sleep_score: '',
    bedtime: '',
    wake_time: '',
    walking_speed_ms: '',
    elevation_gain: '',
    floors_climbed: '',
    oxygen_saturation: '',
  });

  const [activeSection, setActiveSection] = useState('activity');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const submissionData = {
        patient_id: patientId,
        date: formData.date,
        ...Object.fromEntries(
          Object.entries(formData).filter(([key, value]) => 
            key !== 'date' && value !== ''
          ).map(([key, value]) => [key, parseFloat(value) || value])
        )
      };

      await axios.post(`${API}/wearable-data`, submissionData);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
      
      // Reset form
      setFormData({
        date: new Date().toISOString().split('T')[0],
        steps: '',
        distance: '',
        calories_burned: '',
        active_minutes: '',
        sedentary_minutes: '',
        resting_hr: '',
        max_hr: '',
        avg_hr: '',
        hrv: '',
        total_sleep_minutes: '',
        deep_sleep_minutes: '',
        light_sleep_minutes: '',
        rem_sleep_minutes: '',
        awake_minutes: '',
        sleep_efficiency: '',
        sleep_score: '',
        bedtime: '',
        wake_time: '',
        walking_speed_ms: '',
        elevation_gain: '',
        floors_climbed: '',
        oxygen_saturation: '',
      });
    } catch (err) {
      console.error('Failed to submit wearable data:', err);
      setError(err.response?.data?.detail || 'Failed to submit data');
    } finally {
      setLoading(false);
    }
  };

  const sections = [
    { id: 'activity', label: 'Activity', icon: '🏃' },
    { id: 'heart_rate', label: 'Heart Rate', icon: '❤️' },
    { id: 'sleep', label: 'Sleep', icon: '😴' },
    { id: 'other', label: 'Other', icon: '📊' },
  ];

  const renderActivitySection = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Steps
        </label>
        <input
          type="number"
          value={formData.steps}
          onChange={(e) => handleChange('steps', e.target.value)}
          placeholder="Daily steps"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Distance (km)
        </label>
        <input
          type="number"
          step="0.1"
          value={formData.distance}
          onChange={(e) => handleChange('distance', e.target.value)}
          placeholder="Distance walked/run"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Calories Burned
        </label>
        <input
          type="number"
          value={formData.calories_burned}
          onChange={(e) => handleChange('calories_burned', e.target.value)}
          placeholder="Calories burned"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Active Minutes
        </label>
        <input
          type="number"
          value={formData.active_minutes}
          onChange={(e) => handleChange('active_minutes', e.target.value)}
          placeholder="Minutes of activity"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Sedentary Minutes
        </label>
        <input
          type="number"
          value={formData.sedentary_minutes}
          onChange={(e) => handleChange('sedentary_minutes', e.target.value)}
          placeholder="Minutes sedentary"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Walking Speed (m/s)
        </label>
        <input
          type="number"
          step="0.1"
          value={formData.walking_speed_ms}
          onChange={(e) => handleChange('walking_speed_ms', e.target.value)}
          placeholder="Average walking speed"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  );

  const renderHeartRateSection = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Resting Heart Rate (bpm)
        </label>
        <input
          type="number"
          value={formData.resting_hr}
          onChange={(e) => handleChange('resting_hr', e.target.value)}
          placeholder="Resting heart rate"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Maximum Heart Rate (bpm)
        </label>
        <input
          type="number"
          value={formData.max_hr}
          onChange={(e) => handleChange('max_hr', e.target.value)}
          placeholder="Maximum heart rate"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Average Heart Rate (bpm)
        </label>
        <input
          type="number"
          value={formData.avg_hr}
          onChange={(e) => handleChange('avg_hr', e.target.value)}
          placeholder="Average heart rate"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Heart Rate Variability (ms)
        </label>
        <input
          type="number"
          value={formData.hrv}
          onChange={(e) => handleChange('hrv', e.target.value)}
          placeholder="HRV measurement"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  );

  const renderSleepSection = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Total Sleep (minutes)
        </label>
        <input
          type="number"
          value={formData.total_sleep_minutes}
          onChange={(e) => handleChange('total_sleep_minutes', e.target.value)}
          placeholder="Total sleep duration"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Deep Sleep (minutes)
        </label>
        <input
          type="number"
          value={formData.deep_sleep_minutes}
          onChange={(e) => handleChange('deep_sleep_minutes', e.target.value)}
          placeholder="Deep sleep duration"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Light Sleep (minutes)
        </label>
        <input
          type="number"
          value={formData.light_sleep_minutes}
          onChange={(e) => handleChange('light_sleep_minutes', e.target.value)}
          placeholder="Light sleep duration"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          REM Sleep (minutes)
        </label>
        <input
          type="number"
          value={formData.rem_sleep_minutes}
          onChange={(e) => handleChange('rem_sleep_minutes', e.target.value)}
          placeholder="REM sleep duration"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Awake Time (minutes)
        </label>
        <input
          type="number"
          value={formData.awake_minutes}
          onChange={(e) => handleChange('awake_minutes', e.target.value)}
          placeholder="Time awake in bed"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Sleep Efficiency (0-1)
        </label>
        <input
          type="number"
          step="0.01"
          min="0"
          max="1"
          value={formData.sleep_efficiency}
          onChange={(e) => handleChange('sleep_efficiency', e.target.value)}
          placeholder="Sleep efficiency ratio"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Sleep Score (0-100)
        </label>
        <input
          type="number"
          min="0"
          max="100"
          value={formData.sleep_score}
          onChange={(e) => handleChange('sleep_score', e.target.value)}
          placeholder="Overall sleep score"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Bedtime
        </label>
        <input
          type="time"
          value={formData.bedtime}
          onChange={(e) => handleChange('bedtime', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Wake Time
        </label>
        <input
          type="time"
          value={formData.wake_time}
          onChange={(e) => handleChange('wake_time', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  );

  const renderOtherSection = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Elevation Gain (m)
        </label>
        <input
          type="number"
          value={formData.elevation_gain}
          onChange={(e) => handleChange('elevation_gain', e.target.value)}
          placeholder="Elevation climbed"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Floors Climbed
        </label>
        <input
          type="number"
          value={formData.floors_climbed}
          onChange={(e) => handleChange('floors_climbed', e.target.value)}
          placeholder="Number of floors"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Oxygen Saturation (%)
        </label>
        <input
          type="number"
          min="0"
          max="100"
          value={formData.oxygen_saturation}
          onChange={(e) => handleChange('oxygen_saturation', e.target.value)}
          placeholder="SpO2 percentage"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
    </div>
  );

  const renderSectionContent = () => {
    switch (activeSection) {
      case 'activity':
        return renderActivitySection();
      case 'heart_rate':
        return renderHeartRateSection();
      case 'sleep':
        return renderSleepSection();
      case 'other':
        return renderOtherSection();
      default:
        return renderActivitySection();
    }
  };

  return (
    <div className="space-y-6">
      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <span className="text-green-600 mr-2">✓</span>
            <span className="text-green-800">Wearable data submitted successfully!</span>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <span className="text-red-600 mr-2">⚠️</span>
            <span className="text-red-800">{error}</span>
          </div>
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-800 mb-2">Manual Data Entry</h3>
        <p className="text-blue-700">
          Enter wearable data manually for patient <strong>{patient?.name}</strong>. 
          Fill in only the fields you have data for - empty fields will be ignored.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Date
            </label>
            <input
              type="date"
              value={formData.date}
              onChange={(e) => handleChange('date', e.target.value)}
              required
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="border-b border-gray-200 mb-6">
            <div className="flex flex-wrap">
              {sections.map((section) => (
                <button
                  key={section.id}
                  type="button"
                  onClick={() => setActiveSection(section.id)}
                  className={`px-4 py-2 text-sm font-medium border-b-2 ${
                    activeSection === section.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <span className="mr-2">{section.icon}</span>
                  {section.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mb-6">
            {renderSectionContent()}
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {loading ? 'Submitting...' : 'Submit Data'}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default ManualDataEntry;