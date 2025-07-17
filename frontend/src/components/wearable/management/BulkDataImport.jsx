import React, { useState, useCallback } from 'react';
import axios from 'axios';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const BulkDataImport = ({ patientId, patient }) => {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState('csv');
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [preview, setPreview] = useState(null);
  const [mapping, setMapping] = useState({});

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      previewFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      previewFile(e.target.files[0]);
    }
  };

  const previewFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      
      if (fileType === 'csv') {
        const lines = content.split('\n').slice(0, 6); // First 5 rows + header
        const headers = lines[0].split(',').map(h => h.trim());
        const rows = lines.slice(1).map(line => line.split(',').map(cell => cell.trim()));
        
        setPreview({
          headers,
          rows: rows.filter(row => row.length === headers.length)
        });
        
        // Initialize mapping
        const initialMapping = {};
        headers.forEach((header, index) => {
          const normalizedHeader = header.toLowerCase().replace(/[^a-z0-9]/g, '_');
          initialMapping[header] = getFieldSuggestion(normalizedHeader);
        });
        setMapping(initialMapping);
        
      } else if (fileType === 'json') {
        try {
          const jsonData = JSON.parse(content);
          const sample = Array.isArray(jsonData) ? jsonData.slice(0, 5) : [jsonData];
          const headers = sample.length > 0 ? Object.keys(sample[0]) : [];
          
          setPreview({
            headers,
            rows: sample.map(item => headers.map(header => item[header]))
          });
          
          // Initialize mapping
          const initialMapping = {};
          headers.forEach(header => {
            const normalizedHeader = header.toLowerCase().replace(/[^a-z0-9]/g, '_');
            initialMapping[header] = getFieldSuggestion(normalizedHeader);
          });
          setMapping(initialMapping);
          
        } catch (err) {
          setError('Invalid JSON file format');
        }
      }
    };
    reader.readAsText(file);
  };

  const getFieldSuggestion = (header) => {
    const fieldMappings = {
      'date': 'date',
      'steps': 'steps',
      'distance': 'distance',
      'calories': 'calories_burned',
      'calories_burned': 'calories_burned',
      'active_minutes': 'active_minutes',
      'sedentary_minutes': 'sedentary_minutes',
      'resting_hr': 'resting_hr',
      'resting_heart_rate': 'resting_hr',
      'max_hr': 'max_hr',
      'maximum_heart_rate': 'max_hr',
      'avg_hr': 'avg_hr',
      'average_heart_rate': 'avg_hr',
      'hrv': 'hrv',
      'heart_rate_variability': 'hrv',
      'total_sleep': 'total_sleep_minutes',
      'sleep_duration': 'total_sleep_minutes',
      'deep_sleep': 'deep_sleep_minutes',
      'light_sleep': 'light_sleep_minutes',
      'rem_sleep': 'rem_sleep_minutes',
      'awake_time': 'awake_minutes',
      'sleep_efficiency': 'sleep_efficiency',
      'sleep_score': 'sleep_score',
      'bedtime': 'bedtime',
      'wake_time': 'wake_time',
      'walking_speed': 'walking_speed_ms',
      'elevation_gain': 'elevation_gain',
      'floors_climbed': 'floors_climbed',
      'oxygen_saturation': 'oxygen_saturation',
      'spo2': 'oxygen_saturation',
    };
    
    return fieldMappings[header] || 'ignore';
  };

  const handleMappingChange = (sourceField, targetField) => {
    setMapping(prev => ({
      ...prev,
      [sourceField]: targetField
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('patient_id', patientId);
      formData.append('file_type', fileType);
      formData.append('mapping', JSON.stringify(mapping));

      await axios.post(`${API}/wearable-data/bulk-import`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setSuccess(true);
      setFile(null);
      setPreview(null);
      setMapping({});
      setTimeout(() => setSuccess(false), 5000);
    } catch (err) {
      console.error('Failed to import data:', err);
      setError(err.response?.data?.detail || 'Failed to import data');
    } finally {
      setLoading(false);
    }
  };

  const availableFields = [
    'ignore',
    'date',
    'steps',
    'distance',
    'calories_burned',
    'active_minutes',
    'sedentary_minutes',
    'resting_hr',
    'max_hr',
    'avg_hr',
    'hrv',
    'total_sleep_minutes',
    'deep_sleep_minutes',
    'light_sleep_minutes',
    'rem_sleep_minutes',
    'awake_minutes',
    'sleep_efficiency',
    'sleep_score',
    'bedtime',
    'wake_time',
    'walking_speed_ms',
    'elevation_gain',
    'floors_climbed',
    'oxygen_saturation',
  ];

  return (
    <div className="space-y-6">
      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <span className="text-green-600 mr-2">✓</span>
            <span className="text-green-800">Data imported successfully!</span>
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
        <h3 className="text-lg font-semibold text-blue-800 mb-2">Bulk Data Import</h3>
        <p className="text-blue-700">
          Import wearable data in bulk for patient <strong>{patient?.name}</strong>. 
          Supported formats: CSV, JSON. The system will automatically map common field names.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            File Type
          </label>
          <select
            value={fileType}
            onChange={(e) => setFileType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
          </select>
        </div>

        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center ${
            dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="space-y-4">
            <div className="text-gray-500">
              <svg className="mx-auto h-12 w-12" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            
            {file ? (
              <div>
                <p className="text-sm text-gray-600">Selected file:</p>
                <p className="text-sm font-medium text-gray-900">{file.name}</p>
                <p className="text-xs text-gray-500">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
            ) : (
              <div>
                <p className="text-sm text-gray-600">
                  Drag and drop your file here, or{' '}
                  <label className="text-blue-600 hover:text-blue-500 cursor-pointer">
                    browse
                    <input
                      type="file"
                      accept={fileType === 'csv' ? '.csv' : '.json'}
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </label>
                </p>
                <p className="text-xs text-gray-500">
                  Supported formats: {fileType.toUpperCase()}
                </p>
              </div>
            )}
          </div>
        </div>

        {preview && (
          <div className="mt-6">
            <h4 className="text-sm font-medium text-gray-700 mb-4">Data Preview & Field Mapping</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Source Field
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Target Field
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Sample Data
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {preview.headers.map((header, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {header}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <select
                          value={mapping[header] || 'ignore'}
                          onChange={(e) => handleMappingChange(header, e.target.value)}
                          className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          {availableFields.map(field => (
                            <option key={field} value={field}>
                              {field === 'ignore' ? 'Ignore' : field.replace(/_/g, ' ').toUpperCase()}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {preview.rows[0] && preview.rows[0][index] 
                          ? preview.rows[0][index].toString().substring(0, 20) + 
                            (preview.rows[0][index].toString().length > 20 ? '...' : '')
                          : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-4">
              <h5 className="text-sm font-medium text-gray-700 mb-2">Data Preview (First 5 rows):</h5>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      {preview.headers.map((header, index) => (
                        <th key={index} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {preview.rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {cell?.toString().substring(0, 30) + (cell?.toString().length > 30 ? '...' : '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {preview && (
          <div className="mt-6 flex justify-end">
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {loading ? 'Importing...' : 'Import Data'}
            </button>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Import Instructions</h3>
        <div className="space-y-4 text-sm text-gray-600">
          <div>
            <h4 className="font-medium text-gray-700">CSV Format Requirements:</h4>
            <ul className="mt-2 space-y-1 list-disc list-inside">
              <li>First row must contain column headers</li>
              <li>Date format: YYYY-MM-DD</li>
              <li>Time format: HH:MM (24-hour)</li>
              <li>Numeric values only (no units)</li>
              <li>Empty cells are acceptable</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-700">JSON Format Requirements:</h4>
            <ul className="mt-2 space-y-1 list-disc list-inside">
              <li>Array of objects or single object</li>
              <li>Each object represents one day of data</li>
              <li>Date field is required</li>
              <li>Field names should match expected format</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-700">Common Field Names:</h4>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <div>steps, distance, calories_burned, active_minutes</div>
              <div>resting_hr, max_hr, hrv, total_sleep_minutes</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BulkDataImport;