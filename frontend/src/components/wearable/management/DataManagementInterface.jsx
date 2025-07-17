import React, { useState } from 'react';
import ManualDataEntry from './ManualDataEntry';
import BulkDataImport from './BulkDataImport';

const DataManagementInterface = ({ patientId, patient }) => {
  const [activeTab, setActiveTab] = useState('manual');
  const [dataSource, setDataSource] = useState('manual');

  const tabs = [
    { id: 'manual', label: 'Manual Entry', icon: '✏️' },
    { id: 'import', label: 'Bulk Import', icon: '📁' },
    { id: 'sync', label: 'Device Sync', icon: '🔄' },
    { id: 'export', label: 'Export Data', icon: '📊' },
  ];

  const renderManualEntry = () => (
    <ManualDataEntry patientId={patientId} patient={patient} />
  );

  const renderBulkImport = () => (
    <BulkDataImport patientId={patientId} patient={patient} />
  );

  const renderDeviceSync = () => (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-800 mb-4">Device Synchronization</h3>
        <p className="text-blue-700 mb-4">
          Connect and sync data from popular wearable devices and health platforms.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white rounded-lg p-4 border border-blue-200">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-8 h-8 bg-black rounded-full flex items-center justify-center">
                <span className="text-white text-xs font-bold">A</span>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">Apple Health</h4>
                <p className="text-sm text-gray-600">iPhone & Apple Watch</p>
              </div>
            </div>
            <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
              Connect Apple Health
            </button>
          </div>

          <div className="bg-white rounded-lg p-4 border border-blue-200">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                <span className="text-white text-xs font-bold">G</span>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">Google Fit</h4>
                <p className="text-sm text-gray-600">Android & Wear OS</p>
              </div>
            </div>
            <button className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors">
              Connect Google Fit
            </button>
          </div>

          <div className="bg-white rounded-lg p-4 border border-blue-200">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                <span className="text-white text-xs font-bold">F</span>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">Fitbit</h4>
                <p className="text-sm text-gray-600">Fitbit devices</p>
              </div>
            </div>
            <button className="w-full px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors">
              Connect Fitbit
            </button>
          </div>

          <div className="bg-white rounded-lg p-4 border border-blue-200">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center">
                <span className="text-white text-xs font-bold">G</span>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">Garmin</h4>
                <p className="text-sm text-gray-600">Garmin devices</p>
              </div>
            </div>
            <button className="w-full px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors">
              Connect Garmin
            </button>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Sync Status</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
              <span className="text-sm font-medium text-gray-700">Apple Health</span>
            </div>
            <span className="text-sm text-gray-500">Not connected</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
              <span className="text-sm font-medium text-gray-700">Google Fit</span>
            </div>
            <span className="text-sm text-gray-500">Not connected</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderExportData = () => (
    <div className="space-y-6">
      <div className="bg-green-50 border border-green-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-green-800 mb-4">Export Patient Data</h3>
        <p className="text-green-700 mb-4">
          Export patient data in various formats for analysis, reporting, or backup purposes.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white rounded-lg p-4 border border-green-200">
            <h4 className="font-medium text-gray-900 mb-2">CSV Export</h4>
            <p className="text-sm text-gray-600 mb-3">
              Export all data in comma-separated values format for Excel or other spreadsheet applications.
            </p>
            <button className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors">
              Download CSV
            </button>
          </div>

          <div className="bg-white rounded-lg p-4 border border-green-200">
            <h4 className="font-medium text-gray-900 mb-2">JSON Export</h4>
            <p className="text-sm text-gray-600 mb-3">
              Export data in JSON format for API integration or custom applications.
            </p>
            <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
              Download JSON
            </button>
          </div>

          <div className="bg-white rounded-lg p-4 border border-green-200">
            <h4 className="font-medium text-gray-900 mb-2">PDF Report</h4>
            <p className="text-sm text-gray-600 mb-3">
              Generate a comprehensive PDF report with charts and analysis.
            </p>
            <button className="w-full px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors">
              Generate PDF
            </button>
          </div>

          <div className="bg-white rounded-lg p-4 border border-green-200">
            <h4 className="font-medium text-gray-900 mb-2">FHIR Export</h4>
            <p className="text-sm text-gray-600 mb-3">
              Export data in FHIR format for interoperability with healthcare systems.
            </p>
            <button className="w-full px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors">
              Export FHIR
            </button>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Export Options</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-gray-600 mb-3">Date Range</h4>
            <div className="space-y-3">
              <div className="flex items-center space-x-4">
                <label className="text-sm text-gray-600">From:</label>
                <input 
                  type="date" 
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div className="flex items-center space-x-4">
                <label className="text-sm text-gray-600">To:</label>
                <input 
                  type="date" 
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium text-gray-600 mb-3">Data Types</h4>
            <div className="space-y-2">
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" defaultChecked />
                <span className="text-sm text-gray-600">Wearable Data</span>
              </label>
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" defaultChecked />
                <span className="text-sm text-gray-600">PRO Surveys</span>
              </label>
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" defaultChecked />
                <span className="text-sm text-gray-600">Analytics Results</span>
              </label>
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" defaultChecked />
                <span className="text-sm text-gray-600">Clinical Alerts</span>
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'manual':
        return renderManualEntry();
      case 'import':
        return renderBulkImport();
      case 'sync':
        return renderDeviceSync();
      case 'export':
        return renderExportData();
      default:
        return renderManualEntry();
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md">
        <div className="border-b border-gray-200">
          <div className="flex flex-wrap">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 text-sm font-medium border-b-2 ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <div className="p-6">
          {renderTabContent()}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Data Quality Monitor</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Data Completeness</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-green-600">87%</span>
              <div className="text-sm text-gray-600 mt-1">
                Good data coverage
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Missing Data Points</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-yellow-600">3</span>
              <div className="text-sm text-gray-600 mt-1">
                Days with gaps
              </div>
            </div>
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-600">Last Updated</h4>
            <div className="mt-2">
              <span className="text-2xl font-bold text-blue-600">2h</span>
              <div className="text-sm text-gray-600 mt-1">
                Ago
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataManagementInterface;