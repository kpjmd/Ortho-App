import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Link, useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import "./App.css";

// API Configuration
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Dashboard Components
const Header = () => {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-blue-800 shadow-lg">
      <div className="container mx-auto py-4 px-6 flex justify-between items-center">
        <Link to="/" className="text-white text-2xl font-bold">OrthoTrack</Link>
        <nav>
          <ul className="flex space-x-6">
            <li><Link to="/" className="text-white hover:text-blue-200">Dashboard</Link></li>
            <li><Link to="/patients" className="text-white hover:text-blue-200">Patients</Link></li>
            <li><Link to="/surveys" className="text-white hover:text-blue-200">Surveys</Link></li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-gray-300 py-6">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm">&copy; 2025 OrthoTrack. All rights reserved.</p>
          </div>
          <div className="flex space-x-4">
            <a href="#" className="text-gray-300 hover:text-white">Privacy Policy</a>
            <a href="#" className="text-gray-300 hover:text-white">Terms of Service</a>
            <a href="#" className="text-gray-300 hover:text-white">Contact</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

// Dashboard Home
const Dashboard = () => {
  const [stats, setStats] = useState({
    patientCount: 0,
    surveyCount: 0,
    atRiskCount: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [patients, insights] = await Promise.all([
          axios.get(`${API}/patients`),
          axios.get(`${API}/patients`)
        ]);
        
        // Count patients by status (simplified)
        let atRiskCount = 0;
        if (patients.data.length > 0) {
          // Just for demo purposes, we'll flag a third of patients as at risk
          atRiskCount = Math.floor(patients.data.length / 3);
        }
        
        setStats({
          patientCount: patients.data.length,
          surveyCount: patients.data.length * 5, // Approximation
          atRiskCount
        });
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch dashboard data", err);
        setError("Failed to load dashboard data");
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  return (
    <div className="flex-1 container mx-auto px-6 py-8">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Recovery Insights Dashboard</h1>
      
      {loading ? (
        <div className="flex justify-center">
          <div className="loader"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-700 mb-2">Active Patients</h2>
              <p className="text-4xl font-bold text-blue-600">{stats.patientCount}</p>
              <p className="text-gray-500 mt-2">Total patients in recovery</p>
            </div>
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-700 mb-2">Surveys Collected</h2>
              <p className="text-4xl font-bold text-green-600">{stats.surveyCount}</p>
              <p className="text-gray-500 mt-2">Total survey responses</p>
            </div>
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-700 mb-2">At-Risk Patients</h2>
              <p className="text-4xl font-bold text-red-600">{stats.atRiskCount}</p>
              <p className="text-gray-500 mt-2">Patients requiring attention</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-700 mb-4">Recent Activity</h2>
              <div className="space-y-4">
                <div className="border-l-4 border-blue-500 pl-4 py-2">
                  <p className="text-gray-600">New patient added: John Smith</p>
                  <p className="text-sm text-gray-500">ACL Injury • 2 hours ago</p>
                </div>
                <div className="border-l-4 border-green-500 pl-4 py-2">
                  <p className="text-gray-600">Survey completed: Sarah Johnson</p>
                  <p className="text-sm text-gray-500">Rotator Cuff • 3 hours ago</p>
                </div>
                <div className="border-l-4 border-yellow-500 pl-4 py-2">
                  <p className="text-gray-600">Wearable data synced: Michael Brown</p>
                  <p className="text-sm text-gray-500">ACL Injury • 5 hours ago</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-700 mb-4">Quick Actions</h2>
              <div className="grid grid-cols-2 gap-4">
                <Link to="/patients/new" className="bg-blue-600 text-white py-3 px-4 rounded-lg text-center hover:bg-blue-700 transition">
                  Add New Patient
                </Link>
                <Link to="/surveys/new" className="bg-green-600 text-white py-3 px-4 rounded-lg text-center hover:bg-green-700 transition">
                  Record Survey
                </Link>
                <Link to="/wearable/sync" className="bg-purple-600 text-white py-3 px-4 rounded-lg text-center hover:bg-purple-700 transition">
                  Sync Wearables
                </Link>
                <Link to="/insights" className="bg-amber-600 text-white py-3 px-4 rounded-lg text-center hover:bg-amber-700 transition">
                  View Insights
                </Link>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-700 mb-4">Recovery Progress Overview</h2>
            <div className="p-4 bg-gray-100 rounded-lg mb-4 flex items-center">
              <div className="text-gray-600">
                This is a placeholder for data visualizations showing recovery trends across patients.
                In a production app, this would display charts for range of motion, pain levels, and 
                activity metrics over time.
              </div>
            </div>
            <div className="flex justify-end">
              <button className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition">
                View Detailed Analytics
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// Patient Components
const PatientList = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await axios.get(`${API}/patients`);
        setPatients(response.data);
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch patients", err);
        setError("Failed to load patients");
        setLoading(false);
      }
    };

    fetchPatients();
  }, []);

  const handleLoadSampleData = async () => {
    try {
      setLoading(true);
      await axios.post(`${API}/sample-data`);
      const response = await axios.get(`${API}/patients`);
      setPatients(response.data);
      setLoading(false);
    } catch (err) {
      console.error("Failed to load sample data", err);
      setError("Failed to load sample data");
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 container mx-auto px-6 py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Patients</h1>
        <div className="space-x-4">
          <button 
            onClick={handleLoadSampleData}
            className="bg-gray-600 text-white py-2 px-4 rounded hover:bg-gray-700 transition"
          >
            Load Sample Data
          </button>
          <button 
            onClick={() => navigate('/patients/new')}
            className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition"
          >
            Add New Patient
          </button>
        </div>
      </div>
      
      {loading ? (
        <div className="flex justify-center">
          <div className="loader"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      ) : patients.length === 0 ? (
        <div className="bg-white rounded-lg shadow-md p-8 text-center">
          <h2 className="text-xl font-semibold text-gray-700 mb-4">No Patients Found</h2>
          <p className="text-gray-600 mb-6">Start by adding your first patient or load sample data.</p>
          <div className="flex justify-center space-x-4">
            <button 
              onClick={() => navigate('/patients/new')}
              className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition"
            >
              Add New Patient
            </button>
            <button 
              onClick={handleLoadSampleData}
              className="bg-gray-600 text-white py-2 px-4 rounded hover:bg-gray-700 transition"
            >
              Load Sample Data
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Email
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Injury Type
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Surgery Date
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {patients.map((patient) => (
                <tr key={patient.id} onClick={() => navigate(`/patients/${patient.id}`)} className="hover:bg-gray-50 cursor-pointer">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">{patient.name}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-500">{patient.email}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      patient.injury_type === 'ACL' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {patient.injury_type}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-500">
                      {patient.date_of_surgery ? new Date(patient.date_of_surgery).toLocaleDateString() : 'N/A'}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/patients/${patient.id}`);
                      }}
                      className="text-blue-600 hover:text-blue-900 mr-3"
                    >
                      View
                    </button>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/surveys/new?patientId=${patient.id}`);
                      }}
                      className="text-green-600 hover:text-green-900"
                    >
                      Add Survey
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const PatientDetail = () => {
  const { patientId } = useParams();
  const [patient, setPatient] = useState(null);
  const [surveys, setSurveys] = useState([]);
  const [wearableData, setWearableData] = useState([]);
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        // Fetch patient details and their associated data
        const [patientRes, surveysRes, wearableRes, insightsRes] = await Promise.all([
          axios.get(`${API}/patients/${patientId}`),
          axios.get(`${API}/surveys/${patientId}`),
          axios.get(`${API}/wearable-data/${patientId}`),
          axios.get(`${API}/insights/${patientId}`)
        ]);
        
        setPatient(patientRes.data);
        setSurveys(surveysRes.data);
        setWearableData(wearableRes.data);
        setInsights(insightsRes.data);
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch patient data", err);
        setError("Failed to load patient data");
        setLoading(false);
      }
    };

    if (patientId) {
      fetchPatientData();
    }
  }, [patientId]);

  // Helper function to get the latest data item
  const getLatestData = (dataArray) => {
    if (!dataArray || dataArray.length === 0) return null;
    return dataArray.sort((a, b) => new Date(b.date) - new Date(a.date))[0];
  };

  const latestSurvey = getLatestData(surveys);
  const latestWearable = getLatestData(wearableData);
  const latestInsight = getLatestData(insights);

  return (
    <div className="flex-1 container mx-auto px-6 py-8">
      {loading ? (
        <div className="flex justify-center">
          <div className="loader"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      ) : !patient ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          Patient not found
        </div>
      ) : (
        <>
          <div className="mb-6">
            <div className="flex justify-between items-center">
              <h1 className="text-3xl font-bold text-gray-800">{patient.name}</h1>
              <div className="flex space-x-4">
                <button 
                  onClick={() => navigate(`/surveys/new?patientId=${patient.id}`)}
                  className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition"
                >
                  Add Survey
                </button>
                <button 
                  onClick={() => navigate(`/wearable/new?patientId=${patient.id}`)}
                  className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition"
                >
                  Add Wearable Data
                </button>
              </div>
            </div>
            <div className="flex flex-wrap mt-2">
              <span className={`mr-4 px-3 py-1 rounded-full text-sm font-medium ${
                patient.injury_type === 'ACL' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
              }`}>
                {patient.injury_type}
              </span>
              <span className="mr-4 text-gray-600">
                <strong>Injury Date:</strong> {new Date(patient.date_of_injury).toLocaleDateString()}
              </span>
              {patient.date_of_surgery && (
                <span className="mr-4 text-gray-600">
                  <strong>Surgery Date:</strong> {new Date(patient.date_of_surgery).toLocaleDateString()}
                </span>
              )}
              <span className="text-gray-600">
                <strong>Email:</strong> {patient.email}
              </span>
            </div>
          </div>

          <div className="mb-6">
            <div className="border-b border-gray-200">
              <nav className="-mb-px flex">
                <button
                  onClick={() => setActiveTab('overview')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'overview'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Overview
                </button>
                <button
                  onClick={() => setActiveTab('surveys')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'surveys'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Surveys
                </button>
                <button
                  onClick={() => setActiveTab('wearable')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'wearable'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Wearable Data
                </button>
                <button
                  onClick={() => setActiveTab('insights')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'insights'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Insights
                </button>
              </nav>
            </div>
          </div>

          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold text-gray-700 mb-4">Recovery Status</h2>
                {latestInsight ? (
                  <>
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium mb-3 ${
                      latestInsight.recovery_status === 'On Track' ? 'bg-green-100 text-green-800' : 
                      latestInsight.recovery_status === 'At Risk' ? 'bg-yellow-100 text-yellow-800' : 
                      'bg-red-100 text-red-800'
                    }`}>
                      {latestInsight.recovery_status}
                    </div>
                    <div className="mb-2">
                      <div className="flex justify-between mb-1">
                        <span className="text-gray-700 text-sm">Recovery Progress</span>
                        <span className="text-gray-700 text-sm">{latestInsight.progress_percentage.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className={`h-2.5 rounded-full ${
                            latestInsight.recovery_status === 'On Track' ? 'bg-green-600' : 
                            latestInsight.recovery_status === 'At Risk' ? 'bg-yellow-500' : 
                            'bg-red-600'
                          }`}
                          style={{ width: `${latestInsight.progress_percentage}%` }}
                        ></div>
                      </div>
                    </div>
                    {latestInsight.risk_factors.length > 0 && (
                      <div className="mt-4">
                        <h3 className="text-sm font-medium text-gray-700 mb-2">Risk Factors:</h3>
                        <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                          {latestInsight.risk_factors.map((factor, index) => (
                            <li key={index}>{factor}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="text-gray-500">No insights available yet</p>
                )}
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold text-gray-700 mb-4">Latest Survey Results</h2>
                {latestSurvey ? (
                  <>
                    <p className="text-gray-500 text-sm mb-3">
                      Recorded on {new Date(latestSurvey.date).toLocaleDateString()}
                    </p>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-gray-700 text-sm">Pain Level</span>
                          <span className="text-gray-700 text-sm">{latestSurvey.pain_score}/10</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className={`h-2.5 rounded-full ${
                              latestSurvey.pain_score <= 3 ? 'bg-green-600' : 
                              latestSurvey.pain_score <= 6 ? 'bg-yellow-500' : 
                              'bg-red-600'
                            }`}
                            style={{ width: `${latestSurvey.pain_score * 10}%` }}
                          ></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-gray-700 text-sm">Mobility Score</span>
                          <span className="text-gray-700 text-sm">{latestSurvey.mobility_score}/10</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="h-2.5 rounded-full bg-blue-600"
                            style={{ width: `${latestSurvey.mobility_score * 10}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="text-gray-500">No survey data available yet</p>
                )}
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold text-gray-700 mb-4">Latest Wearable Data</h2>
                {latestWearable ? (
                  <>
                    <p className="text-gray-500 text-sm mb-3">
                      Recorded on {new Date(latestWearable.date).toLocaleDateString()}
                    </p>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Steps</span>
                        <span className="font-medium">{latestWearable.steps.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Heart Rate</span>
                        <span className="font-medium">{latestWearable.heart_rate} bpm</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">O₂ Saturation</span>
                        <span className="font-medium">{latestWearable.oxygen_saturation}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Sleep</span>
                        <span className="font-medium">{latestWearable.sleep_hours} hours</span>
                      </div>
                      {latestWearable.walking_speed && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Walking Speed</span>
                          <span className="font-medium">{latestWearable.walking_speed.toFixed(1)} km/h</span>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <p className="text-gray-500">No wearable data available yet</p>
                )}
              </div>
            </div>
          )}

          {activeTab === 'surveys' && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-700">Survey History</h2>
                <button 
                  onClick={() => navigate(`/surveys/new?patientId=${patient.id}`)}
                  className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition"
                >
                  Add New Survey
                </button>
              </div>
              
              {surveys.length === 0 ? (
                <p className="text-gray-500">No survey data available yet</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Date
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Pain Score
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Mobility Score
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Range of Motion
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Notes
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {surveys.map((survey) => (
                        <tr key={survey.id}>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{new Date(survey.date).toLocaleDateString()}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className={`text-sm ${
                              survey.pain_score <= 3 ? 'text-green-600' : 
                              survey.pain_score <= 6 ? 'text-yellow-600' : 
                              'text-red-600'
                            }`}>
                              {survey.pain_score}/10
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{survey.mobility_score}/10</div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="text-sm text-gray-900">
                              {Object.entries(survey.range_of_motion).map(([key, value]) => (
                                <div key={key}>
                                  {key.replace(/_/g, ' ')}: {value.toFixed(1)}°
                                </div>
                              ))}
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="text-sm text-gray-900">{survey.notes || 'None'}</div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {activeTab === 'wearable' && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-700">Wearable Data History</h2>
                <button 
                  onClick={() => navigate(`/wearable/new?patientId=${patient.id}`)}
                  className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition"
                >
                  Add Wearable Data
                </button>
              </div>
              
              {wearableData.length === 0 ? (
                <p className="text-gray-500">No wearable data available yet</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Date
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Steps
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Heart Rate
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          O₂ Saturation
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Sleep Hours
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Walking Speed
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {wearableData.map((data) => (
                        <tr key={data.id}>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{new Date(data.date).toLocaleDateString()}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{data.steps.toLocaleString()}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{data.heart_rate} bpm</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{data.oxygen_saturation}%</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{data.sleep_hours} hours</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">
                              {data.walking_speed ? `${data.walking_speed.toFixed(1)} km/h` : 'N/A'}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {activeTab === 'insights' && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-700">Recovery Insights</h2>
                <button 
                  onClick={async () => {
                    try {
                      await axios.post(`${API}/generate-insights/${patient.id}`);
                      const insightsRes = await axios.get(`${API}/insights/${patient.id}`);
                      setInsights(insightsRes.data);
                    } catch (err) {
                      console.error("Failed to generate insights", err);
                    }
                  }}
                  className="bg-purple-600 text-white py-2 px-4 rounded hover:bg-purple-700 transition"
                >
                  Generate New Insights
                </button>
              </div>
              
              {insights.length === 0 ? (
                <p className="text-gray-500">No insights available yet</p>
              ) : (
                <div className="space-y-6">
                  {insights.map((insight) => (
                    <div key={insight.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                            insight.recovery_status === 'On Track' ? 'bg-green-100 text-green-800' : 
                            insight.recovery_status === 'At Risk' ? 'bg-yellow-100 text-yellow-800' : 
                            'bg-red-100 text-red-800'
                          }`}>
                            {insight.recovery_status}
                          </div>
                          <span className="ml-3 text-sm text-gray-500">
                            {new Date(insight.date).toLocaleDateString()}
                          </span>
                        </div>
                        <div className="text-sm font-medium text-blue-600">
                          {insight.progress_percentage.toFixed(1)}% recovered
                        </div>
                      </div>
                      
                      {insight.risk_factors.length > 0 && (
                        <div className="mb-3">
                          <h3 className="text-sm font-medium text-gray-700 mb-2">Risk Factors:</h3>
                          <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                            {insight.risk_factors.map((factor, index) => (
                              <li key={index}>{factor}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {insight.recommendations.length > 0 && (
                        <div>
                          <h3 className="text-sm font-medium text-gray-700 mb-2">Recommendations:</h3>
                          <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                            {insight.recommendations.map((rec, index) => (
                              <li key={index}>{rec}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};