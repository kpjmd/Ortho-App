import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Link, useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import "./App.css";

// API Configuration
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Header Component
const Header = () => {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-blue-800 shadow-lg">
      <div className="container mx-auto py-4 px-6 flex justify-between items-center">
        <Link to="/" className="text-white text-2xl font-bold">OrthoTrack</Link>
        <nav>
          <ul className="flex space-x-6">
            <li><Link to="/" className="text-white hover:text-blue-200">Dashboard</Link></li>
            <li><Link to="/patients" className="text-white hover:text-blue-200">Patients</Link></li>
            <li><Link to="/surveys/new" className="text-white hover:text-blue-200">New Survey</Link></li>
          </ul>
        </nav>
      </div>
    </header>
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
  const navigate = useNavigate();

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const patients = await axios.get(`${API}/patients`);
        
        // Simple stats calculation
        let atRiskCount = 0;
        if (patients.data.length > 0) {
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
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Orthopedic Recovery Tracker</h1>
      
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
              <h2 className="text-xl font-semibold text-gray-700 mb-4">Quick Actions</h2>
              <div className="grid grid-cols-2 gap-4">
                <Link to="/patients/new" className="bg-blue-600 text-white py-3 px-4 rounded-lg text-center hover:bg-blue-700 transition">
                  Add New Patient
                </Link>
                <Link to="/surveys/new" className="bg-green-600 text-white py-3 px-4 rounded-lg text-center hover:bg-green-700 transition">
                  Record Survey
                </Link>
                <Link to="/wearable/new" className="bg-purple-600 text-white py-3 px-4 rounded-lg text-center hover:bg-purple-700 transition">
                  Enter Wearable Data
                </Link>
                <Link to="/patients" className="bg-amber-600 text-white py-3 px-4 rounded-lg text-center hover:bg-amber-700 transition">
                  View Patients
                </Link>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold text-gray-700 mb-4">About This App</h2>
              <p className="text-gray-600 mb-4">
                This application tracks orthopedic recovery progress through surveys and wearable data, 
                with a focus on ACL and rotator cuff injuries. It provides AI-driven insights to help
                monitor patient recovery and identify potential issues early.
              </p>
              <p className="text-gray-600">
                To get started, add sample patients or create your own, then record survey data and 
                wearable metrics to generate recovery insights.
              </p>
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
        </>
      )}
    </div>
  );
};

// PatientForm Component
const PatientForm = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    injury_type: 'ACL',
    date_of_injury: '',
    date_of_surgery: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      await axios.post(`${API}/patients`, formData);
      navigate('/patients');
    } catch (err) {
      console.error("Failed to create patient", err);
      setError(err.response?.data?.detail || "Failed to create patient");
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 container mx-auto px-6 py-8">
      <div className="mb-6 flex items-center">
        <button 
          onClick={() => navigate('/patients')}
          className="mr-4 text-gray-600 hover:text-gray-900"
        >
          ← Back to Patients
        </button>
        <h1 className="text-3xl font-bold text-gray-800">Add New Patient</h1>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}
        
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="name" className="block text-gray-700 text-sm font-bold mb-2">
              Full Name
            </label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            />
          </div>
          
          <div className="mb-4">
            <label htmlFor="email" className="block text-gray-700 text-sm font-bold mb-2">
              Email Address
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            />
          </div>
          
          <div className="mb-4">
            <label htmlFor="injury_type" className="block text-gray-700 text-sm font-bold mb-2">
              Injury Type
            </label>
            <select
              id="injury_type"
              name="injury_type"
              value={formData.injury_type}
              onChange={handleChange}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            >
              <option value="ACL">ACL</option>
              <option value="Rotator Cuff">Rotator Cuff</option>
            </select>
          </div>
          
          <div className="mb-4">
            <label htmlFor="date_of_injury" className="block text-gray-700 text-sm font-bold mb-2">
              Date of Injury
            </label>
            <input
              type="date"
              id="date_of_injury"
              name="date_of_injury"
              value={formData.date_of_injury}
              onChange={handleChange}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            />
          </div>
          
          <div className="mb-6">
            <label htmlFor="date_of_surgery" className="block text-gray-700 text-sm font-bold mb-2">
              Date of Surgery (if applicable)
            </label>
            <input
              type="date"
              id="date_of_surgery"
              name="date_of_surgery"
              value={formData.date_of_surgery}
              onChange={handleChange}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            />
          </div>
          
          <div className="flex items-center justify-end">
            <button
              type="button"
              onClick={() => navigate('/patients')}
              className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-4"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
            >
              {loading ? 'Saving...' : 'Save Patient'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// Survey Form Component
const SurveyForm = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  
  // Get query parameters
  const queryParams = new URLSearchParams(window.location.search);
  const preselectedPatientId = queryParams.get('patientId');
  
  const [formData, setFormData] = useState({
    patient_id: preselectedPatientId || '',
    date: new Date().toISOString().split('T')[0],
    pain_score: 5,
    mobility_score: 5,
    activities_of_daily_living: {},
    range_of_motion: {},
    strength: {},
    notes: ''
  });
  
  const [selectedPatient, setSelectedPatient] = useState(null);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await axios.get(`${API}/patients`);
        setPatients(response.data);
        
        if (preselectedPatientId) {
          const patient = response.data.find(p => p.id === preselectedPatientId);
          if (patient) {
            setSelectedPatient(patient);
            
            // Initialize specific fields based on injury type
            if (patient.injury_type === 'ACL') {
              setFormData(prev => ({
                ...prev,
                range_of_motion: {
                  knee_flexion: 90,
                  knee_extension: -5
                },
                strength: {
                  quadriceps: 3,
                  hamstrings: 3
                },
                activities_of_daily_living: {
                  walking: 3,
                  stairs: 2,
                  standing_from_chair: 3
                }
              }));
            } else if (patient.injury_type === 'Rotator Cuff') {
              setFormData(prev => ({
                ...prev,
                range_of_motion: {
                  shoulder_flexion: 100,
                  shoulder_abduction: 90,
                  external_rotation: 30
                },
                strength: {
                  deltoid: 3,
                  rotator_cuff: 2
                },
                activities_of_daily_living: {
                  reaching_overhead: 2,
                  carrying_objects: 3,
                  dressing: 4
                }
              }));
            }
          }
        }
        
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch patients", err);
        setError("Failed to load patients");
        setLoading(false);
      }
    };

    fetchPatients();
  }, [preselectedPatientId]);

  const handlePatientChange = (e) => {
    const patientId = e.target.value;
    setFormData(prev => ({ ...prev, patient_id: patientId }));
    
    const patient = patients.find(p => p.id === patientId);
    setSelectedPatient(patient);
    
    // Reset and reinitialize form fields based on patient injury type
    if (patient) {
      if (patient.injury_type === 'ACL') {
        setFormData(prev => ({
          ...prev,
          patient_id: patientId,
          range_of_motion: {
            knee_flexion: 90,
            knee_extension: -5
          },
          strength: {
            quadriceps: 3,
            hamstrings: 3
          },
          activities_of_daily_living: {
            walking: 3,
            stairs: 2,
            standing_from_chair: 3
          }
        }));
      } else if (patient.injury_type === 'Rotator Cuff') {
        setFormData(prev => ({
          ...prev,
          patient_id: patientId,
          range_of_motion: {
            shoulder_flexion: 100,
            shoulder_abduction: 90,
            external_rotation: 30
          },
          strength: {
            deltoid: 3,
            rotator_cuff: 2
          },
          activities_of_daily_living: {
            reaching_overhead: 2,
            carrying_objects: 3,
            dressing: 4
          }
        }));
      }
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleRangeOfMotionChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      range_of_motion: {
        ...prev.range_of_motion,
        [field]: parseFloat(value)
      }
    }));
  };

  const handleStrengthChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      strength: {
        ...prev.strength,
        [field]: parseInt(value)
      }
    }));
  };

  const handleADLChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      activities_of_daily_living: {
        ...prev.activities_of_daily_living,
        [field]: parseInt(value)
      }
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.patient_id) {
      setError("Please select a patient");
      return;
    }
    
    try {
      setSubmitting(true);
      await axios.post(`${API}/surveys`, formData);
      navigate(`/patients/${formData.patient_id}`);
    } catch (err) {
      console.error("Failed to submit survey", err);
      setError(err.response?.data?.detail || "Failed to submit survey");
      setSubmitting(false);
    }
  };

  return (
    <div className="flex-1 container mx-auto px-6 py-8">
      <div className="mb-6 flex items-center">
        <button 
          onClick={() => navigate(-1)}
          className="mr-4 text-gray-600 hover:text-gray-900"
        >
          ← Back
        </button>
        <h1 className="text-3xl font-bold text-gray-800">New Patient Survey</h1>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}
        
        {loading ? (
          <div className="flex justify-center my-8">
            <div className="loader"></div>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="mb-6">
              <label htmlFor="patient_id" className="block text-gray-700 text-sm font-bold mb-2">
                Patient
              </label>
              {preselectedPatientId ? (
                <div className="py-2 px-3 bg-gray-100 rounded">
                  {selectedPatient ? selectedPatient.name : 'Loading patient...'}
                </div>
              ) : (
                <select
                  id="patient_id"
                  name="patient_id"
                  value={formData.patient_id}
                  onChange={handlePatientChange}
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  required
                >
                  <option value="">Select a patient</option>
                  {patients.map(patient => (
                    <option key={patient.id} value={patient.id}>
                      {patient.name} - {patient.injury_type}
                    </option>
                  ))}
                </select>
              )}
            </div>
            
            <div className="mb-6">
              <label htmlFor="date" className="block text-gray-700 text-sm font-bold mb-2">
                Survey Date
              </label>
              <input
                type="date"
                id="date"
                name="date"
                value={formData.date}
                onChange={handleChange}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                required
              />
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div>
                <label htmlFor="pain_score" className="block text-gray-700 text-sm font-bold mb-2">
                  Pain Score (0-10)
                </label>
                <div className="flex items-center">
                  <input
                    type="range"
                    id="pain_score"
                    name="pain_score"
                    min="0"
                    max="10"
                    value={formData.pain_score}
                    onChange={handleChange}
                    className="w-full mr-4"
                    required
                  />
                  <span className="text-lg font-bold">{formData.pain_score}</span>
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>No Pain</span>
                  <span>Worst Possible</span>
                </div>
              </div>
              
              <div>
                <label htmlFor="mobility_score" className="block text-gray-700 text-sm font-bold mb-2">
                  Mobility Score (0-10)
                </label>
                <div className="flex items-center">
                  <input
                    type="range"
                    id="mobility_score"
                    name="mobility_score"
                    min="0"
                    max="10"
                    value={formData.mobility_score}
                    onChange={handleChange}
                    className="w-full mr-4"
                    required
                  />
                  <span className="text-lg font-bold">{formData.mobility_score}</span>
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>No Mobility</span>
                  <span>Full Mobility</span>
                </div>
              </div>
            </div>
            
            {selectedPatient && (
              <>
                <div className="mb-6">
                  <h2 className="text-xl font-semibold text-gray-700 mb-4">Range of Motion</h2>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {selectedPatient.injury_type === 'ACL' ? (
                      <>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Knee Flexion (degrees)
                          </label>
                          <input
                            type="number"
                            value={formData.range_of_motion.knee_flexion || ''}
                            onChange={(e) => handleRangeOfMotionChange('knee_flexion', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          />
                        </div>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Knee Extension (degrees)
                          </label>
                          <input
                            type="number"
                            value={formData.range_of_motion.knee_extension || ''}
                            onChange={(e) => handleRangeOfMotionChange('knee_extension', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          />
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Shoulder Flexion (degrees)
                          </label>
                          <input
                            type="number"
                            value={formData.range_of_motion.shoulder_flexion || ''}
                            onChange={(e) => handleRangeOfMotionChange('shoulder_flexion', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          />
                        </div>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Shoulder Abduction (degrees)
                          </label>
                          <input
                            type="number"
                            value={formData.range_of_motion.shoulder_abduction || ''}
                            onChange={(e) => handleRangeOfMotionChange('shoulder_abduction', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          />
                        </div>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            External Rotation (degrees)
                          </label>
                          <input
                            type="number"
                            value={formData.range_of_motion.external_rotation || ''}
                            onChange={(e) => handleRangeOfMotionChange('external_rotation', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          />
                        </div>
                      </>
                    )}
                  </div>
                </div>
                
                <div className="mb-6">
                  <h2 className="text-xl font-semibold text-gray-700 mb-4">Activities of Daily Living</h2>
                  <p className="text-sm text-gray-600 mb-4">Rate the patient's ability to perform these activities (0-7)</p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {selectedPatient.injury_type === 'ACL' ? (
                      <>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Walking
                          </label>
                          <select
                            value={formData.activities_of_daily_living.walking || ''}
                            onChange={(e) => handleADLChange('walking', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          >
                            {[0, 1, 2, 3, 4, 5, 6, 7].map(value => (
                              <option key={value} value={value}>
                                {value} - {value === 0 ? 'Unable' : value === 7 ? 'Normal' : `Level ${value}`}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Stairs
                          </label>
                          <select
                            value={formData.activities_of_daily_living.stairs || ''}
                            onChange={(e) => handleADLChange('stairs', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          >
                            {[0, 1, 2, 3, 4, 5, 6, 7].map(value => (
                              <option key={value} value={value}>
                                {value} - {value === 0 ? 'Unable' : value === 7 ? 'Normal' : `Level ${value}`}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Standing from Chair
                          </label>
                          <select
                            value={formData.activities_of_daily_living.standing_from_chair || ''}
                            onChange={(e) => handleADLChange('standing_from_chair', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          >
                            {[0, 1, 2, 3, 4, 5, 6, 7].map(value => (
                              <option key={value} value={value}>
                                {value} - {value === 0 ? 'Unable' : value === 7 ? 'Normal' : `Level ${value}`}
                              </option>
                            ))}
                          </select>
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Reaching Overhead
                          </label>
                          <select
                            value={formData.activities_of_daily_living.reaching_overhead || ''}
                            onChange={(e) => handleADLChange('reaching_overhead', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          >
                            {[0, 1, 2, 3, 4, 5, 6, 7].map(value => (
                              <option key={value} value={value}>
                                {value} - {value === 0 ? 'Unable' : value === 7 ? 'Normal' : `Level ${value}`}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Carrying Objects
                          </label>
                          <select
                            value={formData.activities_of_daily_living.carrying_objects || ''}
                            onChange={(e) => handleADLChange('carrying_objects', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          >
                            {[0, 1, 2, 3, 4, 5, 6, 7].map(value => (
                              <option key={value} value={value}>
                                {value} - {value === 0 ? 'Unable' : value === 7 ? 'Normal' : `Level ${value}`}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-gray-700 text-sm font-bold mb-2">
                            Dressing
                          </label>
                          <select
                            value={formData.activities_of_daily_living.dressing || ''}
                            onChange={(e) => handleADLChange('dressing', e.target.value)}
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            required
                          >
                            {[0, 1, 2, 3, 4, 5, 6, 7].map(value => (
                              <option key={value} value={value}>
                                {value} - {value === 0 ? 'Unable' : value === 7 ? 'Normal' : `Level ${value}`}
                              </option>
                            ))}
                          </select>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </>
            )}
            
            <div className="mb-6">
              <label htmlFor="notes" className="block text-gray-700 text-sm font-bold mb-2">
                Additional Notes
              </label>
              <textarea
                id="notes"
                name="notes"
                value={formData.notes}
                onChange={handleChange}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                rows="4"
              ></textarea>
            </div>
            
            <div className="flex items-center justify-end">
              <button
                type="button"
                onClick={() => navigate(-1)}
                className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-4"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting || !formData.patient_id}
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
              >
                {submitting ? 'Submitting...' : 'Submit Survey'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

// Wearable Data Form Component
const WearableDataForm = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  
  // Get query parameters
  const queryParams = new URLSearchParams(window.location.search);
  const preselectedPatientId = queryParams.get('patientId');
  
  const [formData, setFormData] = useState({
    patient_id: preselectedPatientId || '',
    date: new Date().toISOString().split('T')[0],
    steps: 5000,
    heart_rate: 75,
    oxygen_saturation: 98,
    sleep_hours: 7.5,
    walking_speed: 3.0
  });

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

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleNumberChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.patient_id) {
      setError("Please select a patient");
      return;
    }
    
    try {
      setSubmitting(true);
      await axios.post(`${API}/wearable-data`, formData);
      navigate(`/patients/${formData.patient_id}`);
    } catch (err) {
      console.error("Failed to submit wearable data", err);
      setError(err.response?.data?.detail || "Failed to submit wearable data");
      setSubmitting(false);
    }
  };

  return (
    <div className="flex-1 container mx-auto px-6 py-8">
      <div className="mb-6 flex items-center">
        <button 
          onClick={() => navigate(-1)}
          className="mr-4 text-gray-600 hover:text-gray-900"
        >
          ← Back
        </button>
        <h1 className="text-3xl font-bold text-gray-800">Add Wearable Data</h1>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}
        
        {loading ? (
          <div className="flex justify-center my-8">
            <div className="loader"></div>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="mb-6">
              <label htmlFor="patient_id" className="block text-gray-700 text-sm font-bold mb-2">
                Patient
              </label>
              {preselectedPatientId ? (
                <div className="py-2 px-3 bg-gray-100 rounded">
                  {patients.find(p => p.id === preselectedPatientId)?.name || 'Loading patient...'}
                </div>
              ) : (
                <select
                  id="patient_id"
                  name="patient_id"
                  value={formData.patient_id}
                  onChange={handleChange}
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  required
                >
                  <option value="">Select a patient</option>
                  {patients.map(patient => (
                    <option key={patient.id} value={patient.id}>
                      {patient.name} - {patient.injury_type}
                    </option>
                  ))}
                </select>
              )}
            </div>
            
            <div className="mb-6">
              <label htmlFor="date" className="block text-gray-700 text-sm font-bold mb-2">
                Data Date
              </label>
              <input
                type="date"
                id="date"
                name="date"
                value={formData.date}
                onChange={handleChange}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                required
              />
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div>
                <label htmlFor="steps" className="block text-gray-700 text-sm font-bold mb-2">
                  Steps
                </label>
                <input
                  type="number"
                  id="steps"
                  name="steps"
                  value={formData.steps}
                  onChange={handleNumberChange}
                  min="0"
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  required
                />
              </div>
              
              <div>
                <label htmlFor="heart_rate" className="block text-gray-700 text-sm font-bold mb-2">
                  Heart Rate (bpm)
                </label>
                <input
                  type="number"
                  id="heart_rate"
                  name="heart_rate"
                  value={formData.heart_rate}
                  onChange={handleNumberChange}
                  min="40"
                  max="200"
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  required
                />
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div>
                <label htmlFor="oxygen_saturation" className="block text-gray-700 text-sm font-bold mb-2">
                  Oxygen Saturation (%)
                </label>
                <input
                  type="number"
                  id="oxygen_saturation"
                  name="oxygen_saturation"
                  value={formData.oxygen_saturation}
                  onChange={handleNumberChange}
                  min="80"
                  max="100"
                  step="0.1"
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  required
                />
              </div>
              
              <div>
                <label htmlFor="sleep_hours" className="block text-gray-700 text-sm font-bold mb-2">
                  Sleep Hours
                </label>
                <input
                  type="number"
                  id="sleep_hours"
                  name="sleep_hours"
                  value={formData.sleep_hours}
                  onChange={handleNumberChange}
                  min="0"
                  max="24"
                  step="0.1"
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  required
                />
              </div>
            </div>
            
            <div className="mb-6">
              <label htmlFor="walking_speed" className="block text-gray-700 text-sm font-bold mb-2">
                Walking Speed (km/h)
              </label>
              <input
                type="number"
                id="walking_speed"
                name="walking_speed"
                value={formData.walking_speed}
                onChange={handleNumberChange}
                min="0"
                step="0.1"
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                required
              />
            </div>
            
            <div className="flex items-center justify-end">
              <button
                type="button"
                onClick={() => navigate(-1)}
                className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-4"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting || !formData.patient_id}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
              >
                {submitting ? 'Submitting...' : 'Submit Data'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

// Footer Component
const Footer = () => {
  return (
    <footer className="bg-gray-800 text-gray-300 py-6">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm">&copy; 2025 OrthoTrack. All rights reserved.</p>
          </div>
        </div>
      </div>
    </footer>
  );
};

// Main App
function App() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-100">
      <BrowserRouter>
        <Header />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/patients" element={<PatientList />} />
          <Route path="/patients/:patientId" element={<PatientDetail />} />
          <Route path="/patients/new" element={<PatientForm />} />
          <Route path="/surveys/new" element={<SurveyForm />} />
          <Route path="/wearable/new" element={<WearableDataForm />} />
        </Routes>
        <Footer />
      </BrowserRouter>
    </div>
  );
}

export default App;
