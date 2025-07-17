import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Link, useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import "./App.css";
import ScoreTrendsChart from "./components/ScoreTrendsChart";
import ScoreSummaryCard from "./components/ScoreSummaryCard";
import RecoveryTrajectoryChart from "./components/RecoveryTrajectoryChart";
import InsightCard from "./components/InsightCard";
import RiskIndicator from "./components/RiskIndicator";
import WearableDataOverview from "./components/wearable/WearableDataOverview";
import { 
  getKOOSColors, 
  getASESColors, 
  getKOOSDisplayNames, 
  getASESDisplayNames, 
  needsProAssessment 
} from "./utils/proScoreHelpers";

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
                      patient.diagnosis_type && patient.diagnosis_type.includes('Knee') || 
                      ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(patient.diagnosis_type) 
                      ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {patient.diagnosis_type || patient.injury_type}
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
  const [proScores, setProScores] = useState([]);
  const [latestProScore, setLatestProScore] = useState(null);
  const [previousProScore, setPreviousProScore] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        // Fetch patient details first to determine knee vs shoulder
        const patientRes = await axios.get(`${API}/patients/${patientId}`);
        const patientData = patientRes.data;
        
        // Determine patient type
        const isKneePatient = patientData.diagnosis_type && 
          ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(patientData.diagnosis_type) ||
          patientData.injury_type === 'ACL';
        
        // Fetch PRO scores based on patient type
        const proScoreEndpoint = isKneePatient ? 
          `${API}/koos/${patientId}` : 
          `${API}/ases/${patientId}`;
        
        // Fetch all data in parallel
        const [surveysRes, wearableRes, insightsRes, proScoresRes] = await Promise.all([
          axios.get(`${API}/surveys/${patientId}`),
          axios.get(`${API}/wearable-data/${patientId}`),
          axios.get(`${API}/insights/${patientId}`),
          axios.get(proScoreEndpoint).catch(() => ({ data: [] })) // Don't fail if no PRO data
        ]);
        
        setPatient(patientData);
        setSurveys(surveysRes.data);
        setWearableData(wearableRes.data);
        setInsights(insightsRes.data);
        setProScores(proScoresRes.data);
        
        // Set latest and previous PRO scores
        const sortedScores = proScoresRes.data.sort((a, b) => new Date(b.date) - new Date(a.date));
        setLatestProScore(sortedScores[0] || null);
        setPreviousProScore(sortedScores[1] || null);
        
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
                  onClick={() => navigate(`/pro-assessment/${patient.id}`)}
                  className="bg-purple-600 text-white py-2 px-4 rounded hover:bg-purple-700 transition"
                >
                  PRO Assessment
                </button>
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
                patient.diagnosis_type && patient.diagnosis_type.includes('Knee') || 
                ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(patient.diagnosis_type) 
                ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
              }`}>
                {patient.diagnosis_type || patient.injury_type}
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
                  onClick={() => setActiveTab('analytics')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'analytics'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Analytics
                </button>
              </nav>
            </div>
          </div>

          {activeTab === 'overview' && (
            <>
              {/* PRO Score Summary Section */}
              {latestProScore && (
                <div className="mb-8">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-700">PRO Scores</h2>
                    {needsProAssessment(latestProScore?.date) && (
                      <button 
                        onClick={() => navigate(`/pro-assessment/${patient.id}`)}
                        className="bg-purple-600 text-white py-2 px-4 rounded hover:bg-purple-700 transition text-sm"
                      >
                        Complete Assessment
                      </button>
                    )}
                  </div>
                  
                  {(() => {
                    const isKneePatient = patient.diagnosis_type && 
                      ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(patient.diagnosis_type) ||
                      patient.injury_type === 'ACL';
                    
                    const colors = isKneePatient ? getKOOSColors() : getASESColors();
                    const displayNames = isKneePatient ? getKOOSDisplayNames() : getASESDisplayNames();
                    
                    if (isKneePatient) {
                      return (
                        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
                          <ScoreSummaryCard
                            title={displayNames.symptoms_score}
                            currentScore={latestProScore?.symptoms_score}
                            previousScore={previousProScore?.symptoms_score}
                            lastAssessmentDate={latestProScore?.date}
                            color={colors.symptoms_score}
                          />
                          <ScoreSummaryCard
                            title={displayNames.pain_score}
                            currentScore={latestProScore?.pain_score}
                            previousScore={previousProScore?.pain_score}
                            lastAssessmentDate={latestProScore?.date}
                            color={colors.pain_score}
                          />
                          <ScoreSummaryCard
                            title={displayNames.adl_score}
                            currentScore={latestProScore?.adl_score}
                            previousScore={previousProScore?.adl_score}
                            lastAssessmentDate={latestProScore?.date}
                            color={colors.adl_score}
                          />
                          <ScoreSummaryCard
                            title={displayNames.sport_score}
                            currentScore={latestProScore?.sport_score}
                            previousScore={previousProScore?.sport_score}
                            lastAssessmentDate={latestProScore?.date}
                            color={colors.sport_score}
                          />
                          <ScoreSummaryCard
                            title={displayNames.qol_score}
                            currentScore={latestProScore?.qol_score}
                            previousScore={previousProScore?.qol_score}
                            lastAssessmentDate={latestProScore?.date}
                            color={colors.qol_score}
                          />
                        </div>
                      );
                    } else {
                      return (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                          <ScoreSummaryCard
                            title={displayNames.total_score}
                            currentScore={latestProScore?.total_score}
                            previousScore={previousProScore?.total_score}
                            lastAssessmentDate={latestProScore?.date}
                            color={colors.total_score}
                          />
                          <ScoreSummaryCard
                            title={displayNames.pain_component}
                            currentScore={latestProScore?.pain_component}
                            previousScore={previousProScore?.pain_component}
                            lastAssessmentDate={latestProScore?.date}
                            color={colors.pain_component}
                          />
                          <ScoreSummaryCard
                            title={displayNames.function_component}
                            currentScore={latestProScore?.function_component}
                            previousScore={previousProScore?.function_component}
                            lastAssessmentDate={latestProScore?.date}
                            color={colors.function_component}
                          />
                        </div>
                      );
                    }
                  })()}
                </div>
              )}
              
              {/* Score Trends Chart */}
              {patient && (
                <div className="mb-8">
                  <ScoreTrendsChart 
                    patientId={patient.id}
                    patientType={
                      patient.diagnosis_type && 
                      ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(patient.diagnosis_type) ||
                      patient.injury_type === 'ACL' ? 'knee' : 'shoulder'
                    }
                  />
                </div>
              )}
              
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

            {/* Enhanced AI Insights Section */}
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-700 mb-4">AI-Powered Recovery Insights</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="lg:col-span-1">
                  <InsightCard patientId={patient.id} />
                </div>
                <div className="lg:col-span-1">
                  <RiskIndicator patientId={patient.id} />
                </div>
              </div>
            </div>

            {/* Recovery Trajectory Section */}
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-gray-700 mb-4">Recovery Trajectory Analysis</h2>
              <RecoveryTrajectoryChart patientId={patient.id} />
            </div>
            </>
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
          {activeTab === 'analytics' && (
            <WearableDataOverview patientId={patientId} patient={patient} />
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
    diagnosis_type: 'ACL Tear',
    date_of_injury: '',
    date_of_surgery: ''
  });
  const [selectedBodyPart, setSelectedBodyPart] = useState('knee');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleBodyPartChange = (bodyPart) => {
    setSelectedBodyPart(bodyPart);
    // Set default diagnosis for selected body part
    const defaultDiagnosis = bodyPart === 'knee' ? 'ACL Tear' : 'Rotator Cuff Tear';
    setFormData(prev => ({ ...prev, diagnosis_type: defaultDiagnosis }));
  };

  const getAvailableDiagnoses = () => {
    if (selectedBodyPart === 'knee') {
      return [
        'ACL Tear',
        'Meniscus Tear',
        'Cartilage Defect',
        'Knee Osteoarthritis',
        'Post Total Knee Replacement'
      ];
    } else {
      return [
        'Rotator Cuff Tear',
        'Labral Tear',
        'Shoulder Instability',
        'Shoulder Osteoarthritis',
        'Post Total Shoulder Replacement'
      ];
    }
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
            <label className="block text-gray-700 text-sm font-bold mb-2">
              Body Part
            </label>
            <div className="flex space-x-4 mb-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="knee"
                  checked={selectedBodyPart === 'knee'}
                  onChange={(e) => handleBodyPartChange(e.target.value)}
                  className="mr-2"
                />
                Knee
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="shoulder"
                  checked={selectedBodyPart === 'shoulder'}
                  onChange={(e) => handleBodyPartChange(e.target.value)}
                  className="mr-2"
                />
                Shoulder
              </label>
            </div>
          </div>
          
          <div className="mb-4">
            <label htmlFor="diagnosis_type" className="block text-gray-700 text-sm font-bold mb-2">
              Diagnosis
            </label>
            <select
              id="diagnosis_type"
              name="diagnosis_type"
              value={formData.diagnosis_type}
              onChange={handleChange}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            >
              {getAvailableDiagnoses().map(diagnosis => (
                <option key={diagnosis} value={diagnosis}>{diagnosis}</option>
              ))}
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
            
            // Initialize specific fields based on diagnosis type
            const isKneePatient = patient.diagnosis_type && 
              ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(patient.diagnosis_type) ||
              patient.injury_type === 'ACL';
            
            if (isKneePatient) {
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
            } else {
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
    
    // Reset and reinitialize form fields based on patient diagnosis type
    if (patient) {
      const isKneePatient = patient.diagnosis_type && 
        ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(patient.diagnosis_type) ||
        patient.injury_type === 'ACL';
      
      if (isKneePatient) {
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
      } else {
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
                      {patient.name} - {patient.diagnosis_type || patient.injury_type}
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
                    {selectedPatient.diagnosis_type && 
                      ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(selectedPatient.diagnosis_type) ||
                      selectedPatient.injury_type === 'ACL' ? (
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
                    {selectedPatient.diagnosis_type && 
                      ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(selectedPatient.diagnosis_type) ||
                      selectedPatient.injury_type === 'ACL' ? (
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

// KOOSForm Component
const KOOSForm = ({ patient }) => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  const [formData, setFormData] = useState({
    patient_id: patient.id,
    date: new Date().toISOString().split('T')[0],
    // Symptoms subscale
    s1_swelling: 0,
    s2_grinding: 0,
    s3_catching: 0,
    s4_straighten: 0,
    s5_bend: 0,
    s6_stiffness_morning: 0,
    s7_stiffness_later: 0,
    // Pain subscale
    p1_frequency: 0,
    p2_twisting: 0,
    p3_straightening: 0,
    p4_bending: 0,
    p5_walking_flat: 0,
    p6_stairs: 0,
    p7_night: 0,
    p8_sitting: 0,
    p9_standing: 0,
    // ADL subscale
    a1_descending_stairs: 0,
    a2_ascending_stairs: 0,
    a3_rising_sitting: 0,
    a4_standing: 0,
    a5_bending_floor: 0,
    a6_walking_flat: 0,
    a7_car: 0,
    a8_shopping: 0,
    a9_socks_on: 0,
    a10_rising_bed: 0,
    a11_socks_off: 0,
    a12_lying_bed: 0,
    a13_bath: 0,
    a14_sitting: 0,
    a15_toilet: 0,
    a16_heavy_duties: 0,
    a17_light_duties: 0,
    // Sport/Recreation subscale
    sp1_squatting: 0,
    sp2_running: 0,
    sp3_jumping: 0,
    sp4_twisting: 0,
    sp5_kneeling: 0,
    // Quality of Life subscale
    q1_awareness: 0,
    q2_lifestyle: 0,
    q3_confidence: 0,
    q4_difficulty: 0
  });

  const responseLabels = ['None', 'Mild', 'Moderate', 'Severe', 'Extreme'];

  const calculateSubscaleScore = (items) => {
    const values = items.map(item => formData[item]).filter(val => val !== null);
    if (values.length === 0) return 0;
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.round(100 - (mean * 25));
  };

  const symptomsScore = calculateSubscaleScore(['s1_swelling', 's2_grinding', 's3_catching', 's4_straighten', 's5_bend', 's6_stiffness_morning', 's7_stiffness_later']);
  const painScore = calculateSubscaleScore(['p1_frequency', 'p2_twisting', 'p3_straightening', 'p4_bending', 'p5_walking_flat', 'p6_stairs', 'p7_night', 'p8_sitting', 'p9_standing']);
  const adlScore = calculateSubscaleScore(['a1_descending_stairs', 'a2_ascending_stairs', 'a3_rising_sitting', 'a4_standing', 'a5_bending_floor', 'a6_walking_flat', 'a7_car', 'a8_shopping', 'a9_socks_on', 'a10_rising_bed', 'a11_socks_off', 'a12_lying_bed', 'a13_bath', 'a14_sitting', 'a15_toilet', 'a16_heavy_duties', 'a17_light_duties']);
  const sportScore = calculateSubscaleScore(['sp1_squatting', 'sp2_running', 'sp3_jumping', 'sp4_twisting', 'sp5_kneeling']);
  const qolScore = calculateSubscaleScore(['q1_awareness', 'q2_lifestyle', 'q3_confidence', 'q4_difficulty']);

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: parseInt(value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      await axios.post(`${API}/koos`, formData);
      setSuccess(true);
      setTimeout(() => {
        navigate(`/patients/${patient.id}`);
      }, 2000);
    } catch (err) {
      console.error("Failed to submit KOOS", err);
      setError(err.response?.data?.detail || "Failed to submit KOOS questionnaire");
      setLoading(false);
    }
  };

  const renderQuestionGroup = (title, questions) => (
    <div className="mb-8">
      <h3 className="text-lg font-semibold text-gray-700 mb-4">{title}</h3>
      <div className="space-y-4">
        {questions.map(({ field, question }) => (
          <div key={field} className="bg-gray-50 p-4 rounded">
            <p className="text-sm font-medium text-gray-700 mb-3">{question}</p>
            <div className="flex flex-wrap gap-4">
              {responseLabels.map((label, index) => (
                <label key={index} className="flex items-center">
                  <input
                    type="radio"
                    name={field}
                    value={index}
                    checked={formData[field] === index}
                    onChange={(e) => handleChange(field, e.target.value)}
                    className="mr-2"
                  />
                  <span className="text-sm">{index} - {label}</span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  if (success) {
    return (
      <div className="flex-1 container mx-auto px-6 py-8">
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          KOOS questionnaire submitted successfully! Redirecting to patient dashboard...
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 container mx-auto px-6 py-8">
      <div className="mb-6 flex items-center">
        <button 
          onClick={() => navigate(`/patients/${patient.id}`)}
          className="mr-4 text-gray-600 hover:text-gray-900"
        >
          ← Back to {patient.name}
        </button>
        <h1 className="text-3xl font-bold text-gray-800">KOOS Questionnaire</h1>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="mb-6">
          <p className="text-gray-600 mb-4">
            <strong>Patient:</strong> {patient.name} - {patient.diagnosis_type || patient.injury_type}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-center bg-blue-50 p-4 rounded">
            <div>
              <h4 className="text-sm font-medium text-gray-700">Symptoms</h4>
              <p className="text-2xl font-bold text-blue-600">{symptomsScore}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-700">Pain</h4>
              <p className="text-2xl font-bold text-blue-600">{painScore}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-700">Daily Living</h4>
              <p className="text-2xl font-bold text-blue-600">{adlScore}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-700">Sport</h4>
              <p className="text-2xl font-bold text-blue-600">{sportScore}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-700">Quality of Life</h4>
              <p className="text-2xl font-bold text-blue-600">{qolScore}</p>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <label htmlFor="date" className="block text-gray-700 text-sm font-bold mb-2">
              Assessment Date
            </label>
            <input
              type="date"
              id="date"
              value={formData.date}
              onChange={(e) => setFormData(prev => ({ ...prev, date: e.target.value }))}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            />
          </div>

          {renderQuestionGroup("Symptoms", [
            { field: 's1_swelling', question: 'Do you have swelling in your knee?' },
            { field: 's2_grinding', question: 'Do you feel grinding, hear clicking or any other type of noise when your knee moves?' },
            { field: 's3_catching', question: 'Does your knee catch or hang up when moving?' },
            { field: 's4_straighten', question: 'Can you straighten your knee fully?' },
            { field: 's5_bend', question: 'Can you bend your knee fully?' },
            { field: 's6_stiffness_morning', question: 'How severe is your knee stiffness after first wakening in the morning?' },
            { field: 's7_stiffness_later', question: 'How severe is your knee stiffness after sitting, lying or resting later in the day?' }
          ])}

          {renderQuestionGroup("Pain", [
            { field: 'p1_frequency', question: 'How often do you experience knee pain?' },
            { field: 'p2_twisting', question: 'Twisting/pivoting on your knee' },
            { field: 'p3_straightening', question: 'Straightening knee fully' },
            { field: 'p4_bending', question: 'Bending knee fully' },
            { field: 'p5_walking_flat', question: 'Walking on flat surface' },
            { field: 'p6_stairs', question: 'Going up or down stairs' },
            { field: 'p7_night', question: 'At night while in bed' },
            { field: 'p8_sitting', question: 'Sitting or lying' },
            { field: 'p9_standing', question: 'Standing upright' }
          ])}

          {renderQuestionGroup("Activities of Daily Living", [
            { field: 'a1_descending_stairs', question: 'Descending stairs' },
            { field: 'a2_ascending_stairs', question: 'Ascending stairs' },
            { field: 'a3_rising_sitting', question: 'Rising from sitting' },
            { field: 'a4_standing', question: 'Standing' },
            { field: 'a5_bending_floor', question: 'Bending to floor/pick up an object' },
            { field: 'a6_walking_flat', question: 'Walking on flat surface' },
            { field: 'a7_car', question: 'Getting in/out of car' },
            { field: 'a8_shopping', question: 'Going shopping' },
            { field: 'a9_socks_on', question: 'Putting on socks/stockings' },
            { field: 'a10_rising_bed', question: 'Rising from bed' },
            { field: 'a11_socks_off', question: 'Taking off socks/stockings' },
            { field: 'a12_lying_bed', question: 'Lying in bed (turning over, maintaining knee position)' },
            { field: 'a13_bath', question: 'Getting in/out of bath' },
            { field: 'a14_sitting', question: 'Sitting' },
            { field: 'a15_toilet', question: 'Getting on/off toilet' },
            { field: 'a16_heavy_duties', question: 'Heavy domestic duties (moving heavy boxes, scrubbing floors, etc)' },
            { field: 'a17_light_duties', question: 'Light domestic duties (cooking, dusting, etc)' }
          ])}

          {renderQuestionGroup("Sport and Recreation", [
            { field: 'sp1_squatting', question: 'Squatting' },
            { field: 'sp2_running', question: 'Running' },
            { field: 'sp3_jumping', question: 'Jumping' },
            { field: 'sp4_twisting', question: 'Twisting/pivoting on your injured knee' },
            { field: 'sp5_kneeling', question: 'Kneeling' }
          ])}

          {renderQuestionGroup("Quality of Life", [
            { field: 'q1_awareness', question: 'How often are you aware of your knee problem?' },
            { field: 'q2_lifestyle', question: 'Have you modified your life style to avoid potentially damaging activities to your knee?' },
            { field: 'q3_confidence', question: 'How much are you troubled with lack of confidence in your knee?' },
            { field: 'q4_difficulty', question: 'In general, how much difficulty do you have with your knee?' }
          ])}

          <div className="flex items-center justify-end pt-6">
            <button
              type="button"
              onClick={() => navigate(`/patients/${patient.id}`)}
              className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-4"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
            >
              {loading ? 'Submitting...' : 'Submit KOOS Questionnaire'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// ASESForm Component
const ASESForm = ({ patient }) => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  const [formData, setFormData] = useState({
    patient_id: patient.id,
    date: new Date().toISOString().split('T')[0],
    pain_vas: 0,
    f1_coat: 0,
    f2_sleep: 0,
    f3_wash_back: 0,
    f4_toileting: 0,
    f5_comb_hair: 0,
    f6_high_shelf: 0,
    f7_lift_10lbs: 0,
    f8_throw_ball: 0,
    f9_usual_work: 0,
    f10_usual_sport: 0,
    has_instability: false,
    instability_severity: null,
    usual_work_description: '',
    usual_sport_description: ''
  });

  const functionLabels = ['Unable to do', 'Very difficult to do', 'Somewhat difficult to do', 'Not difficult to do'];

  const calculateTotalScore = () => {
    // Pain component: (10 - pain_vas) * 5 (0-50 points)
    const painComponent = (10 - formData.pain_vas) * 5;
    
    // Function component: sum of function items * 5/3 (0-50 points)
    const functionItems = [
      formData.f1_coat, formData.f2_sleep, formData.f3_wash_back, formData.f4_toileting,
      formData.f5_comb_hair, formData.f6_high_shelf, formData.f7_lift_10lbs,
      formData.f8_throw_ball, formData.f9_usual_work, formData.f10_usual_sport
    ];
    const functionSum = functionItems.reduce((sum, val) => sum + val, 0);
    const functionComponent = functionSum * (5/3);
    
    return {
      painComponent: Math.round(painComponent * 10) / 10,
      functionComponent: Math.round(functionComponent * 10) / 10,
      totalScore: Math.round((painComponent + functionComponent) * 10) / 10
    };
  };

  const scores = calculateTotalScore();

  const handleChange = (field, value) => {
    if (field === 'pain_vas') {
      setFormData(prev => ({ ...prev, [field]: parseFloat(value) }));
    } else if (field === 'has_instability') {
      setFormData(prev => ({ 
        ...prev, 
        [field]: value === 'true',
        instability_severity: value === 'false' ? null : prev.instability_severity
      }));
    } else if (field === 'instability_severity') {
      setFormData(prev => ({ ...prev, [field]: parseFloat(value) }));
    } else if (field.startsWith('f')) {
      setFormData(prev => ({ ...prev, [field]: parseInt(value) }));
    } else {
      setFormData(prev => ({ ...prev, [field]: value }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      await axios.post(`${API}/ases`, formData);
      setSuccess(true);
      setTimeout(() => {
        navigate(`/patients/${patient.id}`);
      }, 2000);
    } catch (err) {
      console.error("Failed to submit ASES", err);
      setError(err.response?.data?.detail || "Failed to submit ASES questionnaire");
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="flex-1 container mx-auto px-6 py-8">
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          ASES questionnaire submitted successfully! Redirecting to patient dashboard...
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 container mx-auto px-6 py-8">
      <div className="mb-6 flex items-center">
        <button 
          onClick={() => navigate(`/patients/${patient.id}`)}
          className="mr-4 text-gray-600 hover:text-gray-900"
        >
          ← Back to {patient.name}
        </button>
        <h1 className="text-3xl font-bold text-gray-800">ASES Questionnaire</h1>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="mb-6">
          <p className="text-gray-600 mb-4">
            <strong>Patient:</strong> {patient.name} - {patient.diagnosis_type || patient.injury_type}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center bg-green-50 p-4 rounded">
            <div>
              <h4 className="text-sm font-medium text-gray-700">Pain Component</h4>
              <p className="text-2xl font-bold text-green-600">{scores.painComponent}/50</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-700">Function Component</h4>
              <p className="text-2xl font-bold text-green-600">{scores.functionComponent}/50</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-700">Total ASES Score</h4>
              <p className="text-3xl font-bold text-green-600">{scores.totalScore}/100</p>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <label htmlFor="date" className="block text-gray-700 text-sm font-bold mb-2">
              Assessment Date
            </label>
            <input
              type="date"
              id="date"
              value={formData.date}
              onChange={(e) => setFormData(prev => ({ ...prev, date: e.target.value }))}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            />
          </div>

          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-700 mb-4">Pain Assessment</h3>
            <div className="bg-gray-50 p-4 rounded">
              <p className="text-sm font-medium text-gray-700 mb-3">How would you rate your pain today on a scale of 0-10?</p>
              <div className="flex items-center space-x-4">
                <span className="text-sm text-gray-600">0 (No pain)</span>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.5"
                  value={formData.pain_vas}
                  onChange={(e) => handleChange('pain_vas', e.target.value)}
                  className="flex-1"
                />
                <span className="text-sm text-gray-600">10 (Worst pain)</span>
              </div>
              <div className="text-center mt-2">
                <span className="text-2xl font-bold text-red-600">{formData.pain_vas}</span>
              </div>
            </div>
          </div>

          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-700 mb-4">Functional Assessment</h3>
            <p className="text-sm text-gray-600 mb-4">Rate your ability to perform the following activities:</p>
            
            <div className="space-y-4">
              {[
                { field: 'f1_coat', question: 'Put on a coat' },
                { field: 'f2_sleep', question: 'Sleep on your painful or affected side' },
                { field: 'f3_wash_back', question: 'Wash your back or do up bra in back' },
                { field: 'f4_toileting', question: 'Manage toileting' },
                { field: 'f5_comb_hair', question: 'Comb your hair' },
                { field: 'f6_high_shelf', question: 'Reach a high shelf' },
                { field: 'f7_lift_10lbs', question: 'Lift 10 pounds above shoulder level' },
                { field: 'f8_throw_ball', question: 'Throw a ball overhand' },
                { field: 'f9_usual_work', question: 'Do usual work' },
                { field: 'f10_usual_sport', question: 'Do usual sport' }
              ].map(({ field, question }) => (
                <div key={field} className="bg-gray-50 p-4 rounded">
                  <p className="text-sm font-medium text-gray-700 mb-3">{question}</p>
                  <div className="flex flex-wrap gap-4">
                    {functionLabels.map((label, index) => (
                      <label key={index} className="flex items-center">
                        <input
                          type="radio"
                          name={field}
                          value={index}
                          checked={formData[field] === index}
                          onChange={(e) => handleChange(field, e.target.value)}
                          className="mr-2"
                        />
                        <span className="text-sm">{index} - {label}</span>
                      </label>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-700 mb-4">Additional Information</h3>
            
            <div className="mb-4">
              <p className="text-sm font-medium text-gray-700 mb-3">Do you have instability (feeling of shoulder coming out of joint)?</p>
              <div className="flex space-x-4">
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="has_instability"
                    value="true"
                    checked={formData.has_instability === true}
                    onChange={(e) => handleChange('has_instability', e.target.value)}
                    className="mr-2"
                  />
                  Yes
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="has_instability"
                    value="false"
                    checked={formData.has_instability === false}
                    onChange={(e) => handleChange('has_instability', e.target.value)}
                    className="mr-2"
                  />
                  No
                </label>
              </div>
            </div>

            {formData.has_instability && (
              <div className="mb-4">
                <p className="text-sm font-medium text-gray-700 mb-3">If yes, rate your instability (0-10):</p>
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-600">0 (No instability)</span>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    step="0.5"
                    value={formData.instability_severity || 0}
                    onChange={(e) => handleChange('instability_severity', e.target.value)}
                    className="flex-1"
                  />
                  <span className="text-sm text-gray-600">10 (Severe instability)</span>
                </div>
                <div className="text-center mt-2">
                  <span className="text-xl font-bold text-blue-600">{formData.instability_severity || 0}</span>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-gray-700 text-sm font-bold mb-2">
                  Describe your usual work (optional)
                </label>
                <textarea
                  value={formData.usual_work_description}
                  onChange={(e) => handleChange('usual_work_description', e.target.value)}
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  rows="3"
                  placeholder="e.g., office work, manual labor, etc."
                />
              </div>
              <div>
                <label className="block text-gray-700 text-sm font-bold mb-2">
                  Describe your usual sport (optional)
                </label>
                <textarea
                  value={formData.usual_sport_description}
                  onChange={(e) => handleChange('usual_sport_description', e.target.value)}
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  rows="3"
                  placeholder="e.g., tennis, swimming, basketball, etc."
                />
              </div>
            </div>
          </div>

          <div className="flex items-center justify-end pt-6">
            <button
              type="button"
              onClick={() => navigate(`/patients/${patient.id}`)}
              className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-4"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
            >
              {loading ? 'Submitting...' : 'Submit ASES Questionnaire'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// ProAssessmentRouter Component
const ProAssessmentRouter = () => {
  const { patientId } = useParams();
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPatient = async () => {
      try {
        const response = await axios.get(`${API}/patients/${patientId}`);
        setPatient(response.data);
        setLoading(false);
      } catch (err) {
        console.error("Failed to fetch patient", err);
        setError("Failed to load patient");
        setLoading(false);
      }
    };

    if (patientId) {
      fetchPatient();
    }
  }, [patientId]);

  if (loading) {
    return (
      <div className="flex-1 container mx-auto px-6 py-8">
        <div className="flex justify-center">
          <div className="loader"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 container mx-auto px-6 py-8">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      </div>
    );
  }

  if (!patient) {
    return (
      <div className="flex-1 container mx-auto px-6 py-8">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          Patient not found
        </div>
      </div>
    );
  }

  // Determine body part from diagnosis
  const isKneePatient = patient.diagnosis_type && 
    ['ACL Tear', 'Meniscus Tear', 'Cartilage Defect', 'Knee Osteoarthritis', 'Post Total Knee Replacement'].includes(patient.diagnosis_type) ||
    patient.injury_type === 'ACL';

  if (isKneePatient) {
    return <KOOSForm patient={patient} />;
  } else {
    return <ASESForm patient={patient} />;
  }
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
                      {patient.name} - {patient.diagnosis_type || patient.injury_type}
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
          <Route path="/patients/new" element={<PatientForm />} />
          <Route path="/patients/:patientId" element={<PatientDetail />} />
          <Route path="/surveys/new" element={<SurveyForm />} />
          <Route path="/wearable/new" element={<WearableDataForm />} />
          <Route path="/pro-assessment/:patientId" element={<ProAssessmentRouter />} />
        </Routes>
        <Footer />
      </BrowserRouter>
    </div>
  );
}

export default App;
