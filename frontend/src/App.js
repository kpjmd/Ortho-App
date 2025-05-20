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