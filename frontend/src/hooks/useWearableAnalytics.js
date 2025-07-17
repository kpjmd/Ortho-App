import { useState, useEffect } from 'react';
import axios from 'axios';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export const useWearableAnalytics = (patientId) => {
  const [data, setData] = useState({
    velocity: null,
    clinicalRisk: null,
    correlations: null,
    predictions: null,
    alerts: null,
    plateauRisk: null,
    insights: null,
    providerDashboard: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAnalytics = async () => {
    if (!patientId) return;
    
    try {
      setLoading(true);
      const endpoints = [
        `${API}/analytics/${patientId}/recovery-velocity`,
        `${API}/analytics/${patientId}/clinical-risk`,
        `${API}/analytics/${patientId}/correlations`,
        `${API}/analytics/${patientId}/predictions`,
        `${API}/analytics/${patientId}/clinical-alerts`,
        `${API}/analytics/${patientId}/plateau-risk`,
        `${API}/analytics/${patientId}/personalized-insights`,
        `${API}/analytics/${patientId}/provider-dashboard`,
      ];

      const responses = await Promise.allSettled(
        endpoints.map(endpoint => axios.get(endpoint))
      );

      const newData = {
        velocity: responses[0].status === 'fulfilled' ? responses[0].value.data : null,
        clinicalRisk: responses[1].status === 'fulfilled' ? responses[1].value.data : null,
        correlations: responses[2].status === 'fulfilled' ? responses[2].value.data : null,
        predictions: responses[3].status === 'fulfilled' ? responses[3].value.data : null,
        alerts: responses[4].status === 'fulfilled' ? responses[4].value.data : null,
        plateauRisk: responses[5].status === 'fulfilled' ? responses[5].value.data : null,
        insights: responses[6].status === 'fulfilled' ? responses[6].value.data : null,
        providerDashboard: responses[7].status === 'fulfilled' ? responses[7].value.data : null,
      };

      setData(newData);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch analytics data:', err);
      setError('Failed to load analytics data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, [patientId]);

  return {
    data,
    loading,
    error,
    refetch: fetchAnalytics,
  };
};

export const useRecoveryVelocity = (patientId) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!patientId) return;
      
      try {
        setLoading(true);
        const response = await axios.get(`${API}/analytics/${patientId}/recovery-velocity`);
        setData(response.data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch recovery velocity:', err);
        setError('Failed to load recovery velocity data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [patientId]);

  return { data, loading, error };
};

export const useClinicalAlerts = (patientId) => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAlerts = async () => {
    if (!patientId) return;
    
    try {
      setLoading(true);
      const response = await axios.get(`${API}/analytics/${patientId}/clinical-alerts`);
      setAlerts(response.data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch clinical alerts:', err);
      setError('Failed to load clinical alerts');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, [patientId]);

  return { alerts, loading, error, refetch: fetchAlerts };
};

export const useCorrelationAnalysis = (patientId) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!patientId) return;
      
      try {
        setLoading(true);
        const response = await axios.get(`${API}/analytics/${patientId}/correlations`);
        setData(response.data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch correlation data:', err);
        setError('Failed to load correlation data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [patientId]);

  return { data, loading, error };
};

export const usePredictiveInsights = (patientId) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!patientId) return;
      
      try {
        setLoading(true);
        const response = await axios.get(`${API}/analytics/${patientId}/predictions`);
        setData(response.data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch predictions:', err);
        setError('Failed to load prediction data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [patientId]);

  return { data, loading, error };
};

export const useWearableData = (patientId, startDate, endDate) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!patientId) return;
      
      try {
        setLoading(true);
        let url = `${API}/wearable-data/${patientId}`;
        const params = new URLSearchParams();
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        if (params.toString()) url += `?${params.toString()}`;

        const response = await axios.get(url);
        setData(response.data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch wearable data:', err);
        setError('Failed to load wearable data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [patientId, startDate, endDate]);

  return { data, loading, error };
};