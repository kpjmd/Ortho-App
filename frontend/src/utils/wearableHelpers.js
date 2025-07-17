// Utility functions for wearable data processing and formatting

export const formatDateForDisplay = (date) => {
  if (!date) return 'N/A';
  return new Date(date).toLocaleDateString();
};

export const formatTimeForDisplay = (time) => {
  if (!time) return 'N/A';
  return new Date(`2000-01-01T${time}`).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export const convertMinutesToHours = (minutes) => {
  if (!minutes) return 0;
  return (minutes / 60).toFixed(1);
};

export const convertSecondsToMinutes = (seconds) => {
  if (!seconds) return 0;
  return (seconds / 60).toFixed(1);
};

export const formatSteps = (steps) => {
  if (!steps) return '0';
  return steps.toLocaleString();
};

export const formatDistance = (distance) => {
  if (!distance) return '0.0';
  return distance.toFixed(1);
};

export const formatHeartRate = (hr) => {
  if (!hr) return 'N/A';
  return `${Math.round(hr)} bpm`;
};

export const formatSpeed = (speedMs) => {
  if (!speedMs) return 'N/A';
  const speedKmh = speedMs * 3.6;
  return `${speedKmh.toFixed(1)} km/h`;
};

export const formatSleepDuration = (minutes) => {
  if (!minutes) return 'N/A';
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m`;
};

export const formatPercentage = (value) => {
  if (value === null || value === undefined) return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
};

export const calculateSleepEfficiency = (totalSleep, timeInBed) => {
  if (!totalSleep || !timeInBed) return 0;
  return (totalSleep / timeInBed);
};

export const getActivityLevel = (steps) => {
  if (!steps) return { level: 'Sedentary', color: 'red' };
  if (steps >= 10000) return { level: 'Very Active', color: 'green' };
  if (steps >= 7500) return { level: 'Active', color: 'blue' };
  if (steps >= 5000) return { level: 'Somewhat Active', color: 'yellow' };
  return { level: 'Sedentary', color: 'red' };
};

export const getSleepQuality = (sleepScore) => {
  if (!sleepScore) return { quality: 'Unknown', color: 'gray' };
  if (sleepScore >= 80) return { quality: 'Excellent', color: 'green' };
  if (sleepScore >= 70) return { quality: 'Good', color: 'blue' };
  if (sleepScore >= 60) return { quality: 'Fair', color: 'yellow' };
  return { quality: 'Poor', color: 'red' };
};

export const getHeartRateZone = (hr, maxHR) => {
  if (!hr || !maxHR) return { zone: 'Unknown', color: 'gray' };
  
  const percentage = (hr / maxHR) * 100;
  
  if (percentage >= 90) return { zone: 'Maximum', color: 'red' };
  if (percentage >= 80) return { zone: 'Anaerobic', color: 'orange' };
  if (percentage >= 70) return { zone: 'Aerobic', color: 'yellow' };
  if (percentage >= 60) return { zone: 'Fat Burn', color: 'blue' };
  return { zone: 'Active Recovery', color: 'green' };
};

export const calculateMovingAverage = (data, windowSize = 7) => {
  if (!data || data.length === 0) return [];
  
  const result = [];
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - windowSize + 1);
    const window = data.slice(start, i + 1);
    const sum = window.reduce((acc, val) => acc + (val || 0), 0);
    result.push(sum / window.length);
  }
  return result;
};

export const calculateTrend = (data, field) => {
  if (!data || data.length < 2) return 'stable';
  
  const recent = data.slice(-7);
  const older = data.slice(-14, -7);
  
  if (recent.length === 0 || older.length === 0) return 'stable';
  
  const recentAvg = recent.reduce((sum, item) => sum + (item[field] || 0), 0) / recent.length;
  const olderAvg = older.reduce((sum, item) => sum + (item[field] || 0), 0) / older.length;
  
  const changePercent = ((recentAvg - olderAvg) / olderAvg) * 100;
  
  if (changePercent > 5) return 'increasing';
  if (changePercent < -5) return 'decreasing';
  return 'stable';
};

export const findDataGaps = (data) => {
  if (!data || data.length === 0) return [];
  
  const gaps = [];
  const sortedData = [...data].sort((a, b) => new Date(a.date) - new Date(b.date));
  
  for (let i = 1; i < sortedData.length; i++) {
    const currentDate = new Date(sortedData[i].date);
    const prevDate = new Date(sortedData[i - 1].date);
    const daysDiff = Math.floor((currentDate - prevDate) / (1000 * 60 * 60 * 24));
    
    if (daysDiff > 1) {
      gaps.push({
        startDate: prevDate,
        endDate: currentDate,
        daysMissing: daysDiff - 1
      });
    }
  }
  
  return gaps;
};

export const calculateDataQuality = (data) => {
  if (!data || data.length === 0) return { score: 0, issues: [] };
  
  const issues = [];
  const totalDays = data.length;
  let completeRecords = 0;
  
  const importantFields = ['steps', 'heart_rate', 'total_sleep_minutes'];
  
  data.forEach(record => {
    let fieldCount = 0;
    importantFields.forEach(field => {
      if (record[field] !== null && record[field] !== undefined) {
        fieldCount++;
      }
    });
    
    if (fieldCount === importantFields.length) {
      completeRecords++;
    }
  });
  
  const completenessScore = (completeRecords / totalDays) * 100;
  
  if (completenessScore < 50) {
    issues.push('Low data completeness - many missing values');
  }
  
  const gaps = findDataGaps(data);
  if (gaps.length > 0) {
    issues.push(`${gaps.length} data gaps detected`);
  }
  
  return {
    score: completenessScore,
    issues,
    totalDays,
    completeRecords,
    gaps
  };
};

export const generateSummaryStats = (data, field) => {
  if (!data || data.length === 0) return null;
  
  const values = data.map(item => item[field]).filter(val => val !== null && val !== undefined);
  if (values.length === 0) return null;
  
  const sum = values.reduce((acc, val) => acc + val, 0);
  const avg = sum / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  
  // Calculate median
  const sorted = [...values].sort((a, b) => a - b);
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)];
  
  return {
    avg,
    min,
    max,
    median,
    sum,
    count: values.length
  };
};

export const exportToCSV = (data, filename = 'wearable_data.csv') => {
  if (!data || data.length === 0) return;
  
  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map(row => headers.map(field => row[field] || '').join(','))
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const exportToJSON = (data, filename = 'wearable_data.json') => {
  if (!data || data.length === 0) return;
  
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const validateWearableData = (data) => {
  const errors = [];
  
  if (!data.date) {
    errors.push('Date is required');
  }
  
  if (data.steps && (data.steps < 0 || data.steps > 100000)) {
    errors.push('Steps must be between 0 and 100,000');
  }
  
  if (data.heart_rate && (data.heart_rate < 30 || data.heart_rate > 220)) {
    errors.push('Heart rate must be between 30 and 220 bpm');
  }
  
  if (data.oxygen_saturation && (data.oxygen_saturation < 70 || data.oxygen_saturation > 100)) {
    errors.push('Oxygen saturation must be between 70% and 100%');
  }
  
  if (data.sleep_hours && (data.sleep_hours < 0 || data.sleep_hours > 24)) {
    errors.push('Sleep hours must be between 0 and 24');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};