// Helper functions for PRO score calculations and formatting

/**
 * Calculate the change between current and previous scores
 * @param {number} current - Current score
 * @param {number} previous - Previous score
 * @returns {object} Object with change value and formatted string
 */
export const calculateScoreChange = (current, previous) => {
  if (previous === null || previous === undefined) return { change: 0, formatted: "" };
  
  const change = current - previous;
  const sign = change > 0 ? "↑" : change < 0 ? "↓" : "";
  const formatted = change !== 0 ? `${sign}${Math.abs(change.toFixed(1))}` : "";
  
  return { change, formatted };
};

/**
 * Get color based on score range
 * @param {number} score - Score value (0-100)
 * @returns {string} Color class for styling
 */
export const getScoreColor = (score) => {
  if (score >= 70) return "green";
  if (score >= 40) return "yellow";
  return "red";
};

/**
 * Get background color classes based on score
 * @param {number} score - Score value (0-100)
 * @returns {string} Tailwind background color classes
 */
export const getScoreBackgroundColor = (score) => {
  if (score >= 70) return "bg-green-50 border-green-200";
  if (score >= 40) return "bg-yellow-50 border-yellow-200";
  return "bg-red-50 border-red-200";
};

/**
 * Get text color classes based on score
 * @param {number} score - Score value (0-100)
 * @returns {string} Tailwind text color classes
 */
export const getScoreTextColor = (score) => {
  if (score >= 70) return "text-green-700";
  if (score >= 40) return "text-yellow-700";
  return "text-red-700";
};

/**
 * Get change color classes based on positive/negative change
 * @param {number} change - Change value
 * @returns {string} Tailwind text color classes
 */
export const getChangeColor = (change) => {
  if (change > 0) return "text-green-600";
  if (change < 0) return "text-red-600";
  return "text-gray-600";
};

/**
 * Format date as relative time (e.g., "3 days ago")
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted relative time string
 */
export const formatDateRelative = (date) => {
  const now = new Date();
  const targetDate = new Date(date);
  const diffTime = Math.abs(now - targetDate);
  const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
  
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
  return `${Math.floor(diffDays / 365)} years ago`;
};

/**
 * Format date for chart display (MM/DD format)
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted date string
 */
export const formatChartDate = (date) => {
  const targetDate = new Date(date);
  return targetDate.toLocaleDateString('en-US', {
    month: '2-digit',
    day: '2-digit'
  });
};

/**
 * Get chart colors for KOOS subscales
 * @returns {object} Object with color mappings
 */
export const getKOOSColors = () => ({
  symptoms_score: "#3B82F6",    // Blue
  pain_score: "#EF4444",        // Red
  adl_score: "#10B981",         // Green
  sport_score: "#F97316",       // Orange
  qol_score: "#8B5CF6",         // Purple
  total_score: "#6B7280"        // Gray
});

/**
 * Get chart colors for ASES components
 * @returns {object} Object with color mappings
 */
export const getASESColors = () => ({
  total_score: "#10B981",       // Green
  pain_component: "#EF4444",    // Red
  function_component: "#3B82F6"  // Blue
});

/**
 * Get display names for KOOS subscales
 * @returns {object} Object with display name mappings
 */
export const getKOOSDisplayNames = () => ({
  symptoms_score: "Symptoms",
  pain_score: "Pain",
  adl_score: "Daily Living",
  sport_score: "Sport",
  qol_score: "Quality of Life",
  total_score: "Total"
});

/**
 * Get display names for ASES components
 * @returns {object} Object with display name mappings
 */
export const getASESDisplayNames = () => ({
  total_score: "Total ASES",
  pain_component: "Pain Component",
  function_component: "Function Component"
});

/**
 * Check if patient needs PRO assessment (>14 days since last)
 * @param {string|Date} lastAssessmentDate - Date of last assessment
 * @returns {boolean} True if assessment is needed
 */
export const needsProAssessment = (lastAssessmentDate) => {
  if (!lastAssessmentDate) return true;
  
  const now = new Date();
  const lastDate = new Date(lastAssessmentDate);
  const diffTime = Math.abs(now - lastDate);
  const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
  
  return diffDays > 14;
};

/**
 * Format score for display with proper decimal places
 * @param {number} score - Score to format
 * @returns {string} Formatted score string
 */
export const formatScore = (score) => {
  if (score === null || score === undefined) return "--";
  return Math.round(score).toString();
};

/**
 * Get skeleton loader component for loading states
 * @param {number} width - Width percentage
 * @param {number} height - Height in pixels
 * @returns {JSX.Element} Skeleton loader component
 */
export const SkeletonLoader = ({ width = 100, height = 20 }) => (
  <div 
    className="animate-pulse bg-gray-200 rounded"
    style={{ width: `${width}%`, height: `${height}px` }}
  />
);