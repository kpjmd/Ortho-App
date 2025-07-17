import React from 'react';
import { 
  formatScore, 
  calculateScoreChange, 
  getScoreBackgroundColor, 
  getScoreTextColor, 
  getChangeColor, 
  formatDateRelative 
} from '../utils/proScoreHelpers';

const ScoreSummaryCard = ({ 
  title, 
  currentScore, 
  previousScore, 
  lastAssessmentDate, 
  color 
}) => {
  const scoreChange = calculateScoreChange(currentScore, previousScore);
  const backgroundColorClass = getScoreBackgroundColor(currentScore || 0);
  const textColorClass = getScoreTextColor(currentScore || 0);
  const changeColorClass = getChangeColor(scoreChange.change);

  return (
    <div className={`rounded-lg border-2 p-4 transition-all duration-200 hover:shadow-md ${backgroundColorClass}`}>
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-700">{title}</h4>
        {color && (
          <div 
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: color }}
          />
        )}
      </div>
      
      <div className="flex items-baseline space-x-2">
        <span className={`text-4xl font-bold ${textColorClass}`}>
          {formatScore(currentScore)}
        </span>
        <span className="text-sm text-gray-500">/100</span>
      </div>
      
      {scoreChange.formatted && (
        <div className={`text-sm font-medium mt-1 ${changeColorClass}`}>
          {scoreChange.formatted}
        </div>
      )}
      
      {lastAssessmentDate && (
        <div className="text-xs text-gray-500 mt-2">
          {formatDateRelative(lastAssessmentDate)}
        </div>
      )}
    </div>
  );
};

export default ScoreSummaryCard;