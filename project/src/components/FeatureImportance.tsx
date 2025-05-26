import React from 'react';

interface FeatureImportanceProps {
  data: Array<{
    feature: string;
    importance: number;
  }>;
}

const FeatureImportance: React.FC<FeatureImportanceProps> = ({ data }) => {
  // Sort data by importance (descending)
  const sortedData = [...data].sort((a, b) => b.importance - a.importance);
  
  // Take top 5 features
  const topFeatures = sortedData.slice(0, 5);
  
  // Get the maximum importance value for scaling
  const maxImportance = Math.max(...topFeatures.map(item => item.importance));
  
  // Helper function to format feature names for display
  const formatFeatureName = (name: string) => {
    return name
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase());
  };

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Key Risk Factors</h3>
      <div className="space-y-3">
        {topFeatures.map((feature, index) => (
          <div key={index} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-gray-600">{formatFeatureName(feature.feature)}</span>
              <span className="text-gray-500">
                {(feature.importance * 100 / maxImportance).toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div 
                className="h-1.5 rounded-full bg-blue-600"
                style={{ width: `${(feature.importance * 100 / maxImportance).toFixed(0)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500 mt-3">
        These factors have the most significant impact on hypertension risk prediction.
      </p>
    </div>
  );
};

export default FeatureImportance;