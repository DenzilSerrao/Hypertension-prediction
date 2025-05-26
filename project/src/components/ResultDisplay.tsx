import React from 'react';
import { AlertCircle, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

interface ResultDisplayProps {
  prediction: {
    prediction: number;
    probability: number;
    risk_level: string;
    input_data: any;
  };
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ prediction }) => {
  const { prediction: pred, probability, risk_level } = prediction;
  console.log('Prediction data:', prediction);
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'Moderate':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'High':
        return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'Very High':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };
  
  const getIcon = (risk: string) => {
    switch (risk) {
      case 'Low':
        return <CheckCircle className="h-12 w-12 text-green-500" />;
      case 'Moderate':
        return <AlertTriangle className="h-12 w-12 text-yellow-500" />;
      case 'High':
        return <AlertTriangle className="h-12 w-12 text-orange-500" />;
      case 'Very High':
        return <XCircle className="h-12 w-12 text-red-500" />;
      default:
        return <AlertCircle className="h-12 w-12 text-gray-500" />;
    }
  };
  
  const getRecommendation = (risk: string) => {
    switch (risk) {
      case 'Low':
        return 'Continue maintaining a healthy lifestyle with regular exercise and a balanced diet.';
      case 'Moderate':
        return 'Consider lifestyle modifications including reducing sodium intake, increasing physical activity, and maintaining a healthy weight.';
      case 'High':
        return 'Consult with a healthcare provider soon. Immediate lifestyle changes are recommended.';
      case 'Very High':
        return 'Seek medical attention as soon as possible for proper evaluation and potential treatment.';
      default:
        return 'Please consult with a healthcare provider for personalized advice.';
    }
  };
  
  const riskColor = getRiskColor(risk_level);
  const icon = getIcon(risk_level);
  const recommendation = getRecommendation(risk_level);
  
  // Calculate percentage for the progress bar
  const percentProbability = Math.round(probability * 100);
  
  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="flex items-center space-x-4">
        {icon}
        <div>
          <h3 className="text-lg font-semibold">
            {pred === 1 ? 'Hypertension Risk Detected' : 'Low Hypertension Risk'}
          </h3>
          <p className={`text-sm font-medium ${riskColor.split(' ')[0]}`}>
            {risk_level} Risk Level
          </p>
        </div>
      </div>
      
      <div className={`p-4 border rounded-md ${riskColor}`}>
        <div className="mb-2 flex justify-between">
          <span className="text-sm font-medium">Risk Probability</span>
          <span className="text-sm font-semibold">{percentProbability}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className={`h-2.5 rounded-full ${
              risk_level === 'Low' ? 'bg-green-500' :
              risk_level === 'Moderate' ? 'bg-yellow-500' :
              risk_level === 'High' ? 'bg-orange-500' : 'bg-red-500'
            }`} 
            style={{ width: `${percentProbability}%` }}
          ></div>
        </div>
      </div>
      
      <div className="p-4 bg-blue-50 border border-blue-200 rounded-md">
        <h4 className="text-sm font-semibold text-blue-800 mb-2">Recommendation</h4>
        <p className="text-sm text-blue-700">{recommendation}</p>
      </div>
      
      <div className="text-xs text-gray-500 italic">
        <p>Note: This prediction is based on machine learning and should not replace professional medical advice.</p>
      </div>
    </div>
  );
};

export default ResultDisplay;