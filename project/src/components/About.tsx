import React from 'react';
import { Info, Heart, AlertCircle } from 'lucide-react';

const About: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          <Info className="h-5 w-5 text-blue-500 mr-2" />
          About Hypertension
        </h2>
        
        <div className="space-y-4 text-sm text-gray-600">
          <p>
            Hypertension, or high blood pressure, is a common condition that affects millions of people worldwide. 
            It occurs when the force of blood against your artery walls is consistently too high.
          </p>
          
          <div className="flex items-start space-x-2">
            <Heart className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-gray-700">Risk Factors Include:</p>
              <ul className="list-disc pl-5 mt-1 space-y-1">
                <li>Age (risk increases as you get older)</li>
                <li>Family history of hypertension</li>
                <li>Being overweight or obese</li>
                <li>Physical inactivity</li>
                <li>High sodium diet</li>
                <li>Excessive alcohol consumption</li>
                <li>Smoking</li>
                <li>Stress</li>
                <li>Certain chronic conditions</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-blue-50 p-4 rounded-md border border-blue-100">
            <div className="flex items-start">
              <AlertCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5 mr-2" />
              <div>
                <p className="font-medium text-blue-800">Why Early Detection Matters</p>
                <p className="mt-1 text-blue-700">
                  Hypertension is often called the "silent killer" because it typically has no symptoms 
                  until significant damage has occurred. Regular screening and early detection can help 
                  prevent serious health complications.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          This tool uses machine learning to estimate hypertension risk based on various factors. 
          Results should be discussed with a healthcare provider for proper diagnosis and treatment.
        </p>
      </div>
    </div>
  );
};

export default About;