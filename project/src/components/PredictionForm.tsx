import React, { useState, useEffect } from 'react';
import ResultDisplay from './ResultDisplay';
import FeatureImportance from './FeatureImportance';

// Default values for form
const defaultValues = {
  Age: 45,
  Gender: 'Male',
  Weight_kg: 75,
  Height_cm: 170,
  BMI: 25.95,
  Genetic_Disorders: 'None'
};

// Options for genetic disorders
const geneticDisorderOptions = [
  'None',
  'Cystic Fibrosis',
  'Sickle Cell Anemia',
];

const PredictionForm: React.FC = () => {
  const [formData, setFormData] = useState(defaultValues);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [featureImportance, setFeatureImportance] = useState<any[]>([]);
  
  // Fetch model info on component mount
  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/model-info');
        console.log('Fetching model info:', response);
        if (response.ok) {
          const data = await response.json();
          setFeatureImportance(data.feature_importance || []);
        }
      } catch (err) {
        console.error('Error fetching model info:', err);
      }
    };
    
    fetchModelInfo();
  }, []);

  // Calculate BMI automatically when weight or height changes
  useEffect(() => {
    if (formData.Weight_kg > 0 && formData.Height_cm > 0) {
      const bmi = formData.Weight_kg / ((formData.Height_cm / 100) ** 2);
      setFormData(prev => ({ ...prev, BMI: parseFloat(bmi.toFixed(2)) }));
    }
  }, [formData.Weight_kg, formData.Height_cm]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    // Convert numeric inputs to numbers
    const numericFields = ['Age', 'Weight_kg', 'Height_cm'];
    const newValue = numericFields.includes(name) ? Number(value) : value;
    
    setFormData({ ...formData, [name]: newValue });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      console.log('Submitting form data:', formData);
      console.log('Response:', response);
      
      
      if (!response.ok) {
        throw new Error('Server error. Please try again later.');
      }
      
      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      console.error('Error:', err);
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData(defaultValues);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden transition-all duration-300">
      <div className="p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Hypertension Risk Assessment</h2>
        
        {/* Show results if prediction exists */}
        {prediction ? (
          <div className="space-y-6">
            <ResultDisplay prediction={prediction} />
            <div className="flex space-x-4">
              <button
                onClick={resetForm}
                className="px-4 py-2 bg-gray-100 text-gray-800 rounded hover:bg-gray-200 transition-colors"
              >
                New Assessment
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Age Input */}
              <div>
                <label htmlFor="Age" className="block text-sm font-medium text-gray-700 mb-1">
                  Age
                </label>
                <input
                  type="number"
                  id="Age"
                  name="Age"
                  min="18"
                  max="120"
                  value={formData.Age}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>
              
              {/* Gender Input */}
              <div>
                <label htmlFor="Gender" className="block text-sm font-medium text-gray-700 mb-1">
                  Gender
                </label>
                <select
                  id="Gender"
                  name="Gender"
                  value={formData.Gender}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              
              {/* Weight Input */}
              <div>
                <label htmlFor="Weight_kg" className="block text-sm font-medium text-gray-700 mb-1">
                  Weight (kg)
                </label>
                <input
                  type="number"
                  id="Weight_kg"
                  name="Weight_kg"
                  min="30"
                  max="300"
                  step="0.1"
                  value={formData.Weight_kg}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>
              
              {/* Height Input */}
              <div>
                <label htmlFor="Height_cm" className="block text-sm font-medium text-gray-700 mb-1">
                  Height (cm)
                </label>
                <input
                  type="number"
                  id="Height_cm"
                  name="Height_cm"
                  min="100"
                  max="250"
                  step="0.1"
                  value={formData.Height_cm}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>
              
              {/* BMI Display (read-only) */}
              <div>
                <label htmlFor="BMI" className="block text-sm font-medium text-gray-700 mb-1">
                  BMI (calculated)
                </label>
                <input
                  type="number"
                  id="BMI"
                  name="BMI"
                  value={formData.BMI}
                  readOnly
                  className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-md"
                />
              </div>
              
              {/* Genetic Disorders Input */}
              <div>
                <label htmlFor="Genetic_Disorders" className="block text-sm font-medium text-gray-700 mb-1">
                  Genetic Disorders
                </label>
                <select
                  id="Genetic_Disorders"
                  name="Genetic_Disorders"
                  value={formData.Genetic_Disorders}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  {geneticDisorderOptions.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
            </div>
            
            {/* Error Message */}
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md text-red-600">
                {error}
              </div>
            )}
            
            {/* Submit Button */}
            <div className="flex justify-end">
              <button
                type="submit"
                disabled={loading}
                className={`px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors ${
                  loading ? 'opacity-70 cursor-not-allowed' : ''
                }`}
              >
                {loading ? 'Processing...' : 'Predict Risk'}
              </button>
            </div>
          </form>
        )}
      </div>
      
      {/* Feature Importance */}
      {featureImportance.length > 0 && (
        <div className="border-t border-gray-200 p-6 bg-gray-50">
          <FeatureImportance data={featureImportance} />
        </div>
      )}
    </div>
  );
};

export default PredictionForm;