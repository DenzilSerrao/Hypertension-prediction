import React from 'react';
import { Activity } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-800 text-gray-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <Activity className="h-6 w-6 text-blue-400 mr-2" />
            <span className="text-lg font-semibold text-white">HyperDetect</span>
          </div>
          
          <div className="space-y-1 text-sm text-center md:text-right">
            <p>© {new Date().getFullYear()} HyperDetect. All rights reserved.</p>
            <p className="text-gray-400">
              This tool is for educational purposes only and is not a substitute for professional medical advice.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;