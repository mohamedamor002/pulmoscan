import React from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';

const Modal = ({ isOpen, onClose, title, children }) => {
  const { darkMode } = useTheme();
  
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div className="fixed inset-0 transition-opacity" aria-hidden="true">
          <div className={`absolute inset-0 ${darkMode ? 'bg-gray-900' : 'bg-gray-500'} opacity-75`}></div>
        </div>
        
        {/* Center modal */}
        <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
        
        <div className={`inline-block align-bottom ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full`}>
          {/* Modal header */}
          <div className="flex items-center justify-between px-4 pt-5 pb-2">
            <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {title}
            </h3>
            <button
              type="button"
              className={`rounded-md ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-400 hover:text-gray-500'}`}
              onClick={onClose}
            >
              <span className="sr-only">Close</span>
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>
          
          {/* Modal content */}
          <div className={`px-4 pt-2 pb-4 sm:p-6 sm:pb-4`}>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Modal; 