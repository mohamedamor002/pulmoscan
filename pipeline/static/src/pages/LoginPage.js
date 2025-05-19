import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import MedicalBackground from '../components/MedicalBackground';

const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const { darkMode } = useTheme();

  // Check for pending username when component mounts
  useEffect(() => {
    const pendingUsername = sessionStorage.getItem('pendingUsername');
    if (pendingUsername) {
      // Set the username field
      setUsername(pendingUsername);
      // Add a message about the username change
      setError('Your username has been changed. Please log in with your new username and password.');
      // Remove from session storage
      sessionStorage.removeItem('pendingUsername');
    }
    
    // Check if we have a saved username from remember me
    const savedUsername = localStorage.getItem('rememberedUsername');
    if (savedUsername) {
      setUsername(savedUsername);
      setRememberMe(true);
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const result = await login(username, password);
      if (result.success) {
        // Save username if remember me is checked
        if (rememberMe) {
          localStorage.setItem('rememberedUsername', username);
        } else {
          localStorage.removeItem('rememberedUsername');
        }
        navigate('/');
      } else {
        setError(result.message);
      }
    } catch (error) {
      setError('An unexpected error occurred. Please try again.');
      console.error('Login error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex items-center justify-center ${darkMode ? 'bg-transparent' : 'bg-transparent'} py-12 px-4 sm:px-6 lg:px-8 relative`}>
      {/* Use the default MedicalBackground component that cycles through all animations */}
      <MedicalBackground />

      <div className={`max-w-md w-full space-y-8 p-10 rounded-xl shadow-lg ${darkMode ? 'bg-gray-800 bg-opacity-90' : 'bg-white bg-opacity-90'} relative z-10`}>
        <div>
          <div className="flex justify-center">
            <img src="/images/logo/Logo (1).png" alt="PulmoScan Logo" className="h-48 w-auto" />
          </div>
          <h2 className={`mt-3 text-center text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Lung Nodule Detection System
          </h2>
          <p className={`mt-2 text-center text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Sign in to access advanced pulmonary diagnostics
          </p>
        </div>
        
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md -space-y-px">
            <div className="flex items-center relative">
              <span className={`absolute left-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                </svg>
              </span>
              <input
                id="username"
                name="username"
                type="text"
                required
                className={`appearance-none relative block w-full px-10 py-3 border ${darkMode ? 'border-gray-700 bg-gray-800 placeholder-gray-500 text-white' : 'border-gray-300 placeholder-gray-400 text-gray-900'} rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm transition-all duration-300`}
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div className="flex items-center relative">
              <span className={`absolute left-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                </svg>
              </span>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                className={`appearance-none relative block w-full px-10 py-3 border ${darkMode ? 'border-gray-700 bg-gray-800 placeholder-gray-500 text-white' : 'border-gray-300 placeholder-gray-400 text-gray-900'} rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm transition-all duration-300`}
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoFocus={username !== ''}
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                id="remember-me"
                name="remember-me"
                type="checkbox"
                className={`h-4 w-4 ${darkMode ? 'bg-gray-700 border-gray-600 text-indigo-500' : 'text-indigo-600 border-gray-300'} rounded focus:ring-indigo-500`}
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              <label htmlFor="remember-me" className={`ml-2 block text-sm ${darkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                Remember me
              </label>
            </div>

            <div className="text-sm">
              <a href="#forgot-password" className={`font-medium ${darkMode ? 'text-indigo-400 hover:text-indigo-300' : 'text-indigo-600 hover:text-indigo-500'}`}>
                Forgot your password?
              </a>
            </div>
          </div>

          {error && (
            <div className={`rounded-md ${darkMode ? 'bg-red-900/50' : 'bg-red-50'} p-4 border ${darkMode ? 'border-red-800' : 'border-red-200'}`}>
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 ${darkMode ? 'text-red-400' : 'text-red-400'}`} viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className={`text-sm font-medium ${darkMode ? 'text-red-300' : 'text-red-800'}`}>{error}</h3>
                </div>
              </div>
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={isLoading}
              className={`group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-300 transform hover:scale-102 ${
                isLoading ? 'opacity-70 cursor-not-allowed' : ''
              }`}
            >
              <span className="absolute left-0 inset-y-0 flex items-center pl-3">
                {isLoading ? (
                  <svg className="h-5 w-5 text-indigo-300 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  <svg className="h-5 w-5 text-indigo-300 group-hover:text-indigo-200" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                  </svg>
                )}
              </span>
              {isLoading ? 'Signing in...' : 'Sign in'}
            </button>
          </div>
          
          <div className={`text-sm text-center mt-6 px-4 py-3 rounded ${darkMode ? 'bg-gray-700/50 text-gray-300' : 'bg-gray-50 text-gray-600'}`}>
            <p className="font-medium">Demo credentials</p>
            <p className="mt-1">Admin: <span className="font-mono bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded">admin / admin123</span></p>
            <p className="mt-1">Doctor: <span className="font-mono bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded">doctor / doctor123</span></p>
          </div>
        </form>
      </div>
    </div>
  );
};

export default LoginPage; 