import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { MoonIcon, SunIcon, PencilIcon, CheckIcon, XMarkIcon, ExclamationCircleIcon, QuestionMarkCircleIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

// Create a context for background settings
export const BackgroundContext = React.createContext({
  backgroundVariant: 'default',
  setBackgroundVariant: () => {},
});

// Add this somewhere at the top level of your app (e.g., in App.js or a new context file)
export const BackgroundProvider = ({ children }) => {
  const [backgroundVariant, setBackgroundVariant] = useState(
    localStorage.getItem('backgroundVariant') || 'default'
  );
  
  // Save to localStorage when changed
  useEffect(() => {
    localStorage.setItem('backgroundVariant', backgroundVariant);
  }, [backgroundVariant]);
  
  return (
    <BackgroundContext.Provider value={{ backgroundVariant, setBackgroundVariant }}>
      {children}
    </BackgroundContext.Provider>
  );
};

const SettingsPage = () => {
  const { user, login, logout, updateUserContext } = useAuth();
  const { darkMode, toggleDarkMode } = useTheme();
  const [notifications, setNotifications] = useState(true);
  const [emailUpdates, setEmailUpdates] = useState(true);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(0);
  const [expandedFaqIndex, setExpandedFaqIndex] = useState(null);
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('general');
  const [backgroundVariant, setBackgroundVariant] = useState(
    localStorage.getItem('backgroundVariant') || 'default'
  );
  
  // Profile state
  const [profile, setProfile] = useState({
    username: user?.username || '',
    email: user?.email || '',
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
    firstName: user?.firstName || '',
    lastName: user?.lastName || ''
  });

  // Setup profile from user context when it changes
  useEffect(() => {
    if (user) {
      setProfile(prevProfile => ({
        ...prevProfile,
        username: user.username || '',
        email: user.email || '',
        firstName: user.firstName || '',
        lastName: user.lastName || ''
      }));
    }
  }, [user]);

  // Password strength checker
  useEffect(() => {
    if (!profile.newPassword) {
      setPasswordStrength(0);
      return;
    }

    // Calculate password strength
    let strength = 0;
    
    // Length check
    if (profile.newPassword.length >= 8) strength += 1;
    
    // Contains uppercase
    if (/[A-Z]/.test(profile.newPassword)) strength += 1;
    
    // Contains lowercase
    if (/[a-z]/.test(profile.newPassword)) strength += 1;
    
    // Contains number
    if (/[0-9]/.test(profile.newPassword)) strength += 1;
    
    // Contains special character
    if (/[^A-Za-z0-9]/.test(profile.newPassword)) strength += 1;
    
    setPasswordStrength(strength);
  }, [profile.newPassword]);

  const resetForm = () => {
    setProfile({
      username: user?.username || '',
      email: user?.email || '',
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
      firstName: user?.firstName || '',
      lastName: user?.lastName || ''
    });
    setIsEditMode(false);
  };

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    setProfile(prevProfile => ({
      ...prevProfile,
      [name]: value
    }));
  };

  const validateForm = () => {
    // Check if passwords match if user is changing password
    if (profile.newPassword) {
      if (!profile.currentPassword) {
        setErrorMessage('Current password is required to set a new password');
        return false;
      }
      if (profile.newPassword !== profile.confirmPassword) {
        setErrorMessage('New passwords do not match');
        return false;
      }
      if (profile.newPassword.length < 6) {
        setErrorMessage('New password must be at least 6 characters');
        return false;
      }
    }

    // Validate username
    if (!profile.username || profile.username.length < 3) {
      setErrorMessage('Username must be at least 3 characters');
      return false;
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (profile.email && !emailRegex.test(profile.email)) {
      setErrorMessage('Please enter a valid email address');
      return false;
    }

    return true;
  };

  // FAQs data
  const faqs = [
    {
      question: "How do I change my password?",
      answer: "To change your password, click the 'Edit Profile' button, enter your current password and choose a new one. For security, passwords should be at least 8 characters long with a mix of letters, numbers, and special characters."
    },
    {
      question: "What happens if I change my username?",
      answer: "If you change your username, you'll be automatically logged out and redirected to the login page. You'll need to log in with your new username but using the same password."
    },
    {
      question: "Can I delete my account?",
      answer: "Account deletion is handled by administrators. Please contact your system administrator if you need to delete your account or remove your data from the system."
    },
    {
      question: "How secure is my medical data?",
      answer: "All medical data is stored securely and encrypted. We follow industry best practices to ensure your data remains private and is only accessible to authorized personnel."
    }
  ];

  const toggleFaq = (index) => {
    if (expandedFaqIndex === index) {
      setExpandedFaqIndex(null);
    } else {
      setExpandedFaqIndex(index);
    }
  };

  const updateUserProfile = async () => {
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    setErrorMessage('');
    
    try {
      // Create update payload
      const updateData = {
        username: profile.username,
        email: profile.email,
        first_name: profile.firstName,
        last_name: profile.lastName
      };

      // Only include password fields if user is changing password
      if (profile.newPassword) {
        updateData.current_password = profile.currentPassword;
        updateData.password = profile.newPassword;
      }

      console.log('Sending profile update with data:', updateData);

      // Call API to update user profile
      const response = await axios.put(`/api/users/${user.username}`, updateData);
      
      if (response.data.success) {
        console.log('Profile update successful. Server response:', response.data);
        
        // Check if username changed or password changed
        const usernameChanged = profile.username !== user.username;
        const passwordChanged = profile.newPassword && profile.newPassword.length > 0;
        
        // Username changes require logout and re-login (authentication token is tied to username)
        if (usernameChanged) {
          console.log('Username changed, will need to re-login');
          setSuccessMessage('Username changed successfully! Please login with your new username.');
          
          // Store new username in session storage so login page can prefill it
          sessionStorage.setItem('pendingUsername', profile.username);
          
          // Delayed logout after message is shown
          setTimeout(() => {
            logout();
            navigate('/login');
          }, 3000);
          
          // Return early to prevent additional logic
          return;
        }
        
        // Handle password change - try silent reauthentication if possible
        if (passwordChanged) {
          console.log('Password changed, attempting silent reauthentication');
          
          try {
            // Silent re-login attempt with new password
            const loginResponse = await login(user.username, profile.newPassword);
            if (loginResponse.success) {
              setSuccessMessage('Profile updated successfully!');
              console.log('Silent re-authentication successful');
            } else {
              // This is unlikely to happen - if the profile update worked but re-login failed
              console.warn('Silent re-authentication failed after successful password change');
              setSuccessMessage('Profile updated successfully!');
            }
          } catch (loginError) {
            console.error('Error during silent re-authentication:', loginError);
            // Even if silent re-auth fails, profile was still updated
            setSuccessMessage('Profile updated successfully!');
          }
        } else {
          // Just a regular profile update (no credentials changed)
          console.log('Regular profile update, no credential changes');
          setSuccessMessage('Profile updated successfully!');
        }
        
        // For all changes except username change, update the user context
        updateUserContext({
          firstName: profile.firstName,
          lastName: profile.lastName,
          email: profile.email
        });
        
        console.log('User context updated with new profile information');
        
        // Exit edit mode
        setIsEditMode(false);
        
        // Clear the success message after 3 seconds for normal updates
        setTimeout(() => {
          setSuccessMessage('');
        }, 3000);
      }
    } catch (error) {
      console.error('Error updating profile:', error);
      
      // Handle error response and show appropriate message
      if (error.response?.status === 401) {
        setErrorMessage('Current password is incorrect. Please try again.');
      } else if (error.response?.status === 403) {
        setErrorMessage('Unauthorized. You do not have permission to update this profile.');
      } else if (error.response?.data?.message) {
        setErrorMessage(error.response.data.message);
      } else {
        setErrorMessage('Failed to update profile. Please try again later.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Helper for password strength indicator
  const getPasswordStrengthLabel = () => {
    if (!profile.newPassword) return '';
    
    const labels = ['Very Weak', 'Weak', 'Moderate', 'Strong', 'Very Strong'];
    return labels[Math.min(passwordStrength, 4)];
  };

  const getPasswordStrengthColor = () => {
    if (!profile.newPassword) return 'bg-gray-200';
    
    const colors = [
      'bg-red-500',           // Very Weak
      'bg-orange-500',        // Weak
      'bg-yellow-500',        // Moderate
      'bg-green-400',         // Strong
      'bg-green-500'          // Very Strong
    ];
    
    return colors[Math.min(passwordStrength, 4)];
  };

  // Function to handle background variant change
  const handleBackgroundChange = (variant) => {
    setBackgroundVariant(variant);
    localStorage.setItem('backgroundVariant', variant);
    // You might want to dispatch an event or use context to update this globally
  };

  return (
    <div className={`max-w-4xl mx-auto ${darkMode ? 'text-white' : ''}`}>
      <h1 className={`text-2xl font-semibold mb-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Settings</h1>
      
      {/* Success message notification */}
      {successMessage && (
        <div className={`${darkMode ? 'bg-green-900/50 border-green-700' : 'bg-green-50 border-green-200'} border rounded-md p-4 mb-6`}>
          <div className="flex">
            <div className="flex-shrink-0">
              <CheckIcon className={`h-5 w-5 ${darkMode ? 'text-green-300' : 'text-green-400'}`} aria-hidden="true" />
            </div>
            <div className="ml-3">
              <p className={`text-sm font-medium ${darkMode ? 'text-green-300' : 'text-green-800'}`}>{successMessage}</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Error message notification */}
      {errorMessage && (
        <div className={`${darkMode ? 'bg-red-900/50 border-red-700' : 'bg-red-50 border-red-200'} border rounded-md p-4 mb-6`}>
          <div className="flex">
            <div className="flex-shrink-0">
              <ExclamationCircleIcon className={`h-5 w-5 ${darkMode ? 'text-red-300' : 'text-red-400'}`} aria-hidden="true" />
            </div>
            <div className="ml-3">
              <p className={`text-sm font-medium ${darkMode ? 'text-red-300' : 'text-red-800'}`}>{errorMessage}</p>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-6">
        {/* Profile Card */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow sm:rounded-lg`}>
          <div className="px-4 py-5 sm:p-6">
            <div className="flex justify-between">
              <h3 className={`text-lg leading-6 font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Profile Information
              </h3>
              <div>
                {!isEditMode ? (
                  <button
                    onClick={() => setIsEditMode(true)}
                    className={`flex items-center px-3 py-1.5 rounded-md ${darkMode ? 'bg-gray-700 hover:bg-gray-600 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}
                  >
                    <PencilIcon className="h-4 w-4 mr-1" />
                    Edit Profile
                  </button>
                ) : (
                  <div className="flex space-x-2">
                    <button
                      onClick={resetForm}
                      className={`flex items-center px-3 py-1.5 rounded-md ${darkMode ? 'bg-gray-700 hover:bg-gray-600 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}
                      disabled={isLoading}
                    >
                      <XMarkIcon className="h-4 w-4 mr-1" />
                      Cancel
                    </button>
                    <button
                      onClick={updateUserProfile}
                      className={`flex items-center px-3 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white`}
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <svg className="animate-spin h-4 w-4 mr-1 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      ) : (
                        <CheckIcon className="h-4 w-4 mr-1" />
                      )}
                      Save Changes
                    </button>
                  </div>
                )}
              </div>
            </div>
            
            <div className="mt-5 space-y-4">
              {/* Form Fields */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label htmlFor="username" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    Username
                  </label>
                  <input
                    type="text"
                    name="username"
                    id="username"
                    value={profile.username}
                    onChange={handleProfileChange}
                    disabled={!isEditMode}
                    className={`mt-1 block w-full py-2 px-3 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300 bg-white text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${!isEditMode ? (darkMode ? 'bg-gray-600' : 'bg-gray-100') : ''}`}
                  />
                  {isEditMode && (
                    <p className={`mt-1 text-xs ${darkMode ? 'text-yellow-300' : 'text-yellow-600'}`}>
                      Changing your username will require you to log in again
                    </p>
                  )}
                </div>
                
                <div>
                  <label htmlFor="email" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    Email
                  </label>
                  <input
                    type="email"
                    name="email"
                    id="email"
                    value={profile.email}
                    onChange={handleProfileChange}
                    disabled={!isEditMode}
                    className={`mt-1 block w-full py-2 px-3 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300 bg-white text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${!isEditMode ? (darkMode ? 'bg-gray-600' : 'bg-gray-100') : ''}`}
                  />
                </div>
                
                <div>
                  <label htmlFor="firstName" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    First Name
                  </label>
                  <input
                    type="text"
                    name="firstName"
                    id="firstName"
                    value={profile.firstName}
                    onChange={handleProfileChange}
                    disabled={!isEditMode}
                    className={`mt-1 block w-full py-2 px-3 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300 bg-white text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${!isEditMode ? (darkMode ? 'bg-gray-600' : 'bg-gray-100') : ''}`}
                  />
                </div>
                
                <div>
                  <label htmlFor="lastName" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    Last Name
                  </label>
                  <input
                    type="text"
                    name="lastName"
                    id="lastName"
                    value={profile.lastName}
                    onChange={handleProfileChange}
                    disabled={!isEditMode}
                    className={`mt-1 block w-full py-2 px-3 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300 bg-white text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm ${!isEditMode ? (darkMode ? 'bg-gray-600' : 'bg-gray-100') : ''}`}
                  />
                </div>
              </div>
              
              {/* Password Fields - Only show when in edit mode */}
              {isEditMode && (
                <div className="mt-6">
                  <h4 className={`text-sm font-medium mb-3 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>Change Password</h4>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                    <div>
                      <label htmlFor="currentPassword" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                        Current Password
                      </label>
                      <input
                        type="password"
                        name="currentPassword"
                        id="currentPassword"
                        value={profile.currentPassword}
                        onChange={handleProfileChange}
                        className={`mt-1 block w-full py-2 px-3 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300 bg-white text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm`}
                      />
                    </div>
                    
                    <div>
                      <label htmlFor="newPassword" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                        New Password
                      </label>
                      <input
                        type="password"
                        name="newPassword"
                        id="newPassword"
                        value={profile.newPassword}
                        onChange={handleProfileChange}
                        className={`mt-1 block w-full py-2 px-3 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300 bg-white text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm`}
                      />
                      {profile.newPassword && (
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div className={`h-2.5 rounded-full ${getPasswordStrengthColor()}`} style={{ width: `${(passwordStrength / 5) * 100}%` }}></div>
                          </div>
                          <p className={`text-xs mt-1 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                            Password strength: <span className={`font-medium ${passwordStrength >= 3 ? (darkMode ? 'text-green-300' : 'text-green-600') : (darkMode ? 'text-yellow-300' : 'text-yellow-600')}`}>{getPasswordStrengthLabel()}</span>
                          </p>
                        </div>
                      )}
                    </div>
                    
                    <div>
                      <label htmlFor="confirmPassword" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                        Confirm New Password
                      </label>
                      <input
                        type="password"
                        name="confirmPassword"
                        id="confirmPassword"
                        value={profile.confirmPassword}
                        onChange={handleProfileChange}
                        className={`mt-1 block w-full py-2 px-3 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'border-gray-300 bg-white text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm`}
                      />
                      {profile.newPassword && profile.confirmPassword && (
                        <p className={`mt-1 text-xs ${profile.newPassword === profile.confirmPassword ? (darkMode ? 'text-green-300' : 'text-green-600') : (darkMode ? 'text-red-300' : 'text-red-600')}`}>
                          {profile.newPassword === profile.confirmPassword ? 'Passwords match ✓' : 'Passwords do not match ✗'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Theme Toggle Card */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow sm:rounded-lg`}>
          <div className="px-4 py-5 sm:p-6">
            <h3 className={`text-lg leading-6 font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Appearance
            </h3>
            <div className="mt-4 flex items-center justify-between">
              <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Dark Mode
              </span>
              <button
                onClick={toggleDarkMode}
                className={`relative inline-flex items-center h-6 rounded-full w-11 transition-colors focus:outline-none ${darkMode ? 'bg-indigo-600' : 'bg-gray-300'}`}
              >
                <span className="sr-only">Toggle dark mode</span>
                <span
                  className={`${
                    darkMode ? 'translate-x-6 bg-white text-indigo-600' : 'translate-x-1 bg-white text-gray-400'
                  } inline-block w-4 h-4 transform rounded-full transition-transform relative`}
                >
                  {darkMode ? (
                    <MoonIcon className="h-3 w-3 absolute top-0.5 left-0.5" />
                  ) : (
                    <SunIcon className="h-3 w-3 absolute top-0.5 left-0.5" />
                  )}
                </span>
              </button>
            </div>
          </div>
        </div>
        
        {/* Notification Settings Card */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow sm:rounded-lg`}>
          <div className="px-4 py-5 sm:p-6">
            <h3 className={`text-lg leading-6 font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Notification Settings
            </h3>
            <div className="mt-4 space-y-4">
              <div className="flex items-center justify-between">
                <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  System Notifications
                </span>
                <button
                  onClick={() => setNotifications(!notifications)}
                  className={`relative inline-flex items-center h-6 rounded-full w-11 transition-colors focus:outline-none ${notifications ? 'bg-indigo-600' : 'bg-gray-300'}`}
                >
                  <span className="sr-only">Toggle notifications</span>
                  <span
                    className={`${
                      notifications ? 'translate-x-6' : 'translate-x-1'
                    } inline-block w-4 h-4 transform bg-white rounded-full transition-transform`}
                  ></span>
                </button>
              </div>
              <div className="flex items-center justify-between">
                <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Email Updates
                </span>
                <button
                  onClick={() => setEmailUpdates(!emailUpdates)}
                  className={`relative inline-flex items-center h-6 rounded-full w-11 transition-colors focus:outline-none ${emailUpdates ? 'bg-indigo-600' : 'bg-gray-300'}`}
                >
                  <span className="sr-only">Toggle email updates</span>
                  <span
                    className={`${
                      emailUpdates ? 'translate-x-6' : 'translate-x-1'
                    } inline-block w-4 h-4 transform bg-white rounded-full transition-transform`}
                  ></span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* FAQ Section */}
      <div className={`mt-8 ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow sm:rounded-lg`}>
        <div className="px-4 py-5 sm:p-6">
          <h3 className={`text-lg leading-6 font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            <div className="flex items-center">
              <QuestionMarkCircleIcon className="h-5 w-5 mr-2" />
              Frequently Asked Questions
            </div>
          </h3>
          
          <div className="mt-6 space-y-4">
            {faqs.map((faq, index) => (
              <div 
                key={index} 
                className={`overflow-hidden ${darkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg transition-all duration-200`}
              >
                <button
                  onClick={() => toggleFaq(index)}
                  className={`w-full px-4 py-3 text-left ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} flex justify-between items-center`}
                >
                  <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {faq.question}
                  </span>
                  {expandedFaqIndex === index ? (
                    <ChevronUpIcon className="h-5 w-5" />
                  ) : (
                    <ChevronDownIcon className="h-5 w-5" />
                  )}
                </button>
                
                <div 
                  className={`transition-all duration-300 ease-in-out overflow-hidden ${
                    expandedFaqIndex === index ? 'max-h-40 opacity-100' : 'max-h-0 opacity-0'
                  }`}
                >
                  <div className={`px-4 pb-3 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    {faq.answer}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage; 