import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { Bar, Pie } from 'react-chartjs-2';
import { CloudArrowUpIcon, DocumentTextIcon, ExclamationCircleIcon, ChartPieIcon, UserGroupIcon, CalendarIcon, ClockIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

const DashboardPage = () => {
  const [results, setResults] = useState([]);
  const [recentActivity, setRecentActivity] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    totalScans: 0,
    scansWithNodules: 0,
    totalNodules: 0,
    averageNodulesPerScan: 0,
    failedScans: 0
  });
  const { darkMode } = useTheme();
  const { user } = useAuth();

  // Format dates
  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  // Generate time elapsed string
  const timeElapsed = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);

    if (diffDay > 0) return `${diffDay} day${diffDay !== 1 ? 's' : ''} ago`;
    if (diffHour > 0) return `${diffHour} hour${diffHour !== 1 ? 's' : ''} ago`;
    if (diffMin > 0) return `${diffMin} minute${diffMin !== 1 ? 's' : ''} ago`;
    return 'Just now';
  };

  useEffect(() => {
    const fetchResults = async () => {
      try {
        setIsLoading(true);
        const response = await axios.get('/api/results');
        const data = response.data || [];
        setResults(data);
        
        if (data.length === 0) {
          // No results to process
          setStats({
            totalScans: 0,
            scansWithNodules: 0,
            totalNodules: 0,
            averageNodulesPerScan: 0,
            failedScans: 0
          });
          setRecentActivity([]);
          setIsLoading(false);
          return;
        }
        
        // Create recent activity data
        const sortedByDate = [...data].sort((a, b) => {
          return new Date(b.date || 0) - new Date(a.date || 0);
        });
        
        // Take only the 5 most recent items
        const recentItems = sortedByDate.slice(0, 5).map(result => ({
          id: result.id || Math.random().toString(36).substring(7),
          case_name: result.case_name,
          date: result.date,
          type: 'scan_upload',
          user: result.user || 'Unknown',
          timeAgo: timeElapsed(result.date)
        }));
        
        setRecentActivity(recentItems);
        
        // Calculate statistics
        const scansWithNodules = await Promise.all(
          data.map(async (result) => {
            try {
              const detailsResponse = await axios.get(`/api/results/${result.case_name}/details`);
              const details = detailsResponse.data?.details || '';
              // Simple text parsing to determine if nodules were found
              const hasNodules = details && 
                !details.includes('No nodules detected') && 
                !details.includes('No detailed information available');
              
              // Count nodules by looking for "Nodule X:" patterns in the text
              const noduleMatches = details.match(/Nodule \d+:/g) || [];
              const noduleCount = noduleMatches.length;
              
              return {
                ...result,
                hasNodules,
                noduleCount,
                hasDetails: !!details
              };
            } catch (error) {
              console.error(`Error fetching details for ${result.case_name}:`, error);
              // Return the result with default values even if details fail to load
              return { 
                ...result, 
                hasNodules: false, 
                noduleCount: 0,
                hasDetails: false,
                error: true 
              };
            }
          })
        );
        
        // Filter out results that had errors if needed
        const validResults = scansWithNodules.filter(scan => !scan.error);
        const validResultsCount = validResults.length;
        
        // Continue even if some scans had errors
        const withNodules = scansWithNodules.filter(scan => scan.hasNodules);
        const totalNodules = scansWithNodules.reduce((sum, scan) => sum + scan.noduleCount, 0);
        const failedScans = scansWithNodules.filter(scan => scan.error).length;
        
        setStats({
          totalScans: data.length,
          scansWithNodules: withNodules.length,
          totalNodules: totalNodules,
          averageNodulesPerScan: validResultsCount > 0 ? (totalNodules / validResultsCount).toFixed(2) : 0,
          failedScans: failedScans
        });
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching results:', error);
        
        // Provide more specific error message based on the error
        let errorMessage = 'Failed to load results data. Please try refreshing the page.';
        if (error.response) {
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          console.error('Error response data:', error.response.data);
          errorMessage = `Server error: ${error.response.status}. Please try again later.`;
        } else if (error.request) {
          // The request was made but no response was received
          errorMessage = 'No response from server. Please check your network connection.';
        }
        
        setError(errorMessage);
        
        // Still set some empty results to allow partial dashboard rendering
        setResults([]);
        setStats({
          totalScans: 0,
          scansWithNodules: 0,
          totalNodules: 0,
          averageNodulesPerScan: 0,
          failedScans: 0
        });
        
        setIsLoading(false);
      }
    };

    fetchResults();
  }, []);

  // Prepare data for the bar chart
  const barChartData = {
    labels: ['With Nodules', 'Without Nodules'],
    datasets: [
      {
        label: 'CT Scans',
        data: [stats.scansWithNodules, stats.totalScans - stats.scansWithNodules],
        backgroundColor: ['rgba(79, 70, 229, 0.6)', 'rgba(209, 213, 219, 0.6)'],
        borderColor: ['rgba(79, 70, 229, 1)', 'rgba(209, 213, 219, 1)'],
        borderWidth: 1,
      },
    ],
  };

  // Prepare data for the pie chart
  const pieChartData = {
    labels: ['With Nodules', 'Without Nodules'],
    datasets: [
      {
        data: [stats.scansWithNodules, stats.totalScans - stats.scansWithNodules],
        backgroundColor: ['rgba(79, 70, 229, 0.6)', 'rgba(209, 213, 219, 0.6)'],
        borderColor: ['rgba(79, 70, 229, 1)', 'rgba(209, 213, 219, 1)'],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: darkMode ? '#fff' : '#000',
          font: {
            size: 12
          }
        }
      },
      title: {
        display: true,
        text: 'Nodule Detection Results',
        color: darkMode ? '#fff' : '#000',
        font: {
          size: 14,
          weight: 'bold'
        }
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          precision: 0,
          color: darkMode ? '#fff' : '#000',
        },
        grid: {
          color: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        }
      },
      x: {
        ticks: {
          color: darkMode ? '#fff' : '#000',
        },
        grid: {
          color: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        }
      }
    },
  };

  const pieChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: darkMode ? '#fff' : '#000',
          font: {
            size: 12
          }
        }
      },
      title: {
        display: true,
        text: 'Scans Distribution',
        color: darkMode ? '#fff' : '#000',
        font: {
          size: 14,
          weight: 'bold'
        }
      },
    },
  };

  // Quick action buttons for the dashboard
  const quickActions = [
    {
      title: 'Upload New Scan',
      icon: <CloudArrowUpIcon className="h-6 w-6" />,
      link: '/upload',
      primary: true,
      description: 'Upload a new CT scan for analysis'
    },
    {
      title: 'View All Results',
      icon: <DocumentTextIcon className="h-6 w-6" />,
      link: '/results',
      primary: false,
      description: 'Browse through all previous scan results'
    },
    {
      title: 'User Settings',
      icon: <UserGroupIcon className="h-6 w-6" />,
      link: '/settings',
      primary: false,
      description: 'Manage user account settings'
    },
  ];

  return (
    <div className={`space-y-6 ${darkMode ? 'text-white' : ''}`}>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className={`text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Welcome to PulmoScan, {user?.username || 'User'}
          </h1>
          <p className={`text-sm mt-1 ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
            Here's your dashboard overview of lung nodule detection activity
          </p>
        </div>
        <div className="flex gap-3">
          {quickActions.map((action, index) => (
            <Link 
              key={index} 
              to={action.link} 
              className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                action.primary
                  ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                  : darkMode
                    ? 'bg-gray-700 text-white hover:bg-gray-600'
                    : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-300'
              }`}
            >
              <span className="mr-2">{action.icon}</span>
              {action.title}
            </Link>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="text-center py-12">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-indigo-600 border-r-transparent"></div>
          <p className={`mt-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading dashboard data...</p>
        </div>
      ) : error ? (
        <div className={`${darkMode ? 'bg-red-900' : 'bg-red-50'} p-4 rounded-md`}>
          <div className="flex">
            <ExclamationCircleIcon className={`h-5 w-5 ${darkMode ? 'text-red-300' : 'text-red-400'}`} aria-hidden="true" />
            <div className="ml-3">
              <h3 className={`text-sm font-medium ${darkMode ? 'text-red-100' : 'text-red-800'}`}>{error}</h3>
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Stats Cards */}
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} overflow-hidden shadow rounded-lg transition-transform duration-300 hover:shadow-lg transform hover:-translate-y-1`}>
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-indigo-500 rounded-md p-3">
                    <DocumentTextIcon className="h-6 w-6 text-white" aria-hidden="true" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} truncate`}>Total Scans</dt>
                      <dd className={`text-3xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{stats.totalScans}</dd>
                    </dl>
                  </div>
                </div>
              </div>
              <div className={`px-4 py-4 sm:px-6 ${darkMode ? 'bg-gray-700/50' : 'bg-gray-50'}`}>
                <div className="text-sm flex justify-between items-center">
                  <Link to="/results" className={`font-medium ${darkMode ? 'text-indigo-400 hover:text-indigo-300' : 'text-indigo-600 hover:text-indigo-700'}`}>
                    View all
                  </Link>
                  <span className={darkMode ? 'text-gray-300' : 'text-gray-500'}>
                    <CalendarIcon className="inline-block h-4 w-4 mr-1" />
                    All time
                  </span>
                </div>
              </div>
            </div>

            <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} overflow-hidden shadow rounded-lg transition-transform duration-300 hover:shadow-lg transform hover:-translate-y-1`}>
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-green-500 rounded-md p-3">
                    <DocumentTextIcon className="h-6 w-6 text-white" aria-hidden="true" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} truncate`}>Scans With Nodules</dt>
                      <dd className={`text-3xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{stats.scansWithNodules}</dd>
                    </dl>
                  </div>
                </div>
              </div>
              <div className={`px-4 py-4 sm:px-6 ${darkMode ? 'bg-gray-700/50' : 'bg-gray-50'}`}>
                <div className="text-sm flex justify-between items-center">
                  <div className={`font-medium ${stats.scansWithNodules > 0 ? (darkMode ? 'text-green-400' : 'text-green-600') : (darkMode ? 'text-gray-400' : 'text-gray-500')}`}>
                    {stats.scansWithNodules > 0 ? `${((stats.scansWithNodules / stats.totalScans) * 100).toFixed(1)}% of total` : 'No nodules detected'}
                  </div>
                  <span className={darkMode ? 'text-gray-300' : 'text-gray-500'}>
                    <ChartPieIcon className="inline-block h-4 w-4 mr-1" />
                    Statistics
                  </span>
                </div>
              </div>
            </div>

            <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} overflow-hidden shadow rounded-lg transition-transform duration-300 hover:shadow-lg transform hover:-translate-y-1`}>
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-yellow-500 rounded-md p-3">
                    <DocumentTextIcon className="h-6 w-6 text-white" aria-hidden="true" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} truncate`}>Total Nodules Detected</dt>
                      <dd className={`text-3xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{stats.totalNodules}</dd>
                    </dl>
                  </div>
                </div>
              </div>
              <div className={`px-4 py-4 sm:px-6 ${darkMode ? 'bg-gray-700/50' : 'bg-gray-50'}`}>
                <div className="text-sm flex justify-between items-center">
                  <div className={`font-medium ${darkMode ? 'text-yellow-400' : 'text-yellow-600'}`}>
                    {stats.averageNodulesPerScan} avg per scan
                  </div>
                  <span className={darkMode ? 'text-gray-300' : 'text-gray-500'}>
                    <CalendarIcon className="inline-block h-4 w-4 mr-1" />
                    All time
                  </span>
                </div>
              </div>
            </div>

            <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} overflow-hidden shadow rounded-lg transition-transform duration-300 hover:shadow-lg transform hover:-translate-y-1`}>
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-indigo-500 rounded-md p-3">
                    <UserGroupIcon className="h-6 w-6 text-white" aria-hidden="true" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} truncate`}>Role</dt>
                      <dd className={`text-3xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{(user?.role || '').charAt(0).toUpperCase() + (user?.role || '').slice(1)}</dd>
                    </dl>
                  </div>
                </div>
              </div>
              <div className={`px-4 py-4 sm:px-6 ${darkMode ? 'bg-gray-700/50' : 'bg-gray-50'}`}>
                <div className="text-sm flex justify-between items-center">
                  <Link to="/settings" className={`font-medium ${darkMode ? 'text-indigo-400 hover:text-indigo-300' : 'text-indigo-600 hover:text-indigo-700'}`}>
                    Account Settings
                  </Link>
                  <span className={darkMode ? 'text-gray-300' : 'text-gray-500'}>
                    {user?.username || 'User'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Charts and Recent Activity */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mt-6">
            <div className={`lg:col-span-2 ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow p-5`}>
              <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Nodule Detection Results</h2>
              {stats.totalScans > 0 ? (
                <div className="h-64">
                  <Bar data={barChartData} options={chartOptions} />
                </div>
              ) : (
                <div className={`flex flex-col items-center justify-center h-64 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  <ChartPieIcon className="h-16 w-16 mb-4 opacity-50" />
                  <p className="text-lg font-medium">No scan data available</p>
                  <p className="text-sm mt-2">Upload your first scan to see results</p>
                  <Link to="/upload" className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-md text-sm font-medium hover:bg-indigo-700 transition-colors">
                    Upload Now
                  </Link>
                </div>
              )}
            </div>

            <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow p-5`}>
              <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Recent Activity</h2>
              
              {recentActivity.length > 0 ? (
                <div className="space-y-4">
                  {recentActivity.map((activity, index) => (
                    <div 
                      key={activity.id || index} 
                      className={`flex items-start p-3 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-700/80' : 'bg-gray-50 hover:bg-gray-100'} transition-colors`}
                    >
                      <div className={`flex-shrink-0 rounded-full p-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-200'}`}>
                        <ClockIcon className={`h-5 w-5 ${darkMode ? 'text-indigo-400' : 'text-indigo-500'}`} />
                      </div>
                      <div className="ml-3 flex-1">
                        <div className="flex justify-between">
                          <p className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                            <Link to={`/results/${activity.case_name}`} className="hover:underline">
                              {activity.case_name}
                            </Link>
                          </p>
                          <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{activity.timeAgo}</span>
                        </div>
                        <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                          Uploaded by {activity.user}
                        </p>
                      </div>
                    </div>
                  ))}
                  <div className="text-center mt-4">
                    <Link to="/results" className={`text-sm font-medium ${darkMode ? 'text-indigo-400 hover:text-indigo-300' : 'text-indigo-600 hover:text-indigo-700'}`}>
                      View all activity →
                    </Link>
                  </div>
                </div>
              ) : (
                <div className={`flex flex-col items-center justify-center h-64 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  <ClockIcon className="h-12 w-12 mb-3 opacity-50" />
                  <p className="text-base font-medium">No recent activity</p>
                  <p className="text-sm mt-1">Your activity will appear here</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Quick Actions Section */}
          <div className={`mt-8 ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow p-5`}>
            <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {quickActions.map((action, index) => (
                <Link
                  key={index}
                  to={action.link}
                  className={`flex flex-col items-center justify-center p-6 rounded-lg transition-all transform hover:scale-102 ${
                    action.primary
                      ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                      : darkMode
                        ? 'bg-gray-700 text-white hover:bg-gray-700/80'
                        : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <div className={`rounded-full p-4 mb-3 ${
                    action.primary
                      ? 'bg-indigo-500'
                      : darkMode
                        ? 'bg-gray-600'
                        : 'bg-white'
                  }`}>
                    {action.icon}
                  </div>
                  <h3 className="text-lg font-medium mb-1">{action.title}</h3>
                  <p className={`text-sm text-center ${
                    action.primary
                      ? 'text-indigo-100'
                      : darkMode
                        ? 'text-gray-300'
                        : 'text-gray-500'
                  }`}>
                    {action.description}
                  </p>
                </Link>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default DashboardPage; 