import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend, ArcElement, RadialLinearScale } from 'chart.js';
import { Bar, Line, Pie, Radar } from 'react-chartjs-2';
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';
import { ChartBarIcon, ArrowPathIcon, ExclamationCircleIcon, CalendarIcon, ClockIcon } from '@heroicons/react/24/outline';

// Register ChartJS components
ChartJS.register(
  CategoryScale, 
  LinearScale, 
  BarElement, 
  PointElement, 
  LineElement,
  Title, 
  Tooltip, 
  Legend, 
  ArcElement,
  RadialLinearScale
);

const AnalyticsPage = () => {
  const [data, setData] = useState({
    scansByMonth: [],
    noduleLocationData: [],
    confidenceDistribution: [],
    scansByDoctor: [],
    noduleSize: []
  });
  const [dateRange, setDateRange] = useState('6months');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const { darkMode } = useTheme();
  const { user } = useAuth();

  // Mock data generation
  useEffect(() => {
    const fetchAnalyticsData = async () => {
      try {
        setIsLoading(true);
        
        // In a real implementation, this would be an API call to fetch analytics data
        // For now, we'll generate mock data that looks realistic
        
        // Fetch real scan results for reference
        const resultsResponse = await axios.get('/api/results');
        const actualResults = resultsResponse.data || [];
        
        // Generate realistic mock data based on the actual results count
        const mockData = generateMockAnalyticsData(actualResults.length, dateRange);
        setData(mockData);
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching analytics data:', error);
        setError('Failed to load analytics data. Please try again later.');
        setIsLoading(false);
      }
    };

    fetchAnalyticsData();
  }, [dateRange]);

  // Generate mock data for visualization
  const generateMockAnalyticsData = (resultCount, range) => {
    // Define date ranges
    const now = new Date();
    let monthsToShow = 6;
    
    switch(range) {
      case '3months':
        monthsToShow = 3;
        break;
      case '12months':
        monthsToShow = 12;
        break;
      default:
        monthsToShow = 6;
    }
    
    // Generate months labels
    const monthLabels = [];
    for (let i = monthsToShow - 1; i >= 0; i--) {
      const d = new Date();
      d.setMonth(now.getMonth() - i);
      monthLabels.push(d.toLocaleString('default', { month: 'short', year: 'numeric' }));
    }
    
    // Scans by month (with some randomization for realistic data)
    const baseScanCount = Math.max(Math.floor(resultCount / monthsToShow), 1);
    const scansByMonth = monthLabels.map((month, i) => {
      // Make more recent months have slightly more scans for a realistic trend
      const factor = 0.5 + ((i + 1) / monthsToShow) * 0.5;
      return {
        month,
        totalScans: Math.floor(baseScanCount * factor * (0.8 + Math.random() * 0.4)),
        withNodules: Math.floor(baseScanCount * factor * 0.6 * (0.7 + Math.random() * 0.5))
      };
    });
    
    // Nodule location data (distribution across lung areas)
    const noduleLocationData = [
      { location: 'Upper Right Lobe', count: Math.floor(10 + Math.random() * 20) },
      { location: 'Middle Right Lobe', count: Math.floor(5 + Math.random() * 15) },
      { location: 'Lower Right Lobe', count: Math.floor(15 + Math.random() * 25) },
      { location: 'Upper Left Lobe', count: Math.floor(10 + Math.random() * 20) },
      { location: 'Lower Left Lobe', count: Math.floor(15 + Math.random() * 25) },
    ];
    
    // AI confidence distribution
    const confidenceRanges = ['90-100%', '80-90%', '70-80%', '60-70%', '<60%'];
    const confidenceDistribution = confidenceRanges.map(range => ({
      range,
      count: Math.floor(5 + Math.random() * 20)
    }));
    
    // Scans by doctor (top 5)
    const doctorNames = [
      'Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Jones', 
      'Dr. Garcia', 'Dr. Miller', 'Dr. Davis', 'Dr. Rodriguez', 'Dr. Martinez'
    ];
    
    // Select 5 random doctors
    const selectedDoctors = doctorNames.sort(() => 0.5 - Math.random()).slice(0, 5);
    
    const scansByDoctor = selectedDoctors.map(doctor => ({
      doctor,
      scans: Math.floor(5 + Math.random() * 20)
    })).sort((a, b) => b.scans - a.scans);
    
    // Nodule size distribution
    const noduleSizeRanges = ['<3mm', '3-6mm', '6-10mm', '10-15mm', '>15mm'];
    const noduleSize = noduleSizeRanges.map(range => ({
      range,
      count: Math.floor(5 + Math.random() * 15)
    }));
    
    return {
      scansByMonth,
      noduleLocationData,
      confidenceDistribution,
      scansByDoctor,
      noduleSize
    };
  };

  // Chart configuration and data preparation
  const monthlyScansChartData = {
    labels: data.scansByMonth.map(item => item.month),
    datasets: [
      {
        label: 'Total Scans',
        data: data.scansByMonth.map(item => item.totalScans),
        backgroundColor: 'rgba(79, 70, 229, 0.6)',
        borderColor: 'rgba(79, 70, 229, 1)',
        borderWidth: 1,
      },
      {
        label: 'Scans With Nodules',
        data: data.scansByMonth.map(item => item.withNodules),
        backgroundColor: 'rgba(236, 72, 153, 0.6)',
        borderColor: 'rgba(236, 72, 153, 1)',
        borderWidth: 1,
      }
    ],
  };

  const monthlyTrendChartData = {
    labels: data.scansByMonth.map(item => item.month),
    datasets: [
      {
        label: 'Total Scans',
        data: data.scansByMonth.map(item => item.totalScans),
        backgroundColor: 'rgba(79, 70, 229, 0.1)',
        borderColor: 'rgba(79, 70, 229, 1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Scans With Nodules',
        data: data.scansByMonth.map(item => item.withNodules),
        backgroundColor: 'rgba(236, 72, 153, 0.1)',
        borderColor: 'rgba(236, 72, 153, 1)',
        fill: true,
        tension: 0.4,
      }
    ],
  };

  const noduleLocationChartData = {
    labels: data.noduleLocationData.map(item => item.location),
    datasets: [
      {
        label: 'Nodule Distribution by Location',
        data: data.noduleLocationData.map(item => item.count),
        backgroundColor: [
          'rgba(79, 70, 229, 0.6)',
          'rgba(16, 185, 129, 0.6)',
          'rgba(245, 158, 11, 0.6)',
          'rgba(236, 72, 153, 0.6)',
          'rgba(99, 102, 241, 0.6)'
        ],
        borderColor: [
          'rgba(79, 70, 229, 1)',
          'rgba(16, 185, 129, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(236, 72, 153, 1)',
          'rgba(99, 102, 241, 1)'
        ],
        borderWidth: 1,
      },
    ],
  };

  const confidenceChartData = {
    labels: data.confidenceDistribution.map(item => item.range),
    datasets: [
      {
        label: 'AI Confidence Distribution',
        data: data.confidenceDistribution.map(item => item.count),
        backgroundColor: 'rgba(16, 185, 129, 0.6)',
        borderColor: 'rgba(16, 185, 129, 1)',
        borderWidth: 1,
      },
    ],
  };

  const doctorScansChartData = {
    labels: data.scansByDoctor.map(item => item.doctor),
    datasets: [
      {
        label: 'Scans by Doctor',
        data: data.scansByDoctor.map(item => item.scans),
        backgroundColor: [
          'rgba(79, 70, 229, 0.6)',
          'rgba(16, 185, 129, 0.6)',
          'rgba(245, 158, 11, 0.6)',
          'rgba(236, 72, 153, 0.6)',
          'rgba(99, 102, 241, 0.6)'
        ],
        borderColor: [
          'rgba(79, 70, 229, 1)',
          'rgba(16, 185, 129, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(236, 72, 153, 1)',
          'rgba(99, 102, 241, 1)'
        ],
        borderWidth: 1,
      },
    ],
  };

  const noduleSizeChartData = {
    labels: data.noduleSize.map(item => item.range),
    datasets: [
      {
        label: 'Nodule Size Distribution',
        data: data.noduleSize.map(item => item.count),
        backgroundColor: 'rgba(245, 158, 11, 0.6)',
        borderColor: [
          'rgba(245, 158, 11, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(245, 158, 11, 1)'
        ],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
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
      tooltip: {
        backgroundColor: darkMode ? 'rgba(0, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.8)',
        titleColor: darkMode ? '#fff' : '#000',
        bodyColor: darkMode ? '#e5e7eb' : '#374151',
        borderColor: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          precision: 0,
          color: darkMode ? '#e5e7eb' : '#374151',
        },
        grid: {
          color: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        }
      },
      x: {
        ticks: {
          color: darkMode ? '#e5e7eb' : '#374151',
        },
        grid: {
          color: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        }
      }
    }
  };

  // Special options for pie chart
  const pieChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          color: darkMode ? '#fff' : '#000',
          font: {
            size: 12
          },
          padding: 20
        }
      },
      tooltip: {
        backgroundColor: darkMode ? 'rgba(0, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.8)',
        titleColor: darkMode ? '#fff' : '#000',
        bodyColor: darkMode ? '#e5e7eb' : '#374151',
        borderColor: darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1
      }
    }
  };

  if (isLoading) {
    return (
      <div className={`text-center py-12 ${darkMode ? 'text-white' : ''}`}>
        <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-indigo-600 border-r-transparent"></div>
        <p className={`mt-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading analytics data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`${darkMode ? 'bg-red-900' : 'bg-red-50'} p-4 rounded-md`}>
        <div className="flex">
          <ExclamationCircleIcon className={`h-5 w-5 ${darkMode ? 'text-red-300' : 'text-red-400'}`} aria-hidden="true" />
          <div className="ml-3">
            <h3 className={`text-sm font-medium ${darkMode ? 'text-red-300' : 'text-red-800'}`}>{error}</h3>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={darkMode ? 'text-white' : ''}>
      <div className="mb-6">
        <h1 className={`text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          <ChartBarIcon className="h-7 w-7 inline-block mr-2 mb-1" />
          Analytics Dashboard
        </h1>
        <p className={`mt-1 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
          Comprehensive data visualization for scan results and nodule detection statistics
        </p>
      </div>

      {/* Date filter */}
      <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm mb-6`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <CalendarIcon className={`h-5 w-5 mr-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
            <span className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              Time Period:
            </span>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setDateRange('3months')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                dateRange === '3months'
                  ? darkMode
                    ? 'bg-indigo-600 text-white'
                    : 'bg-indigo-50 text-indigo-700'
                  : darkMode
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Last 3 Months
            </button>
            <button
              onClick={() => setDateRange('6months')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                dateRange === '6months'
                  ? darkMode
                    ? 'bg-indigo-600 text-white'
                    : 'bg-indigo-50 text-indigo-700'
                  : darkMode
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Last 6 Months
            </button>
            <button
              onClick={() => setDateRange('12months')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                dateRange === '12months'
                  ? darkMode
                    ? 'bg-indigo-600 text-white'
                    : 'bg-indigo-50 text-indigo-700'
                  : darkMode
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Last 12 Months
            </button>
          </div>
          <button
            onClick={() => {
              setIsLoading(true);
              setTimeout(() => {
                // Generate new mock data with the same date range
                const mockData = generateMockAnalyticsData(data.scansByMonth.reduce((sum, item) => sum + item.totalScans, 0), dateRange);
                setData(mockData);
                setIsLoading(false);
              }, 500);
            }}
            className={`flex items-center px-3 py-1.5 rounded-md text-sm font-medium ${
              darkMode 
                ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <ArrowPathIcon className="h-4 w-4 mr-1" />
            Refresh
          </button>
        </div>
      </div>

      {/* Charts grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Monthly scans chart */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm`}>
          <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Monthly Scan Volume
          </h2>
          <div className="h-72">
            <Bar data={monthlyScansChartData} options={chartOptions} />
          </div>
        </div>

        {/* Monthly trend chart */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm`}>
          <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Scan Volume Trends
          </h2>
          <div className="h-72">
            <Line data={monthlyTrendChartData} options={chartOptions} />
          </div>
        </div>

        {/* Nodule location chart */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm`}>
          <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Nodule Location Distribution
          </h2>
          <div className="h-72">
            <Pie data={noduleLocationChartData} options={pieChartOptions} />
          </div>
        </div>

        {/* AI confidence distribution */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm`}>
          <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            AI Confidence Distribution
          </h2>
          <div className="h-72">
            <Bar data={confidenceChartData} options={chartOptions} />
          </div>
        </div>

        {/* Scans by doctor */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm`}>
          <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Top Doctors by Scan Volume
          </h2>
          <div className="h-72">
            <Bar 
              data={doctorScansChartData} 
              options={{
                ...chartOptions,
                indexAxis: 'y'
              }} 
            />
          </div>
        </div>

        {/* Nodule size distribution */}
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm`}>
          <h2 className={`text-lg font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Nodule Size Distribution
          </h2>
          <div className="h-72">
            <Bar data={noduleSizeChartData} options={chartOptions} />
          </div>
        </div>
      </div>

      {/* Advanced analytics note */}
      <div className={`${darkMode ? 'bg-indigo-900/30 border-indigo-800' : 'bg-indigo-50 border-indigo-100'} border rounded-lg p-4 mb-6`}>
        <h3 className={`text-base font-medium mb-2 ${darkMode ? 'text-indigo-300' : 'text-indigo-800'}`}>
          Advanced Analytics
        </h3>
        <p className={`text-sm ${darkMode ? 'text-indigo-200' : 'text-indigo-700'}`}>
          This dashboard contains sample visualization data. In a production environment, these charts would be populated with real analytics data from your scan database. Contact your administrator for access to detailed reporting features.
        </p>
      </div>
    </div>
  );
};

export default AnalyticsPage; 