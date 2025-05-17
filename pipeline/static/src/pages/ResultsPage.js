import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import axios from 'axios';
import { DocumentTextIcon, ExclamationCircleIcon, TrashIcon, XMarkIcon, ArrowUpIcon, ArrowDownIcon, FunnelIcon, MagnifyingGlassIcon, ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { ROLES } from '../contexts/AuthContext';
import InteractiveViewer from '../components/InteractiveViewer';

const ResultsPage = () => {
  const location = useLocation();
  const [results, setResults] = useState([]);
  const [filteredResults, setFilteredResults] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [scanToDelete, setScanToDelete] = useState(null);
  const [deleteSuccess, setDeleteSuccess] = useState(location.state?.deleteSuccess || '');
  const { user } = useAuth();
  const { darkMode } = useTheme();
  
  // Filter and sort states
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState('timestamp');
  const [sortDirection, setSortDirection] = useState('desc');
  const [showFilters, setShowFilters] = useState(false);
  const [dateFilter, setDateFilter] = useState('all');
  const [noduleFilter, setNoduleFilter] = useState('all');

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(6);
  const [totalPages, setTotalPages] = useState(1);

  // Animation state for newly loaded items
  const [animatedItems, setAnimatedItems] = useState([]);

  // Clear the location state after displaying a success message
  useEffect(() => {
    if (location.state?.deleteSuccess) {
      // Clear the success message after 3 seconds
      const timer = setTimeout(() => {
        setDeleteSuccess('');
        // Clear the state from the location to avoid showing the message on refresh
        window.history.replaceState({}, document.title);
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [location.state]);

  // Animate items as they appear
  useEffect(() => {
    if (filteredResults.length > 0 && animatedItems.length === 0) {
      // Stagger the animation of items for visual effect
      let timeoutId;
      filteredResults.forEach((result, index) => {
        timeoutId = setTimeout(() => {
          setAnimatedItems(prev => [...prev, result.case_name]);
        }, index * 50); // 50ms stagger between items
      });

      return () => clearTimeout(timeoutId);
    }
  }, [filteredResults]);

    const fetchResults = async () => {
      try {
        setIsLoading(true);
      // Use the user-specific endpoint for doctors and the global endpoint for admins/superadmins
      const endpoint = (user.role === ROLES.ADMIN || user.role === ROLES.SUPERADMIN) 
        ? '/api/results' 
        : '/api/results/user';
      const response = await axios.get(endpoint);
      
        // Sort by timestamp, newest first
        const sortedResults = response.data.sort((a, b) => 
          new Date(b.timestamp) - new Date(a.timestamp)
        );
      
      // Debug log to see what data we're getting
      console.log('Results data:', sortedResults);
      
        setResults(sortedResults);
      setFilteredResults(sortedResults);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching results:', error);
        setError('Failed to load results. Please try again later.');
        setIsLoading(false);
      }
    };

  useEffect(() => {
    fetchResults();
  }, [user.role]);

  // Apply filters and sorting when relevant states change
  useEffect(() => {
    // Reset animations when filters change
    setAnimatedItems([]);
    
    let filtered = [...results];
    
    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(scan => 
        (scan.case_name && scan.case_name.toLowerCase().includes(query)) ||
        (scan.details && scan.details.toLowerCase().includes(query)) ||
        (scan.patient_info?.name && scan.patient_info.name.toLowerCase().includes(query)) ||
        (scan.patient_info?.description && scan.patient_info.description.toLowerCase().includes(query))
      );
    }
    
    // Apply date filter
    if (dateFilter !== 'all') {
      const now = new Date();
      const oneDayAgo = new Date(now.setDate(now.getDate() - 1));
      const oneWeekAgo = new Date(now.setDate(now.getDate() - 7));
      const oneMonthAgo = new Date(now.setMonth(now.getMonth() - 1));
      
      filtered = filtered.filter(scan => {
        const scanDate = new Date(scan.timestamp);
        
        switch (dateFilter) {
          case 'today':
            return scanDate >= oneDayAgo;
          case 'week':
            return scanDate >= oneWeekAgo;
          case 'month':
            return scanDate >= oneMonthAgo;
          default:
            return true;
        }
      });
    }
    
    // Apply nodule filter
    if (noduleFilter !== 'all') {
      filtered = filtered.filter(scan => {
        const hasNodules = scan.details && 
          !scan.details.includes('No nodules detected') && 
          !scan.details.includes('No detailed information available');
        
        return noduleFilter === 'with_nodules' ? hasNodules : !hasNodules;
      });
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let aValue, bValue;
      
      switch (sortField) {
        case 'timestamp':
          aValue = new Date(a.timestamp || 0);
          bValue = new Date(b.timestamp || 0);
          break;
        case 'name':
          aValue = a.case_name || '';
          bValue = b.case_name || '';
          break;
        case 'patient':
          aValue = extractPatientName(a).toLowerCase();
          bValue = extractPatientName(b).toLowerCase();
          break;
        default:
          aValue = new Date(a.timestamp || 0);
          bValue = new Date(b.timestamp || 0);
      }
      
      // For string comparisons
      if (typeof aValue === 'string') {
        return sortDirection === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      
      // For date comparisons and other types
      return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
    });
    
    setFilteredResults(filtered);
    setTotalPages(Math.ceil(filtered.length / itemsPerPage));
    setCurrentPage(1); // Reset to first page when filters change
  }, [results, searchQuery, sortField, sortDirection, dateFilter, noduleFilter]);

  // Get current page items
  const getCurrentPageItems = () => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return filteredResults.slice(startIndex, endIndex);
  };

  // Handle page change
  const handlePageChange = (page) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Toggle sort direction or change sort field
  const handleSort = (field) => {
    if (sortField === field) {
      // Toggle direction if same field
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // New field, default to descending for dates, ascending for text
      setSortField(field);
      setSortDirection(field === 'timestamp' ? 'desc' : 'asc');
    }
  };

  // Check if the current user can delete a specific scan
  const canDeleteScan = (scan) => {
    // Extract owner username from case name (if the format is username_filename)
    const scanOwner = scan.case_name.split('_')[0];
    
    // Superadmin can delete any scan
    if (user.role === ROLES.SUPERADMIN) {
      return true;
    }
    
    // Owners can always delete their own scans
    if (scanOwner === user.username) {
      return true;
    }
    
    // Admins can delete scans from doctors they've created
    if (user.role === ROLES.ADMIN) {
      // In a real implementation, this would check if the admin created the doctor
      // For now, we'll assume admins can delete any doctor's scans
      return true;
    }
    
    // By default, users can't delete other users' scans
    return false;
  };

  // Opens the delete confirmation modal
  const openDeleteModal = (scan, e) => {
    e.preventDefault(); // Prevent navigation to scan details
    e.stopPropagation(); // Prevent event bubbling
    setScanToDelete(scan);
    setShowDeleteModal(true);
  };

  // Handles scan deletion
  const handleDeleteScan = async () => {
    try {
      await axios.delete(`/api/results/${scanToDelete.case_name}`);
      setResults(results.filter(r => r.case_name !== scanToDelete.case_name));
      setDeleteSuccess(`Scan ${scanToDelete.case_name} has been deleted successfully.`);
      setScanToDelete(null);
      setShowDeleteModal(false);
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setDeleteSuccess('');
      }, 3000);
    } catch (error) {
      console.error('Error deleting scan:', error);
      setError('Failed to delete scan. Please try again later.');
      setShowDeleteModal(false);
    }
  };

  // Helper function to extract a title from scan details
  const extractTitle = (scan) => {
    // Check if there's patient information available
    if (scan.patient_info && scan.patient_info.description) {
      // Limit the length of the description to avoid too long titles
      const description = scan.patient_info.description;
      return description.length > 100 ? description.substring(0, 100) + '...' : description;
    }
    
    // Fall back to extracting from details if no patient info description
    if (!scan.details) return "No details available";
    
    // Look for patient information in the details
    const lines = scan.details.split('\n');
    
    // Look for Clinical Notes section
    let foundClinicNotes = false;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes('Clinical Notes:')) {
        foundClinicNotes = true;
        // Return the clinical notes (on the next line or after the colon)
        if (lines[i].split('Clinical Notes:').length > 1) {
          const notes = lines[i].split('Clinical Notes:')[1].trim();
          // Limit length of long notes
          return notes.length > 100 ? notes.substring(0, 100) + '...' : notes || "No clinical notes available";
        } else if (i + 1 < lines.length) {
          const notes = lines[i + 1].trim();
          return notes.length > 100 ? notes.substring(0, 100) + '...' : notes || "No clinical notes available";
        }
      }
    }
    
    // If no clinical notes found, extract the first meaningful line
    const titleLine = lines.find(line => 
      !line.startsWith('Nodule') && 
      !line.includes('Coordinates') && 
      !line.includes('Radius') && 
      !line.includes('Confidence') &&
      !line.includes('segmentation performed')
    );
    
    if (titleLine) {
      return titleLine.length > 100 ? titleLine.substring(0, 100) + '...' : titleLine;
    }
    
    return scan.details.length > 100 ? scan.details.substring(0, 100) + '...' : scan.details;
  };

  // Helper function to extract patient name
  const extractPatientName = (scan) => {
    // Default username to use if no patient name is found
    const defaultName = scan.case_name.split('_')[0];
    
    // First check patient_info object if available
    if (scan.patient_info && scan.patient_info.name) {
      return scan.patient_info.name;
    }
    
    // Fall back to extracting from details
    if (!scan.details) return defaultName;
    
    const lines = scan.details.split('\n');
    
    // Look for Name: or Patient Name: in the details
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes('Name:') || lines[i].includes('Patient Name:')) {
        // Extract name from current line if it's after the colon
        if (lines[i].split(':').length > 1) {
          const name = lines[i].split(':')[1].trim();
          return name || defaultName;
        } 
        // Otherwise check the next line which might contain the name
        else if (i + 1 < lines.length) {
          return lines[i + 1].trim() || defaultName;
        }
      }
    }
    
    // Default to the scan owner username if no patient name found
    return defaultName;
  };

  // Helper function to determine if a scan has nodules
  const scanHasNodules = (scan) => {
    return scan.details && 
      !scan.details.includes('No nodules detected') && 
      !scan.details.includes('No detailed information available');
  };

  // Helper function to format date
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (isLoading) {
    return (
      <div className={`text-center py-12 ${darkMode ? 'text-white' : ''}`}>
        <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-indigo-600 border-r-transparent"></div>
        <p className={`mt-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading scan results...</p>
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
      <div className="md:flex md:items-center md:justify-between mb-6">
        <div>
          <h1 className={`text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Scan Results
          </h1>
          <p className={`mt-1 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
            {filteredResults.length} {filteredResults.length === 1 ? 'result' : 'results'} found
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
          <Link to="/upload" className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700">
            Upload New Scan
          </Link>
        </div>
        </div>
        
      {/* Success message */}
      {deleteSuccess && (
        <div className={`mb-6 ${darkMode ? 'bg-green-900/50 border-green-700' : 'bg-green-50 border-green-200'} border rounded-md p-4`}>
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className={`h-5 w-5 ${darkMode ? 'text-green-300' : 'text-green-400'}`} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.707a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className={`text-sm font-medium ${darkMode ? 'text-green-300' : 'text-green-800'}`}>{deleteSuccess}</p>
            </div>
          </div>
        </div>
      )}

      {/* Search and filters */}
      <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-sm mb-6`}>
        <div className="p-4">
          <div className="flex flex-col md:flex-row space-y-3 md:space-y-0 md:space-x-4">
            {/* Search input */}
            <div className="flex-grow relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <MagnifyingGlassIcon className={`h-5 w-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              </div>
              <input
                type="text"
                placeholder="Search scans..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className={`block w-full pl-10 pr-4 py-2 border ${darkMode ? 'bg-gray-700 border-gray-600 placeholder-gray-400 text-white' : 'border-gray-300 placeholder-gray-500 text-gray-900'} rounded-md`}
              />
            </div>

            {/* Sort controls */}
            <div className="flex space-x-2">
              <div className={`inline-flex shadow-sm rounded-md ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <button
                  className={`px-3 py-2 rounded-l-md text-sm font-medium ${sortField === 'timestamp' ? (darkMode ? 'bg-gray-600 text-indigo-300' : 'bg-indigo-100 text-indigo-700') : ''}`}
                  onClick={() => handleSort('timestamp')}
                >
                  Date
                  {sortField === 'timestamp' && (
                    sortDirection === 'asc' 
                      ? <ArrowUpIcon className="ml-1 h-4 w-4 inline" />
                      : <ArrowDownIcon className="ml-1 h-4 w-4 inline" />
                  )}
                </button>
                <button
                  className={`px-3 py-2 text-sm font-medium ${sortField === 'name' ? (darkMode ? 'bg-gray-600 text-indigo-300' : 'bg-indigo-100 text-indigo-700') : ''}`}
                  onClick={() => handleSort('name')}
                >
                  Case ID
                  {sortField === 'name' && (
                    sortDirection === 'asc' 
                      ? <ArrowUpIcon className="ml-1 h-4 w-4 inline" />
                      : <ArrowDownIcon className="ml-1 h-4 w-4 inline" />
                  )}
                </button>
                <button
                  className={`px-3 py-2 rounded-r-md text-sm font-medium ${sortField === 'patient' ? (darkMode ? 'bg-gray-600 text-indigo-300' : 'bg-indigo-100 text-indigo-700') : ''}`}
                  onClick={() => handleSort('patient')}
                >
                  Patient
                  {sortField === 'patient' && (
                    sortDirection === 'asc' 
                      ? <ArrowUpIcon className="ml-1 h-4 w-4 inline" />
                      : <ArrowDownIcon className="ml-1 h-4 w-4 inline" />
                  )}
                </button>
              </div>
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`inline-flex items-center px-4 py-2 border rounded-md text-sm font-medium ${showFilters 
                  ? (darkMode ? 'bg-indigo-700 text-white border-indigo-500' : 'bg-indigo-100 text-indigo-700 border-indigo-300') 
                  : (darkMode ? 'border-gray-600 text-gray-300 hover:bg-gray-700' : 'border-gray-300 text-gray-700 hover:bg-gray-50')
                }`}
              >
                <FunnelIcon className="mr-2 h-4 w-4" />
                Filters
              </button>
        </div>
      </div>

          {/* Expanded filters */}
          {showFilters && (
            <div className={`mt-4 pt-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Time Period</label>
                  <select
                    value={dateFilter}
                    onChange={(e) => setDateFilter(e.target.value)}
                    className={`block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md ${
                      darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white'
                    }`}
                  >
                    <option value="all">All time</option>
                    <option value="today">Last 24 hours</option>
                    <option value="week">Last 7 days</option>
                    <option value="month">Last 30 days</option>
                  </select>
                </div>
                <div>
                  <label className={`block text-sm font-medium mb-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Nodules</label>
                  <select
                    value={noduleFilter}
                    onChange={(e) => setNoduleFilter(e.target.value)}
                    className={`block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md ${
                      darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white'
                    }`}
                  >
                    <option value="all">All scans</option>
                    <option value="with_nodules">With nodules</option>
                    <option value="without_nodules">Without nodules</option>
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Results grid */}
      {filteredResults.length === 0 ? (
        <div className={`text-center py-12 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          <DocumentTextIcon className="mx-auto h-12 w-12 mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-1">No results found</h3>
          <p className="text-sm">Try adjusting your search or filters</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-8">
            {getCurrentPageItems().map(scan => {
              const hasNodules = scanHasNodules(scan);
              const isAnimated = animatedItems.includes(scan.case_name);
              const patientName = extractPatientName(scan);
              const clinicalNotes = extractTitle(scan);
              const creationDate = formatDate(scan.timestamp);
              
              return (
              <Link 
                  to={`/results/${scan.case_name}`} 
                  key={scan.case_name}
                  className={`block ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-lg overflow-hidden transition-all duration-300 hover:shadow-xl transform hover:-translate-y-1 ${
                    isAnimated ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
                  }`}
                >
                  <div className="flex flex-col md:flex-row">
                    {/* Scan Image Preview - Left side */}
                    <div className="relative md:w-1/3 bg-gray-100 overflow-hidden" style={{ height: '260px' }}>
                      <InteractiveViewer 
                        caseId={scan.case_name}
                        selectedNoduleId={null}
                        resultsPageView={true}
                      />
                    </div>

                    {/* Content - Right side */}
                    <div className={`p-6 md:w-2/3 ${hasNodules ? (darkMode ? 'bg-green-900/20' : 'bg-green-50') : ''}`}>
                      <div className="flex justify-between items-start mb-2">
                        <h3 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {patientName}
                        </h3>
                        <div className="flex items-center space-x-2">
                          {hasNodules && (
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              darkMode ? 'bg-green-800 text-green-100' : 'bg-green-100 text-green-800'
                            }`}>
                              Nodules Detected
                            </span>
                          )}
                          {canDeleteScan(scan) && (
                            <button
                              onClick={(e) => openDeleteModal(scan, e)}
                              className={`p-1 rounded-full ${darkMode ? 'hover:bg-gray-700 text-gray-400 hover:text-gray-200' : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700'}`}
                            >
                              <TrashIcon className="h-4 w-4" />
                            </button>
                          )}
                        </div>
                      </div>
                      
                      {/* Date */}
                      <div className={`text-sm mb-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {creationDate}
                      </div>
                      
                      {/* Clinical Notes */}
                      <div className="mb-4">
                        <h4 className={`text-sm font-medium mb-1 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                          Clinical Notes:
                        </h4>
                        <p className={`text-sm line-clamp-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                          {clinicalNotes || "No clinical notes available"}
                        </p>
                      </div>
                      
                      <div className="flex justify-end items-center mt-4">
                        <span className={`font-medium ${darkMode ? 'text-indigo-400' : 'text-indigo-600'}`}>
                          View details →
                        </span>
                      </div>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-8 flex justify-center">
              <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                  className={`relative inline-flex items-center px-2 py-2 rounded-l-md border ${
                    darkMode ? 'border-gray-600 bg-gray-700 text-gray-300' : 'border-gray-300 bg-white text-gray-500'
                  } text-sm font-medium ${
                    currentPage === 1 
                      ? 'opacity-50 cursor-not-allowed' 
                      : darkMode 
                        ? 'hover:bg-gray-600' 
                        : 'hover:bg-gray-50'
                  }`}
                >
                  <span className="sr-only">Previous</span>
                  <ChevronLeftIcon className="h-5 w-5" aria-hidden="true" />
                </button>

                {[...Array(totalPages)].map((_, index) => (
                  <button
                    key={index + 1}
                    onClick={() => handlePageChange(index + 1)}
                    className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                      currentPage === index + 1
                        ? darkMode
                          ? 'z-10 bg-indigo-600 border-indigo-500 text-white'
                          : 'z-10 bg-indigo-50 border-indigo-500 text-indigo-600'
                        : darkMode
                          ? 'border-gray-600 bg-gray-700 text-gray-300 hover:bg-gray-600'
                          : 'border-gray-300 bg-white text-gray-500 hover:bg-gray-50'
                    }`}
                  >
                    {index + 1}
                  </button>
                ))}

                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className={`relative inline-flex items-center px-2 py-2 rounded-r-md border ${
                    darkMode ? 'border-gray-600 bg-gray-700 text-gray-300' : 'border-gray-300 bg-white text-gray-500'
                  } text-sm font-medium ${
                    currentPage === totalPages 
                      ? 'opacity-50 cursor-not-allowed' 
                      : darkMode 
                        ? 'hover:bg-gray-600' 
                        : 'hover:bg-gray-50'
                  }`}
                >
                  <span className="sr-only">Next</span>
                  <ChevronRightIcon className="h-5 w-5" aria-hidden="true" />
                </button>
              </nav>
            </div>
          )}
        </>
      )}

      {/* Delete confirmation modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-10 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" aria-hidden="true" onClick={() => setShowDeleteModal(false)}></div>
            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
            <div className={`inline-block align-bottom ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full`}>
              <div className="px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-red-100 sm:mx-0 sm:h-10 sm:w-10">
                    <ExclamationCircleIcon className="h-6 w-6 text-red-600" aria-hidden="true" />
                  </div>
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className={`text-lg leading-6 font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`} id="modal-title">
                      Delete Scan
                    </h3>
                    <div className="mt-2">
                      <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                        Are you sure you want to delete this scan? This action cannot be undone.
                      </p>
                      <p className={`mt-2 text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                        {scanToDelete?.case_name}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className={`px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <button
                  type="button"
                  className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-red-600 text-base font-medium text-white hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 sm:ml-3 sm:w-auto sm:text-sm"
                  onClick={handleDeleteScan}
                >
                  Delete
                </button>
                <button
                  type="button"
                  className={`mt-3 w-full inline-flex justify-center rounded-md border ${
                    darkMode ? 'border-gray-500 bg-gray-700 text-white hover:bg-gray-600' : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                  } shadow-sm px-4 py-2 text-base font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:w-auto sm:text-sm`}
                  onClick={() => setShowDeleteModal(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
      </div>
      )}
    </div>
  );
};

export default ResultsPage; 