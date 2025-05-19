import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { 
  ExclamationCircleIcon, 
  ArrowLeftIcon, 
  UserCircleIcon,
  TrashIcon 
} from '@heroicons/react/24/outline';
import { useAuth, ROLES } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import InteractiveViewer from '../components/InteractiveViewer';

const ResultDetailPage = () => {
  const { caseId } = useParams();
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedNoduleId, setSelectedNoduleId] = useState(null);
  const { user } = useAuth();
  const { darkMode } = useTheme();
  const navigate = useNavigate();

  // Extract owner username from case ID (if the format is username_filename)
  const ownerUsername = caseId.split('_')[0];
  const isOwner = user.username === ownerUsername;
  const isAdmin = user.role === ROLES.ADMIN;
  const isSuperAdmin = user.role === ROLES.SUPERADMIN;
  
  // Check if the current user can delete this scan
  const canDelete = () => {
    // Superadmin can delete any scan
    if (isSuperAdmin) {
      return true;
    }
    
    // Owner can delete their own scan
    if (isOwner) {
      return true;
    }
    
    // Admin can delete doctors' scans (in a real implementation, you'd check if the admin created the doctor)
    if (isAdmin) {
      return true;
    }
    
    return false;
  };
  
  useEffect(() => {
    const fetchResultDetails = async () => {
      try {
        setIsLoading(true);
        const response = await axios.get(`/api/results/${caseId}`);
        setResult(response.data);
        
        // Get the JWT token to use for image access
        const token = localStorage.getItem('token');
        if (token && response.data.image_url) {
          // Ensure the image URL has the correct format and token
          const baseUrl = response.data.image_url.split('?')[0]; // Remove any existing query params
          response.data.image_url = `${baseUrl}?token=${token}`;
          
          // Log success for debugging
          console.log("Image URL set with authentication token");
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching result details:', error);
        
        // Handle 403 Forbidden errors with a specific message
        if (error.response && error.response.status === 403) {
          setError('You do not have permission to view this scan. You can only view scans you have uploaded.');
        } else {
          setError('Failed to load result details. Please try again later.');
        }
        
        setIsLoading(false);
      }
    };

    fetchResultDetails();
  }, [caseId]);

  // Handle scan deletion
  const handleDeleteScan = async () => {
    try {
      await axios.delete(`/api/results/${caseId}`);
      // Redirect to results page after successful deletion
      navigate('/results', { 
        state: { deleteSuccess: `Scan ${caseId} has been deleted successfully.` }
      });
    } catch (error) {
      console.error('Error deleting scan:', error);
      setError('Failed to delete scan. Please try again later.');
      setShowDeleteModal(false);
    }
  };

  // Handle nodule selection
  const handleNoduleClick = (noduleId) => {
    // First, set the selected nodule ID which will be passed to the InteractiveViewer
    setSelectedNoduleId(noduleId);
    
    // Log for debugging
    console.log(`ResultDetailPage: Clicking on nodule ID: ${noduleId}`);
    
    // Find the nodule in our list to get its details
    const nodule = nodules.find(n => n.id === noduleId);
    if (nodule) {
      console.log(`Found nodule at coordinates z=${nodule.z}, y=${nodule.y}, x=${nodule.x}`);
    }
    
    // Scroll to the InteractiveViewer component for better UX
    const viewerElement = document.getElementById('interactive-viewer');
    if (viewerElement) {
      viewerElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  if (isLoading) {
    return (
      <div className={`text-center py-12 ${darkMode ? 'text-white' : ''}`}>
        <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-indigo-600 border-r-transparent"></div>
        <p className={`mt-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading result details...</p>
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

  if (!result) {
    return (
      <div className={`text-center py-12 ${darkMode ? 'text-white' : ''}`}>
        <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>No result found for this case ID.</p>
        <Link to="/results" className={`mt-4 inline-block ${darkMode ? 'text-indigo-400 hover:text-indigo-300' : 'text-indigo-600 hover:text-indigo-500'}`}>
          Back to all results
        </Link>
      </div>
    );
  }

  // Format the timestamp
  const formattedDate = new Date(result.timestamp).toLocaleString();

  // Parse details if available
  const hasNodules = result.details && !result.details.includes('No nodules detected');
  
  // Extract nodule information if available
  const nodules = [];
  if (hasNodules && result.details) {
    const lines = result.details.split('\n');
    let currentNodule = null;
    
    for (const line of lines) {
      // Skip segmentation information line
      if (line.includes('Nodule segmentation performed:')) {
        continue;
      }
      
      if (line.startsWith('Nodule ')) {
        if (currentNodule) {
          nodules.push(currentNodule);
        }
        
        // Extract the ID properly, making sure to remove any trailing colon
        const noduleIdMatch = line.match(/Nodule\s+(\S+):?/);
        const noduleId = noduleIdMatch ? noduleIdMatch[1].replace(':', '') : line.replace(':', '').trim();
        
        currentNodule = {
          id: noduleId,
          coordinates: '',
          x: 0,
          y: 0, 
          z: 0,
          radius: '',
          confidence: '',
          merged: '',
          malignancy: '',
          cancer_type: '',
          malignancy_score: '',
          cancer_type_confidence: ''
        };
      } else if (currentNodule) {
        if (line.includes('Coordinates')) {
          currentNodule.coordinates = line.split(':')[1].trim();
          
          // Parse coordinates (z,y,x) from format like "(45, 256, 189)"
          try {
            const coordsMatch = currentNodule.coordinates.match(/\((\d+),\s*(\d+),\s*(\d+)\)/);
            if (coordsMatch && coordsMatch.length === 4) {
              currentNodule.z = parseInt(coordsMatch[1]);
              currentNodule.y = parseInt(coordsMatch[2]);
              currentNodule.x = parseInt(coordsMatch[3]);
              console.log(`Parsed nodule ${currentNodule.id} coordinates: z=${currentNodule.z}, y=${currentNodule.y}, x=${currentNodule.x}`);
            }
          } catch (e) {
            console.error('Error parsing coordinates:', e);
          }
        } else if (line.includes('Radius')) {
          currentNodule.radius = line.split(':')[1].trim();
        } else if (line.includes('Confidence') && !line.includes('Cancer Type Confidence')) {
          currentNodule.confidence = line.split(':')[1].trim();
        } else if (line.includes('Merged from')) {
          currentNodule.merged = line.split(':')[1].trim();
        } else if (line.includes('Malignancy Score')) {
          currentNodule.malignancy_score = line.split(':')[1].trim();
        } else if (line.includes('Malignancy') && !line.includes('Score')) {
          currentNodule.malignancy = line.split(':')[1].trim();
        } else if (line.includes('Cancer Type Confidence')) {
          currentNodule.cancer_type_confidence = line.split(':')[1].trim();
        } else if (line.includes('Cancer Type') && !line.includes('Confidence')) {
          currentNodule.cancer_type = line.split(':')[1].trim();
        }
      }
    }
    
    // Add the last nodule
    if (currentNodule) {
      nodules.push(currentNodule);
    }
  }
  
  // Correctly display the count of nodules
  const noduleCount = nodules.length;
  const noduleSummary = hasNodules 
    ? `${noduleCount} potential ${noduleCount === 1 ? 'nodule' : 'nodules'} detected in this scan` 
    : 'No potential nodules detected in this scan';

  return (
    <div className={`space-y-6 ${darkMode ? 'text-white' : ''}`}>
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <Link to="/results" className={`${darkMode ? 'text-gray-300 hover:text-gray-100' : 'text-gray-500 hover:text-gray-700'}`}>
            <ArrowLeftIcon className="h-5 w-5" />
          </Link>
          <h1 className={`text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Result: {result.case_name}
          </h1>
          
          {/* Ownership indicator */}
          {isAdmin && !isOwner ? (
            <span className={`ml-2 px-3 py-1 inline-flex items-center text-xs font-medium rounded-full ${darkMode ? 'bg-purple-900 text-purple-100' : 'bg-purple-100 text-purple-800'}`}>
              <UserCircleIcon className="h-3 w-3 mr-1" />
              Dr. {ownerUsername}'s scan
            </span>
          ) : (
            <span className={`ml-2 px-3 py-1 inline-flex items-center text-xs font-medium rounded-full ${darkMode ? 'bg-blue-900 text-blue-100' : 'bg-blue-100 text-blue-800'}`}>
              <UserCircleIcon className="h-3 w-3 mr-1" />
              Your scan
            </span>
          )}
        </div>
        <div className="flex items-center space-x-4">
          <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
            Processed on {formattedDate}
          </span>
          
          {canDelete() && (
            <button
              onClick={() => setShowDeleteModal(true)}
              className={`px-3 py-1.5 text-sm font-medium flex items-center rounded-md ${
                darkMode 
                  ? 'bg-red-900 text-red-100 hover:bg-red-800' 
                  : 'bg-red-100 text-red-800 hover:bg-red-200'
              }`}
            >
              <TrashIcon className="h-4 w-4 mr-1" />
              Delete Scan
            </button>
          )}
        </div>
      </div>

      {/* Admin viewing notice */}
      {isAdmin && !isOwner && (
        <div className={`${darkMode ? 'bg-yellow-900 border-yellow-700' : 'bg-yellow-50 border-yellow-400'} border-l-4 p-4`}>
          <div className="flex">
            <div className="flex-shrink-0">
              <ExclamationCircleIcon className={`h-5 w-5 ${darkMode ? 'text-yellow-500' : 'text-yellow-400'}`} aria-hidden="true" />
            </div>
            <div className="ml-3">
              <p className={`text-sm ${darkMode ? 'text-yellow-200' : 'text-yellow-700'}`}>
                You are viewing this scan as an administrator. This scan was uploaded by <strong>Dr. {ownerUsername}</strong>.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Interactive Visualization */}
      <div id="interactive-viewer" className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow rounded-lg overflow-hidden`}>
        <div className={`px-4 py-3 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} sm:px-6`}>
          <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>Interactive Visualization</h3>
        </div>
        <InteractiveViewer 
          caseId={caseId} 
          selectedNoduleId={selectedNoduleId}
          onNoduleClick={handleNoduleClick}
        />
      </div>

      {/* Result details */}
      <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow rounded-lg overflow-hidden`}>
        <div className={`px-4 py-5 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} sm:px-6`}>
          <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>Details</h3>
        </div>
        <div className="p-6">
          {/* Patient information section if available */}
          {result.patient_info && (
            <div className={`mb-6 p-4 rounded-md ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <h4 className={`text-md font-medium ${darkMode ? 'text-white' : 'text-gray-900'} mb-2`}>
                Patient Information
              </h4>
              <div className={`grid grid-cols-1 md:grid-cols-2 gap-4 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                <div>
                  <p className="text-sm font-medium mb-1">Patient Name:</p>
                  <p>{result.patient_info.name || 'Not provided'}</p>
                </div>
                <div>
                  <p className="text-sm font-medium mb-1">Age:</p>
                  <p>{result.patient_info.age || 'Not provided'}</p>
                </div>
                {result.patient_info.description && (
                  <div className="col-span-1 md:col-span-2">
                    <p className="text-sm font-medium mb-1">Clinical Notes:</p>
                    <p className="whitespace-pre-wrap">{result.patient_info.description}</p>
                  </div>
                )}
                <div className="col-span-1 md:col-span-2 mt-2">
                  <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Added by: {result.patient_info.user} on {new Date(result.patient_info.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          )}

          {hasNodules ? (
            <div>
              <div className="mb-4">
                <h4 className={`text-md font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>Summary</h4>
                <p className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  {noduleSummary}
                </p>
              </div>
              
              <h4 className={`text-md font-medium ${darkMode ? 'text-white' : 'text-gray-900'} mb-2`}>Detected Nodules</h4>
              <div className="overflow-x-auto">
                <table className={`min-w-full divide-y ${darkMode ? 'divide-gray-700' : 'divide-gray-200'}`}>
                  <thead className={`${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                    <tr>
                      <th scope="col" className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                        Nodule
                      </th>
                      <th scope="col" className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                        Coordinates (z,y,x)
                      </th>
                      <th scope="col" className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                        Radius (mm)
                      </th>
                      <th scope="col" className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                        Confidence
                      </th>
                      <th scope="col" className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                        Malignancy
                      </th>
                      <th scope="col" className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>
                        Cancer Type
                      </th>
                    </tr>
                  </thead>
                  <tbody className={`${darkMode ? 'bg-gray-800 divide-y divide-gray-700' : 'bg-white divide-y divide-gray-200'}`}>
                    {nodules.map((nodule, index) => (
                      <tr 
                        key={index} 
                        className={`${index % 2 === 0 ? (darkMode ? 'bg-gray-900' : 'bg-gray-50') : ''} 
                          cursor-pointer hover:${darkMode ? 'bg-gray-700' : 'bg-gray-100'}
                          ${selectedNoduleId === nodule.id ? (darkMode ? 'bg-blue-900' : 'bg-blue-100') : ''}`}
                        onClick={() => handleNoduleClick(nodule.id)}
                      >
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {nodule.id}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {nodule.coordinates}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {nodule.radius}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {nodule.confidence}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {nodule.malignancy}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {nodule.cancer_type}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className={`p-4 rounded-md ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                {noduleSummary}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultDetailPage;