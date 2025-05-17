import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, ExclamationCircleIcon, InformationCircleIcon, CheckCircleIcon, XCircleIcon, ViewfinderCircleIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';
import InteractiveViewer from '../components/InteractiveViewer';

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [rawFile, setRawFile] = useState(null);
  const [confidence, setConfidence] = useState(0.5);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [showRawUpload, setShowRawUpload] = useState(false);
  const [previewResult, setPreviewResult] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [selectedNoduleId, setSelectedNoduleId] = useState(null);
  const [showLeaveConfirmation, setShowLeaveConfirmation] = useState(false);
  const [showPatientForm, setShowPatientForm] = useState(false);
  const [patientInfo, setPatientInfo] = useState({
    name: '',
    age: '',
    description: ''
  });
  const [isNavigating, setIsNavigating] = useState(false);
  const [pendingNavigation, setPendingNavigation] = useState(null);
  const [uploadMode, setUploadMode] = useState('single'); // 'single' or 'dicom-volume'
  const [dicomFiles, setDicomFiles] = useState([]);
  const navigate = useNavigate();
  const location = useLocation();
  const { darkMode } = useTheme();
  
  // Track current location to detect changes
  const [prevLocation, setPrevLocation] = useState(null);
  
  // Handle location changes to detect navigation attempts
  useEffect(() => {
    // Skip on first render
    if (prevLocation === null) {
      setPrevLocation(location);
      return;
    }
    
    // If the location has changed and we haven't explicitly allowed navigation
    if (
      location !== prevLocation && 
      !isNavigating && 
      showPreview && 
      previewResult
    ) {
      // Block navigation by going back to previous location
      // This happens synchronously, before the confirmation dialog
      navigate(-1);
      
      // Show confirmation dialog
      setShowLeaveConfirmation(true);
      setPendingNavigation(location.pathname + location.search);
      
      return;
    }
    
    // Update the previous location
    setPrevLocation(location);
  }, [location, prevLocation, isNavigating, showPreview, previewResult, navigate]);

  // Browser back/refresh confirmation
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (showPreview && previewResult) {
        e.preventDefault();
        e.returnValue = '';
        return '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [showPreview, previewResult]);

  // Custom navigation function (can be used for in-app links that need confirmation)
  const customNavigate = (to) => {
    if (showPreview && previewResult) {
      // Store pending navigation and show confirmation
      setPendingNavigation(to);
      setShowLeaveConfirmation(true);
    } else {
      // If no unsaved changes, navigate directly
      navigate(to);
    }
  };

  // Configure dropzone for CT scan files
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/octet-stream': ['.mhd', '.raw', '.nii.gz', '.nii'],
      'application/dicom': ['.dcm'],
    },
    multiple: true, // Allow multiple files to select both .mhd and .raw together
    onDrop: (acceptedFiles) => {
      // Reset preview state when new files are dropped
      setPreviewResult(null);
      setShowPreview(false);
      
      // Sort files to identify MHD and RAW files
      const mhdFile = acceptedFiles.find(file => file.name.toLowerCase().endsWith('.mhd'));
      const rawFile = acceptedFiles.find(file => file.name.toLowerCase().endsWith('.raw'));
      
      if (mhdFile && rawFile) {
        // Both files uploaded together
        setFiles([mhdFile]);
        setRawFile(rawFile);
        setShowRawUpload(false);
        setError(null);
      } else if (mhdFile) {
        // Only MHD file uploaded
        setFiles([mhdFile]);
        setShowRawUpload(true);
        setError("You've uploaded an MHD file. Please also upload the associated RAW file.");
      } else if (rawFile) {
        // Only RAW file uploaded
        setRawFile(rawFile);
        setShowRawUpload(true);
        if (files.length === 0 || !files[0].name.toLowerCase().endsWith('.mhd')) {
          setError("You've uploaded a RAW file. Please also upload the associated MHD file.");
        } else {
          setError(null);
        }
      } else if (acceptedFiles.length > 0) {
        // Other valid file types (NIFTI, DICOM)
        setFiles(acceptedFiles);
        setShowRawUpload(false);
        setError(null);
      }
    },
  });

  // Configure dropzone specifically for RAW files
  const { getRootProps: getRawRootProps, getInputProps: getRawInputProps, isDragActive: isRawDragActive } = useDropzone({
    accept: {
      'application/octet-stream': ['.raw'],
    },
    multiple: false,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setRawFile(acceptedFiles[0]);
        setError(null);
      }
    },
  });

  // Configure dropzone for DICOM series upload
  const { getRootProps: getDicomVolumeRootProps, getInputProps: getDicomVolumeInputProps, isDragActive: isDicomVolumeDragActive } = useDropzone({
    accept: {
      'application/dicom': ['.dcm'],
      'application/zip': ['.zip'],
    },
    multiple: true,
    onDrop: (acceptedFiles) => {
      // Reset preview state when new files are dropped
      setPreviewResult(null);
      setShowPreview(false);
      
      // Check if there's a single zip file
      const zipFile = acceptedFiles.find(file => file.name.toLowerCase().endsWith('.zip'));
      if (zipFile) {
        setDicomFiles([zipFile]);
        setError(null);
        return;
      }
      
      // Filter only DICOM files
      const dicomFilesOnly = acceptedFiles.filter(file => file.name.toLowerCase().endsWith('.dcm'));
      
      if (dicomFilesOnly.length === 0) {
        setError("Please upload DICOM (.dcm) files or a ZIP file containing DICOM files.");
        return;
      }
      
      if (dicomFilesOnly.length < 3) {
        setError("Please upload at least 3 DICOM slices to create a 3D volume.");
        return;
      }
      
      setDicomFiles(dicomFilesOnly);
      setError(null);
    },
  });

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select a file to upload');
      return;
    }

    // If user selected an MHD file but no RAW file, show error
    if (files[0].name.toLowerCase().endsWith('.mhd') && !rawFile) {
      setError('Please upload the associated RAW file for your MHD file');
      return;
    }

    // Validate file sizes
    const maxSizeInBytes = 300 * 1024 * 1024; // 300MB
    if (files[0].size > maxSizeInBytes) {
      setError(`File size exceeds the maximum limit of 300MB. Your file is ${(files[0].size / (1024 * 1024)).toFixed(2)}MB.`);
      return;
    }
    
    if (rawFile && rawFile.size > maxSizeInBytes) {
      setError(`RAW file size exceeds the maximum limit of 300MB. Your file is ${(rawFile.size / (1024 * 1024)).toFixed(2)}MB.`);
      return;
    }

    // Check file name patterns for MHD and RAW to ensure they match
    if (files[0].name.toLowerCase().endsWith('.mhd') && rawFile) {
      const mhdBaseName = files[0].name.toLowerCase().slice(0, -4); // Remove .mhd
      const rawBaseName = rawFile.name.toLowerCase().slice(0, -4);  // Remove .raw
      
      if (mhdBaseName !== rawBaseName && !window.confirm(`The MHD file (${files[0].name}) and RAW file (${rawFile.name}) have different base names. Are you sure they belong together?`)) {
        setError('Please make sure your MHD and RAW files have matching names');
        return;
      }
    }

    setIsUploading(true);
    setProgress(0);
    setProcessingProgress(0);
    setError(null);

    const formData = new FormData();
    formData.append('file', files[0]);
    
    // Add RAW file if present
    if (rawFile) {
      formData.append('raw_file', rawFile);
    }
    
    // Ensure confidence value is a string with proper format
    const confidenceValue = confidence.toString();
    console.log("Uploading with confidence value:", confidenceValue);
    formData.append('confidence', confidenceValue);
    
    // Debug the form data being sent
    for (let pair of formData.entries()) {
      console.log(pair[0] + ': ' + pair[1]);
    }
    
    // Always set no_segmentation to true to disable nodule segmentation completely
    formData.append('lungs_only', 'false');
    formData.append('no_segmentation', 'true');

    try {
      // Upload phase
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
          
          // When upload is complete (100%), start the processing progress simulation
          if (percentCompleted === 100) {
            setIsUploading(false);
            setIsProcessing(true);
            
            // Simulate processing progress
            // This is just a visual indicator since we don't have real-time backend progress updates
            let currentProgress = 0;
            const processingInterval = setInterval(() => {
              // Increment progress in a non-linear way to simulate real processing
              // Start slower, then speed up, then slow down again at the end
              if (currentProgress < 30) {
                currentProgress += 1;
              } else if (currentProgress < 70) {
                currentProgress += 2;
              } else if (currentProgress < 90) {
                currentProgress += 0.5;
              } else if (currentProgress < 98) {
                currentProgress += 0.2;
              }
              
              if (currentProgress >= 99) {
                clearInterval(processingInterval);
                currentProgress = 99; // Let the final 100% happen when we get the response
              }
              
              setProcessingProgress(Math.min(99, currentProgress));
            }, 300);
          }
        },
      });

      if (response.data.success) {
        // Set processing to complete
        setProcessingProgress(100);
        setIsProcessing(false);
        
        // Now fetch the preview result to display
        const jobId = response.data.job_id;
        try {
          const previewResponse = await axios.get(`/api/results/preview/${jobId}`);
          setPreviewResult(previewResponse.data);
          setShowPreview(true);
        } catch (previewError) {
          console.error('Preview error:', previewError);
          setError(previewError.response?.data?.error || 'Error retrieving result preview');
        }
      } else {
        setError(response.data.error || 'Unknown error occurred');
        setIsUploading(false);
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Upload error:', error);
      
      // Provide more detailed error messages based on the error
      let errorMessage = 'Error uploading file. Please try again.';
      
      if (error.response) {
        // Server responded with an error
        console.error('Error response:', error.response.data);
        if (error.response.status === 413) {
          errorMessage = 'File too large. The maximum allowed size is 300MB.';
        } else if (error.response.data && error.response.data.error) {
          errorMessage = error.response.data.error;
        } else {
          errorMessage = `Server error (${error.response.status}). Please try again later.`;
        }
      } else if (error.request) {
        // Request made but no response received
        errorMessage = 'No response from server. Please check your network connection.';
      }
      
      setError(errorMessage);
      setIsUploading(false);
      setIsProcessing(false);
    }
  };

  const handleDicomVolumeUpload = async () => {
    if (dicomFiles.length === 0) {
      setError('Please select DICOM files to upload');
      return;
    }

    // Validate file sizes (total size limit check)
    const maxTotalSizeInBytes = 500 * 1024 * 1024; // 500MB
    const totalSize = dicomFiles.reduce((sum, file) => sum + file.size, 0);
    
    if (totalSize > maxTotalSizeInBytes) {
      setError(`Total file size exceeds the maximum limit of 500MB. Your files total ${(totalSize / (1024 * 1024)).toFixed(2)}MB.`);
      return;
    }

    setIsUploading(true);
    setProgress(0);
    setProcessingProgress(0);
    setError(null);

    const formData = new FormData();
    
    // Add all DICOM files to the form data
    dicomFiles.forEach(file => {
      formData.append('files[]', file);
    });
    
    // Add processing options
    formData.append('confidence', confidence.toString());
    formData.append('lungs_only', 'false');
    formData.append('no_segmentation', 'true');

    try {
      // Upload phase
      const response = await axios.post('/api/upload/dicom-volume', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
          
          // When upload is complete (100%), start the processing progress simulation
          if (percentCompleted === 100) {
            setIsUploading(false);
            setIsProcessing(true);
            
            // Simulate processing progress
            // This is just a visual indicator since we don't have real-time backend progress updates
            let currentProgress = 0;
            const processingInterval = setInterval(() => {
              // Increment progress in a non-linear way to simulate real processing
              // Start slower, then speed up, then slow down again at the end
              if (currentProgress < 30) {
                currentProgress += 1;
              } else if (currentProgress < 70) {
                currentProgress += 2;
              } else if (currentProgress < 90) {
                currentProgress += 0.5;
              } else if (currentProgress < 98) {
                currentProgress += 0.2;
              }
              
              if (currentProgress >= 99) {
                clearInterval(processingInterval);
              }
              
              setProcessingProgress(currentProgress);
            }, 500);
          }
        }
      });
      
      console.log("DICOM volume upload response:", response.data);
      
      if (response.data && response.data.job_id) {
        // Start polling for the processing status
        const jobId = response.data.job_id;
        const pollInterval = setInterval(async () => {
          try {
            const statusResponse = await axios.get(`/api/results/preview/${jobId}`);
            console.log("Poll status response:", statusResponse.data);
            
            // Check if we have a status field
            if (statusResponse.data.status === 'completed') {
              clearInterval(pollInterval);
              setIsProcessing(false);
              setPreviewResult(statusResponse.data);
              setShowPreview(true);
            } else if (statusResponse.data.status === 'error') {
              clearInterval(pollInterval);
              setIsProcessing(false);
              setError(`Error processing DICOM volume: ${statusResponse.data.error || 'Unknown error'}`);
            } else {
              // Still processing
              console.log("Processing still in progress...");
            }
          } catch (err) {
            console.error("Error polling for job status:", err);
            // Count consecutive errors for potential timeout
            if (!window.pollErrorCount) window.pollErrorCount = 0;
            window.pollErrorCount++;
            
            // If we've had too many consecutive errors, stop polling
            if (window.pollErrorCount > 30) { // After about 2.5 minutes of errors
              clearInterval(pollInterval);
              setIsProcessing(false);
              setError("Timed out waiting for processing to complete. The process may still be running in the background.");
            }
          }
        }, 5000);
      } else {
        setIsProcessing(false);
        setError("Invalid response from server. Processing may have failed.");
      }
      
    } catch (err) {
      console.error("Error uploading DICOM volume:", err);
      setIsUploading(false);
      setIsProcessing(false);
      setError(err.response?.data?.error || `Error uploading DICOM volume: ${err.message}`);
    }
  };

  // Handle patient info form inputs
  const handlePatientInfoChange = (e) => {
    const { name, value } = e.target;
    setPatientInfo(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle saving with patient info
  const handleSaveWithPatientInfo = async () => {
    if (!previewResult || !previewResult.job_id) {
      setError('No result to save');
      return;
    }

    try {
      const response = await axios.post(`/api/results/action/${previewResult.job_id}`, {
        action: 'save',
        patient_info: patientInfo
      });

      if (response.data.success) {
        // Hide the form
        setShowPatientForm(false);
        
        // Set navigating flag to allow navigation
        setIsNavigating(true);
        
        // If we have a pending navigation, go there
        if (pendingNavigation) {
          navigate(pendingNavigation);
          setPendingNavigation(null);
        } else {
          // Navigate to the result page using the path provided in the response
          const resultPath = response.data.result_path || `/results/${response.data.job_id || response.data.case_name}`;
          
          // Log the navigation
          console.log(`Navigating to result path: ${resultPath}`);
          
          // Navigate to the result page
          navigate(resultPath.replace('/api', ''));
        }
      } else {
        setError(response.data.error || 'Failed to save results');
      }
    } catch (error) {
      console.error('Error saving result:', error);
      setError(error.response?.data?.error || 'Error saving result');
    }
  };

  // Handle result actions (save/discard)
  const handleResultAction = async (action) => {
    if (!previewResult || !previewResult.job_id) {
      setError('No result to save or discard');
      return;
    }

    if (action === 'save') {
      // Show patient information form
      setShowPatientForm(true);
    } else {
      try {
        const response = await axios.post(`/api/results/action/${previewResult.job_id}`, {
          action: action
        });

        if (response.data.success) {
          // Reset the form on discard
          setFiles([]);
          setRawFile(null);
          setPreviewResult(null);
          setShowPreview(false);
          
          // Allow navigation after discarding
          setIsNavigating(true);
          
          // If we were trying to navigate somewhere when discarding, navigate there
          if (pendingNavigation) {
            navigate(pendingNavigation);
            setPendingNavigation(null);
          }
        } else {
          setError(response.data.error || `Failed to ${action} results`);
        }
      } catch (error) {
        console.error(`Error ${action}ing result:`, error);
        setError(error.response?.data?.error || `Error ${action}ing result`);
      }
    }
  };

  // Handle action on leave confirmation
  const handleLeaveConfirmation = (shouldSave) => {
    if (shouldSave === 'save') {
      // Show patient form
      setShowPatientForm(true);
      setShowLeaveConfirmation(false);
    } else if (shouldSave === 'discard') {
      // Discard and continue navigation
      handleResultAction('discard');
      setShowLeaveConfirmation(false);
      
      // Set navigating flag to allow navigation
      setIsNavigating(true);
      
      // If we have a pending navigation, go there
      if (pendingNavigation) {
        navigate(pendingNavigation);
        setPendingNavigation(null);
      }
    } else {
      // Stay on page
      setShowLeaveConfirmation(false);
      setPendingNavigation(null);
    }
  };

  // Handle nodule selection
  const handleNoduleClick = (noduleId) => {
    setSelectedNoduleId(noduleId);
  };

  return (
    <div className={`container mx-auto px-4 py-8 ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
      {/* Add this section for mode switching */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-4">Upload CT Scan</h1>
        
        <div className="flex space-x-4 mb-4">
          <button
            onClick={() => setUploadMode('single')}
            className={`px-4 py-2 rounded-lg ${uploadMode === 'single' 
              ? (darkMode ? 'bg-indigo-600 text-white' : 'bg-indigo-100 text-indigo-700') 
              : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700')
            }`}
          >
            Single CT Scan
          </button>
          <button
            onClick={() => setUploadMode('dicom-volume')}
            className={`px-4 py-2 rounded-lg ${uploadMode === 'dicom-volume' 
              ? (darkMode ? 'bg-indigo-600 text-white' : 'bg-indigo-100 text-indigo-700') 
              : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700')
            }`}
          >
            DICOM Volume
          </button>
        </div>
      </div>
      
      {!showPreview ? (
        <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          {/* Show different upload interface based on mode */}
          {uploadMode === 'single' ? (
            <>
              <h2 className="text-xl font-semibold mb-4">Upload Single CT Scan</h2>
              <p className="mb-6 text-sm">
                Supported file formats: .mhd/.raw pair, .nii, .nii.gz, or single .dcm
              </p>
              
              {/* Original single file upload section */}
              <div 
                {...getRootProps()} 
                className={`border-2 border-dashed rounded-lg p-8 cursor-pointer transition-colors ${
                  isDragActive 
                    ? (darkMode ? 'border-indigo-400 bg-indigo-900 bg-opacity-20' : 'border-indigo-500 bg-indigo-50') 
                    : (darkMode ? 'border-gray-600 hover:border-gray-500' : 'border-gray-300 hover:border-gray-400')
                }`}
              >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center justify-center">
                  <CloudArrowUpIcon className={`h-12 w-12 mb-3 ${isDragActive ? 'text-indigo-500' : 'text-gray-400'}`} />
                  {isDragActive ? (
                    <p>Drop the file here...</p>
                  ) : (
                    <>
                      <p className="text-center">Drag & drop your CT scan file here, or click to select file</p>
                      <p className="text-center text-xs mt-2 text-gray-500">
                        For MHD files, you can upload the RAW file separately in the next step
                      </p>
                    </>
                  )}
                </div>
              </div>
              
              {/* Display selected files */}
              {files.length > 0 && (
                <div className={`mt-4 p-4 rounded-md ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                  <h3 className="font-medium mb-2">Selected File:</h3>
                  <div className="flex justify-between items-center">
                    <span className="truncate">{files[0].name} ({(files[0].size / (1024 * 1024)).toFixed(2)} MB)</span>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setFiles([]);
                        setShowRawUpload(false);
                        setError(null);
                      }}
                      className="text-red-500 hover:text-red-700"
                    >
                      <XMarkIcon className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              )}
              
              {/* Raw file upload section for MHD files */}
              {showRawUpload && (
                <div className="mt-4">
                  <h3 className="font-medium mb-2">Upload associated RAW file:</h3>
                  <div 
                    {...getRawRootProps()} 
                    className={`border-2 border-dashed rounded-lg p-6 cursor-pointer transition-colors ${
                      isRawDragActive 
                        ? (darkMode ? 'border-indigo-400 bg-indigo-900 bg-opacity-20' : 'border-indigo-500 bg-indigo-50') 
                        : (darkMode ? 'border-gray-600 hover:border-gray-500' : 'border-gray-300 hover:border-gray-400')
                    }`}
                  >
                    <input {...getRawInputProps()} />
                    <div className="flex flex-col items-center justify-center">
                      <CloudArrowUpIcon className={`h-8 w-8 mb-2 ${isRawDragActive ? 'text-indigo-500' : 'text-gray-400'}`} />
                      {isRawDragActive ? (
                        <p>Drop the RAW file here...</p>
                      ) : (
                        <p className="text-center text-sm">Drag & drop your RAW file here, or click to select file</p>
                      )}
                    </div>
                  </div>
                  
                  {/* Display selected RAW file */}
                  {rawFile && (
                    <div className={`mt-2 p-3 rounded-md ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                      <div className="flex justify-between items-center">
                        <span className="truncate">{rawFile.name} ({(rawFile.size / (1024 * 1024)).toFixed(2)} MB)</span>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            setRawFile(null);
                          }}
                          className="text-red-500 hover:text-red-700"
                        >
                          <XMarkIcon className="h-5 w-5" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Original upload button */}
              <div className="mt-6">
                <button 
                  onClick={handleUpload}
                  disabled={isUploading || isProcessing || files.length === 0}
                  className={`px-4 py-2 rounded-md text-white font-medium 
                    ${isUploading || isProcessing || files.length === 0 
                      ? 'bg-gray-400 cursor-not-allowed' 
                      : 'bg-blue-600 hover:bg-blue-700'}
                  `}
                >
                  {isUploading ? 'Uploading...' : isProcessing ? 'Processing...' : 'Upload and Process'}
                </button>
              </div>
            </>
          ) : (
            <>
              <h2 className="text-xl font-semibold mb-4">Upload DICOM Volume</h2>
              <p className="mb-6 text-sm">
                Upload multiple DICOM (.dcm) files to create a 3D volume, or a single .zip file containing DICOM files
              </p>
              
              {/* DICOM Volume file upload section */}
              <div 
                {...getDicomVolumeRootProps()} 
                className={`border-2 border-dashed rounded-lg p-8 cursor-pointer transition-colors ${
                  isDicomVolumeDragActive 
                    ? (darkMode ? 'border-indigo-400 bg-indigo-900 bg-opacity-20' : 'border-indigo-500 bg-indigo-50') 
                    : (darkMode ? 'border-gray-600 hover:border-gray-500' : 'border-gray-300 hover:border-gray-400')
                }`}
              >
                <input {...getDicomVolumeInputProps()} />
                <div className="flex flex-col items-center justify-center">
                  <CloudArrowUpIcon className={`h-12 w-12 mb-3 ${isDicomVolumeDragActive ? 'text-indigo-500' : 'text-gray-400'}`} />
                  {isDicomVolumeDragActive ? (
                    <p>Drop the files here...</p>
                  ) : (
                    <>
                      <p className="text-center">Drag & drop multiple DICOM files or a ZIP file here, or click to select files</p>
                      <p className="text-center text-xs mt-2 text-gray-500">
                        All slices from the same DICOM series will be combined into a 3D volume
                      </p>
                    </>
                  )}
                </div>
              </div>
              
              {/* Display selected DICOM files */}
              {dicomFiles.length > 0 && (
                <div className={`mt-4 p-4 rounded-md ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                  <h3 className="font-medium mb-2">Selected Files: {dicomFiles.length}</h3>
                  <div className="max-h-40 overflow-y-auto pr-2">
                    {dicomFiles.length === 1 && dicomFiles[0].name.toLowerCase().endsWith('.zip') ? (
                      <div className="flex justify-between items-center">
                        <span className="truncate">{dicomFiles[0].name} ({(dicomFiles[0].size / (1024 * 1024)).toFixed(2)} MB)</span>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            setDicomFiles([]);
                            setError(null);
                          }}
                          className="text-red-500 hover:text-red-700"
                        >
                          <XMarkIcon className="h-5 w-5" />
                        </button>
                      </div>
                    ) : (
                      <div>
                        <div className="flex justify-between mb-2">
                          <span>{dicomFiles.length} DICOM files selected ({(dicomFiles.reduce((sum, file) => sum + file.size, 0) / (1024 * 1024)).toFixed(2)} MB total)</span>
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setDicomFiles([]);
                              setError(null);
                            }}
                            className="text-red-500 hover:text-red-700"
                          >
                            <XMarkIcon className="h-5 w-5" />
                          </button>
                        </div>
                        <div className="text-xs text-gray-400">Files: {dicomFiles.slice(0, 3).map(f => f.name).join(', ')}{dicomFiles.length > 3 ? ` ... and ${dicomFiles.length - 3} more` : ''}</div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* DICOM Volume upload button */}
              <div className="mt-6">
                <button 
                  onClick={handleDicomVolumeUpload}
                  disabled={isUploading || isProcessing || dicomFiles.length === 0}
                  className={`px-4 py-2 rounded-md text-white font-medium 
                    ${isUploading || isProcessing || dicomFiles.length === 0 
                      ? 'bg-gray-400 cursor-not-allowed' 
                      : 'bg-blue-600 hover:bg-blue-700'}
                  `}
                >
                  {isUploading ? 'Uploading...' : isProcessing ? 'Processing...' : 'Upload and Process'}
                </button>
              </div>
            </>
          )}
        </div>
      ) : (
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow rounded-lg p-6 max-w-4xl mx-auto`}>
          <div className="space-y-6">
            <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Review Scan Results
            </h2>
            
            <div className={`${darkMode ? 'bg-blue-900' : 'bg-blue-50'} p-4 rounded-md mb-4`}>
              <div className="flex">
                <InformationCircleIcon className={`h-5 w-5 ${darkMode ? 'text-blue-300' : 'text-blue-400'}`} />
                <div className={`ml-3 text-sm ${darkMode ? 'text-blue-200' : 'text-blue-700'}`}>
                  <p>Please review the results below. Use the interactive visualization to examine detected nodules. You can save the results to your records or discard them if not satisfied.</p>
                </div>
              </div>
            </div>
            
            {/* Visualization section */}
            <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-lg overflow-hidden`}>
              <div className={`px-4 py-3 border-b ${darkMode ? 'border-gray-600' : 'border-gray-200'} sm:px-6`}>
                <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>Interactive Visualization</h3>
              </div>
              
              {/* Interactive viewer */}
              <div className="p-4">
                <InteractiveViewer 
                  caseId={previewResult.case_name} 
                  selectedNoduleId={selectedNoduleId}
                />
              </div>
            </div>
            
            {/* Nodule table for interactive navigation */}
            {previewResult.details && previewResult.details.includes('Nodule ') && (
              <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} p-4 rounded-md`}>
                <h3 className={`text-md font-medium ${darkMode ? 'text-white' : 'text-gray-900'} mb-2`}>
                  Detected Nodules:
                </h3>
                <div className="overflow-x-auto">
                  <table className={`min-w-full divide-y ${darkMode ? 'divide-gray-600' : 'divide-gray-200'}`}>
                    <thead className={`${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
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
                      </tr>
                    </thead>
                    <tbody className={`${darkMode ? 'bg-gray-700 divide-y divide-gray-600' : 'bg-white divide-y divide-gray-200'}`}>
                      {extractNodulesFromDetails(previewResult.details).map((nodule, index) => (
                        <tr 
                          key={index} 
                          className={`${index % 2 === 0 ? (darkMode ? 'bg-gray-800' : 'bg-gray-50') : ''} 
                            cursor-pointer hover:${darkMode ? 'bg-gray-600' : 'bg-gray-100'} ${selectedNoduleId === nodule.id ? (darkMode ? 'bg-indigo-900' : 'bg-indigo-100') : ''}`}
                          onClick={() => handleNoduleClick(nodule.id)}
                        >
                          <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                            {nodule.id}
                          </td>
                          <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                            {nodule.coordinates}
                          </td>
                          <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                            {nodule.radius}
                          </td>
                          <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                            {nodule.confidence}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            
            {/* Details section */}
            <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} p-4 rounded-md`}>
              <h3 className={`text-md font-medium ${darkMode ? 'text-white' : 'text-gray-900'} mb-2`}>
                Analysis Details:
              </h3>
              <pre className={`text-sm whitespace-pre-wrap ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                {previewResult.details || "No detailed analysis available."}
              </pre>
            </div>
            
            {/* Action buttons */}
            <div className="flex space-x-4 justify-center mt-6">
              <button
                onClick={() => handleResultAction('discard')}
                className={`px-4 py-2 rounded-md flex items-center ${
                  darkMode 
                    ? 'bg-red-800 text-white hover:bg-red-700' 
                    : 'bg-red-600 text-white hover:bg-red-700'
                }`}
              >
                <XCircleIcon className="h-5 w-5 mr-2" />
                Discard Results
              </button>
              
              <button
                onClick={() => handleResultAction('save')}
                className={`px-4 py-2 rounded-md flex items-center ${
                  darkMode 
                    ? 'bg-green-800 text-white hover:bg-green-700' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                <CheckCircleIcon className="h-5 w-5 mr-2" />
                Save to Records
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Leave confirmation dialog */}
      {showLeaveConfirmation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-xl p-6 max-w-md w-full`}>
            <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'} mb-4`}>
              Unsaved Results
            </h3>
            <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} mb-6`}>
              You have unsaved analysis results. What would you like to do?
            </p>
            <div className="flex justify-end space-x-4">
              <button
                onClick={() => handleLeaveConfirmation('stay')}
                className={`px-4 py-2 rounded-md ${
                  darkMode 
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                    : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                }`}
              >
                Stay on Page
              </button>
              <button
                onClick={() => handleLeaveConfirmation('discard')}
                className={`px-4 py-2 rounded-md ${
                  darkMode 
                    ? 'bg-red-800 text-white hover:bg-red-700' 
                    : 'bg-red-600 text-white hover:bg-red-700'
                }`}
              >
                Discard & Leave
              </button>
              <button
                onClick={() => handleLeaveConfirmation('save')}
                className={`px-4 py-2 rounded-md ${
                  darkMode 
                    ? 'bg-green-800 text-white hover:bg-green-700' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                Save Results
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Patient information form */}
      {showPatientForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-xl p-6 max-w-md w-full`}>
            <div className="flex justify-between items-center mb-4">
              <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Patient Information
              </h3>
              <button 
                onClick={() => setShowPatientForm(false)} 
                className={`rounded-full p-1 ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-200'}`}
              >
                <XMarkIcon className={`h-5 w-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              </button>
            </div>
            
            <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} mb-4`}>
              Please enter the patient's information to save this result to their medical record.
            </p>
            
            <div className="space-y-4">
              <div>
                <label htmlFor="name" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-1`}>
                  Patient Name
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={patientInfo.name}
                  onChange={handlePatientInfoChange}
                  required
                  className={`w-full px-3 py-2 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500`}
                />
              </div>
              
              <div>
                <label htmlFor="age" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-1`}>
                  Patient Age
                </label>
                <input
                  type="number"
                  id="age"
                  name="age"
                  value={patientInfo.age}
                  onChange={handlePatientInfoChange}
                  min="0"
                  max="120"
                  className={`w-full px-3 py-2 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500`}
                />
              </div>
              
              <div>
                <label htmlFor="description" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-1`}>
                  Clinical Notes
                </label>
                <textarea
                  id="description"
                  name="description"
                  value={patientInfo.description}
                  onChange={handlePatientInfoChange}
                  rows="3"
                  className={`w-full px-3 py-2 border ${darkMode ? 'bg-gray-700 border-gray-600 text-white' : 'bg-white border-gray-300 text-gray-900'} rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500`}
                  placeholder="Enter any relevant clinical notes about this scan..."
                ></textarea>
              </div>
            </div>
            
            <div className="flex justify-end space-x-4 mt-6">
              <button
                onClick={() => setShowPatientForm(false)}
                className={`px-4 py-2 rounded-md ${
                  darkMode 
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                    : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                }`}
              >
                Cancel
              </button>
              <button
                onClick={handleSaveWithPatientInfo}
                disabled={!patientInfo.name || !patientInfo.age}
                className={`px-4 py-2 rounded-md ${
                  !patientInfo.name || !patientInfo.age
                    ? `${darkMode ? 'bg-green-900 text-gray-400' : 'bg-green-300 text-gray-600'} cursor-not-allowed`
                    : darkMode 
                      ? 'bg-green-800 text-white hover:bg-green-700' 
                      : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                Save to Records
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function to extract nodule information from details text
const extractNodulesFromDetails = (details) => {
  if (!details) return [];

  const nodules = [];
  const lines = details.split('\n');
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
      currentNodule = {
        id: line.replace(':', '').trim(),
        coordinates: '',
        radius: '',
        confidence: '',
      };
    } else if (currentNodule) {
      if (line.includes('Coordinates')) {
        currentNodule.coordinates = line.split(':')[1].trim();
      } else if (line.includes('Radius')) {
        currentNodule.radius = line.split(':')[1].trim();
      } else if (line.includes('Confidence')) {
        currentNodule.confidence = line.split(':')[1].trim();
      }
    }
  }
  
  // Add the last nodule if present
  if (currentNodule) {
    nodules.push(currentNodule);
  }
  
  return nodules;
};

export default UploadPage; 