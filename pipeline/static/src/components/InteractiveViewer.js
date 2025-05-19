import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { ArrowPathIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';

const InteractiveViewer = ({ caseId, selectedNoduleId, resultsPageView = false, onNoduleClick }) => {
  const [volumeInfo, setVolumeInfo] = useState(null);
  const [currentAxis, setCurrentAxis] = useState('axial');
  const [currentSlice, setCurrentSlice] = useState(0);
  const [sliceImage, setSliceImage] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [nodulesInSlice, setNodulesInSlice] = useState([]);
  const [selectedNoduleState, setSelectedNoduleState] = useState(selectedNoduleId);
  const imageRef = useRef(null);
  const containerRef = useRef(null);
  const { darkMode } = useTheme();

  // Update local state when prop changes
  useEffect(() => {
    setSelectedNoduleState(selectedNoduleId);
    
    // If a nodule is selected from outside (e.g., from nodule table), navigate to it
    if (selectedNoduleId && volumeInfo && volumeInfo.nodules) {
      const nodule = volumeInfo.nodules.find(n => n.id === selectedNoduleId);
      if (nodule) {
        console.log(`External selection of nodule ${selectedNoduleId}, navigating to slice ${nodule.z}`);
        
        // Always switch to axial view for nodule navigation from table
        setCurrentAxis('axial');
        
        // Set the slice to the nodule's z-coordinate (with bounds checking)
        const zSlice = Math.min(Math.max(0, Math.round(nodule.z)), 
                               volumeInfo.dimensions ? (volumeInfo.dimensions.depth - 1) : 100);
        setCurrentSlice(zSlice);
      }
    }
  }, [selectedNoduleId, volumeInfo]);

  // Get the auth token from localStorage
  const getAuthToken = () => {
    return localStorage.getItem('token');
  };

  // Setup axios request with auth headers
  const createAuthorizedRequest = () => {
    const token = getAuthToken();
    return {
      headers: token ? { 'Authorization': `Bearer ${token}` } : {}
    };
  };

  // Function to go to a specific nodule slice
  const goToNoduleSlice = (noduleId) => {
    if (!volumeInfo || !volumeInfo.nodules) {
      console.error('Cannot go to nodule slice: volumeInfo or nodules not available');
      return;
    }
    
    console.log(`InteractiveViewer.goToNoduleSlice called with ID: "${noduleId}"`);
    console.log('Available nodules:', volumeInfo.nodules.map(n => `"${n.id}"`).join(', '));
    
    // Find the selected nodule in the list - try exact match first
    let nodule = volumeInfo.nodules.find(n => n.id === noduleId);
    
    // If no exact match, try looser matching (sometimes IDs get transformed)
    if (!nodule && typeof noduleId === 'string') {
      // Try without spaces
      const normalizedId = noduleId.replace(/\s+/g, '');
      nodule = volumeInfo.nodules.find(n => 
        (typeof n.id === 'string' && n.id.replace(/\s+/g, '') === normalizedId) ||
        String(n.id) === noduleId
      );
      
      if (nodule) {
        console.log(`Found nodule with normalized ID matching: ${nodule.id}`);
      }
    }
    
    if (!nodule) {
      console.error(`Nodule with ID ${noduleId} not found, available nodules:`, volumeInfo.nodules);
      return;
    }
    
    console.log(`Navigating to nodule ${noduleId} at position z=${nodule.z}, y=${nodule.y}, x=${nodule.x}`);
    
    // Update the selected nodule state
    setSelectedNoduleState(noduleId);
    
    // Always switch to axial view for best nodule visualization
    setCurrentAxis('axial');
    
    // Navigate to the Z slice of the nodule (axial view)
    const zSlice = Math.min(Math.max(0, Math.round(nodule.z)), volumeInfo.dimensions.depth - 1);
    setCurrentSlice(zSlice);
    
    console.log(`Set axis to axial and slice to ${zSlice}`);
  };

  // Navigate to selected nodule when selectedNoduleId or volumeInfo changes
  useEffect(() => {
    if (selectedNoduleId && volumeInfo && volumeInfo.nodules) {
      goToNoduleSlice(selectedNoduleId);
    }
  }, [selectedNoduleId, volumeInfo]);

  // Fetch volume metadata
  useEffect(() => {
    const fetchVolumeInfo = async () => {
      try {
        setIsLoading(true);
        
        // Add token to the request
        const token = getAuthToken();
        let url = `/api/results/${caseId}/slices`;
        
        // Add token as query parameter as fallback
        if (token) {
          url += `?token=${token}`;
        }
        
        const response = await axios.get(url, createAuthorizedRequest());
        
        // Check if we received valid volume information
        if (response.data.volume_info) {
          setVolumeInfo(response.data.volume_info);
          
          // Check if we're using placeholder data
          const usingPlaceholder = response.data.using_placeholder || false;
          if (usingPlaceholder) {
            console.warn('Using placeholder data for volume');
          }
          
          // Set initial slice to middle of volume
          const initialSlice = Math.floor(response.data.volume_info.dimensions.depth / 2) || 0;
          setCurrentSlice(initialSlice);
          
          // If we're in a simplified view for the results page and there are nodules,
          // try to show a slice with a nodule if possible
          if (response.data.volume_info.nodules && response.data.volume_info.nodules.length > 0) {
            // Find the slice with the most confident nodule
            const mainNodule = response.data.volume_info.nodules.reduce(
              (prev, current) => (current.confidence > prev.confidence) ? current : prev,
              response.data.volume_info.nodules[0]
            );
            
            // If there's a selected nodule ID and it matches one of our nodules, use that one
            if (selectedNoduleId) {
              const selectedNodule = response.data.volume_info.nodules.find(n => n.id === selectedNoduleId);
              if (selectedNodule) {
                // Ensure we're setting a valid number, not NaN
                const noduleZ = Math.round(selectedNodule.z);
                if (!isNaN(noduleZ) && noduleZ >= 0 && noduleZ < response.data.volume_info.dimensions.depth) {
                  setCurrentSlice(noduleZ);
                  setSelectedNoduleState(selectedNoduleId);
                }
              }
            } 
            // Else if no selected nodule or it wasn't found, navigate to the most confident nodule
            else {
              // Ensure we're setting a valid number, not NaN
              const noduleZ = Math.round(mainNodule.z);
              if (!isNaN(noduleZ) && noduleZ >= 0 && noduleZ < response.data.volume_info.dimensions.depth) {
                setCurrentSlice(noduleZ);
              }
            }
          }
        } else {
          console.warn('No volume information received from server');
          // Set default volume info to prevent errors
          setVolumeInfo({
            dimensions: { depth: 1, height: 512, width: 512 },
            spacing: [1.0, 1.0, 1.0],
            nodules: []
          });
          setCurrentSlice(0);
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching volume info:', error);
        setError(error.response?.data?.message || 'Failed to load volume data');
        // Set default volume info to prevent errors
        setVolumeInfo({
          dimensions: { depth: 1, height: 512, width: 512 },
          spacing: [1.0, 1.0, 1.0],
          nodules: []
        });
        setCurrentSlice(0);
        setIsLoading(false);
      }
    };

    fetchVolumeInfo();
  }, [caseId, resultsPageView]);

  // Fetch slice when axis or slice index changes
  useEffect(() => {
    const fetchSlice = async () => {
      if (!volumeInfo) return;
      
      try {
        setIsLoading(true);
        
        // Add token to the request
        const token = getAuthToken();
        let url = `/api/results/${caseId}/slices?axis=${currentAxis}&index=${currentSlice}`;
        
        // Add token as query parameter as fallback
        if (token) {
          url += `&token=${token}`;
        }
        
        const response = await axios.get(url, createAuthorizedRequest());
        
        // Check if we received a placeholder image
        const isPlaceholder = response.data.is_placeholder || false;
        
        // Get the slice data, handling whether it includes the data:image prefix or not
        let imageData = response.data.slice_data;
        if (!imageData) {
          console.error('No slice data received');
          setError('No image data received from server');
          setIsLoading(false);
          return;
        }
        
        if (!imageData.startsWith('data:image')) {
          imageData = `data:image/png;base64,${imageData}`;
        }
        
        // Update max dimension information if available
        if (response.data.all_dimensions) {
          // Don't update volumeInfo here as it causes a render loop
          // Just store the dimensions for reference
          const allDimensions = {
            axial: response.data.all_dimensions.axial,
            coronal: response.data.all_dimensions.coronal,
            sagittal: response.data.all_dimensions.sagittal
          };
          
          // Only update the current max index if needed - this won't trigger a re-render
          // since we're not updating the volumeInfo state
          if (allDimensions[currentAxis] && currentSlice > allDimensions[currentAxis] - 1) {
            setCurrentSlice(allDimensions[currentAxis] - 1);
          }
        }
        
        // If max_index is provided, ensure we're not exceeding it
        if (response.data.max_index !== undefined && currentSlice > response.data.max_index) {
          setCurrentSlice(response.data.max_index);
        }
        
        setSliceImage({
          data: imageData,
          isPlaceholder: isPlaceholder
        });
        
        // If there's an error message, display it but still show the image
        if (response.data.error || response.data.message) {
          console.warn('Server returned an error or message:', response.data.error || response.data.message);
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching slice:', error);
        setError(error.response?.data?.message || 'Failed to load slice');
        
        // Try to display error image if available
        if (error.response?.data?.slice_data) {
          try {
            let imageData = error.response.data.slice_data;
            if (!imageData.startsWith('data:image')) {
              imageData = `data:image/png;base64,${imageData}`;
            }
            
            setSliceImage({
              data: imageData,
              isPlaceholder: true
            });
          } catch (imgError) {
            console.error('Error processing error image:', imgError);
          }
        }
        
        setIsLoading(false);
      }
    };

    fetchSlice();
  }, [caseId, currentAxis, currentSlice, volumeInfo]);

  // Find nodules in the current slice
  useEffect(() => {
    if (!volumeInfo || !volumeInfo.nodules) return;
    
    // Determine which nodules are visible in this slice
    const sliceNodules = volumeInfo.nodules.filter(nodule => {
      // Check if this nodule is visible in the current slice
      if (currentAxis === 'axial') {
        // Check if the nodule's z-coordinate is close to this slice
        return Math.abs(nodule.z - currentSlice) <= nodule.radius;
      } else if (currentAxis === 'coronal') {
        // Check if the nodule's y-coordinate is close to this slice
        return Math.abs(nodule.y - currentSlice) <= nodule.radius;
      } else { // sagittal
        // Check if the nodule's x-coordinate is close to this slice
        return Math.abs(nodule.x - currentSlice) <= nodule.radius;
      }
    });
    
    setNodulesInSlice(sliceNodules);
  }, [volumeInfo, currentAxis, currentSlice]);

  // Draw nodule circles and center points when image loads or nodulesInSlice changes
  useEffect(() => {
    if (!sliceImage || !imageRef.current || nodulesInSlice.length === 0) return;
    
    // Skip drawing nodules on placeholder images
    if (sliceImage.isPlaceholder) return;

    // Check if we're using MHD format with server-rendered nodules
    const isMhdFormat = caseId && (
      caseId.endsWith('.mhd') || 
      caseId.includes('1.3.6.1.4.1.14519') || // LIDC-IDRI format
      caseId.includes('1.2.826') // DICOM UID format
    );

    const drawNodules = () => {
      // For MHD files with server-rendered nodules, we may not need to draw client-side
      // Client-side drawing might interfere with server-rendered nodules
      if (isMhdFormat) {
        console.log('Using server-rendered nodules for MHD format');
        // We'll still make the nodules clickable, but won't draw overlays
        const img = imageRef.current;
        if (!img.complete) {
          img.onload = setupNoduleInteractions;
          return;
        }
        setupNoduleInteractions();
        return;
      }
      
      // Normal drawing for non-MHD formats
      const img = imageRef.current;
      if (!img.complete) {
        // Wait for image to load before drawing
        img.onload = drawNodulesOnImage;
        return;
      }
      
      drawNodulesOnImage();
    };

    // This function just makes nodules clickable without drawing SVG overlays
    const setupNoduleInteractions = () => {
      const img = imageRef.current;
      const container = containerRef.current;
      
      // Create overlay div for event handling
      const existingOverlay = container.querySelector('.nodule-overlay');
      if (existingOverlay) {
        container.removeChild(existingOverlay);
      }
      
      // Create a minimal overlay just for handling clicks
      const overlay = document.createElement('div');
      overlay.className = 'nodule-overlay';
      overlay.style.position = 'absolute';
      overlay.style.top = '0';
      overlay.style.left = '0';
      overlay.style.width = '100%';
      overlay.style.height = '100%';
      overlay.style.pointerEvents = 'auto';
      
      // Add click handler for the whole image
      overlay.addEventListener('click', (event) => {
        // Get click coordinates relative to the image
        const rect = img.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Normalize to 0-1 range
        const xNorm = x / rect.width;
        const yNorm = y / rect.height;
        
        // Convert to image coordinates
        const imgX = Math.round(xNorm * volumeInfo.dimensions.width);
        const imgY = Math.round(yNorm * volumeInfo.dimensions.height);
        
        // Find closest nodule to click point
        let closestNodule = null;
        let minDistance = Infinity;
        
        nodulesInSlice.forEach(nodule => {
          let nodeX, nodeY;
          
          if (currentAxis === 'axial') {
            nodeX = nodule.x;
            nodeY = nodule.y;
          } else if (currentAxis === 'coronal') {
            nodeX = nodule.x;
            nodeY = nodule.z;
          } else { // sagittal
            nodeX = nodule.y;
            nodeY = nodule.z;
          }
          
          const distance = Math.sqrt(
            Math.pow(nodeX - imgX, 2) + 
            Math.pow(nodeY - imgY, 2)
          );
          
          // Consider a nodule clicked if within 30px of its center
          if (distance < nodule.radius * 2 && distance < minDistance) {
            minDistance = distance;
            closestNodule = nodule;
          }
        });
        
        if (closestNodule) {
          setSelectedNoduleState(closestNodule.id);
          
          if (onNoduleClick) {
            onNoduleClick(closestNodule.id);
          } else {
            goToNoduleSlice(closestNodule.id);
          }
        }
      });
      
      container.appendChild(overlay);
    };

    // Original function for drawing nodules with SVG overlays
    const drawNodulesOnImage = () => {
      const img = imageRef.current;
      const container = containerRef.current;
      
      // Clear any existing overlay elements
      const existingOverlay = container.querySelector('.nodule-overlay');
      if (existingOverlay) {
        container.removeChild(existingOverlay);
      }
      
      // Create overlay div that will contain the SVG with nodule circles
      const overlay = document.createElement('div');
      overlay.className = 'nodule-overlay';
      overlay.style.position = 'absolute';
      overlay.style.top = '0';
      overlay.style.left = '0';
      overlay.style.width = '100%';
      overlay.style.height = '100%';
      overlay.style.pointerEvents = 'auto';
      
      // Calculate image display dimensions and position
      const imgRect = img.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      
      // Calculate aspect ratio for the current view
      let aspectRatio = 1.0;
      if (volumeInfo && volumeInfo.spacing) {
        if (currentAxis === 'axial') {
          // Y/X aspect ratio for axial view
          aspectRatio = volumeInfo.spacing[1] / volumeInfo.spacing[0];
        } else if (currentAxis === 'coronal') {
          // Z/X aspect ratio for coronal view
          aspectRatio = volumeInfo.spacing[2] / volumeInfo.spacing[0];
        } else { // sagittal
          // Z/Y aspect ratio for sagittal view
          aspectRatio = volumeInfo.spacing[2] / volumeInfo.spacing[1];
        }
      }
      
      // Create SVG element for drawing
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');
      svg.style.position = 'absolute';
      svg.style.top = '0';
      svg.style.left = '0';
      
      // Get display dimensions for scaling
      const displayWidth = imgRect.width;
      const displayHeight = imgRect.height;
      
      // Draw each nodule
      nodulesInSlice.forEach(nodule => {
        let x, y, slicePos;
        
        // Calculate 2D coordinates based on the current viewing axis
        if (currentAxis === 'axial') {
          x = nodule.x;
          y = nodule.y;
          slicePos = currentSlice;
        } else if (currentAxis === 'coronal') {
          x = nodule.x;
          y = nodule.z;
          slicePos = currentSlice;
        } else { // sagittal
          x = nodule.y;
          y = nodule.z;
          slicePos = currentSlice;
        }
        
        // Calculate the position in normalized coordinates (0-1 range)
        let xNorm, yNorm;
        
        if (currentAxis === 'axial') {
          xNorm = x / volumeInfo.dimensions.width;
          yNorm = y / volumeInfo.dimensions.height;
        } else if (currentAxis === 'coronal') {
          xNorm = x / volumeInfo.dimensions.width;
          yNorm = y / volumeInfo.dimensions.depth;
        } else { // sagittal
          xNorm = y / volumeInfo.dimensions.height;
          yNorm = nodule.z / volumeInfo.dimensions.depth;
        }
        
        // Apply proper scaling for the current view resolution
        // Convert from normalized coordinates to display pixels
        const xPx = xNorm * displayWidth;
        
        // Apply aspect ratio correction to y-coordinate
        // This ensures that the nodule appears at the right vertical position
        // regardless of how the image is stretched or squeezed
        const yPx = yNorm * displayHeight;
        
        // Calculate radius in pixels based on the nodule's radius in mm
        // Scale according to the current display size
        let radiusMm = nodule.radius || 5; // Default to 5mm if not specified
        let pixelsPerMm;
        
        // Calculate pixels per mm based on the current view's spacing
        if (currentAxis === 'axial') {
          // Use the minimum of width and height spacing for consistent sizing
          const spacingX = volumeInfo.spacing ? volumeInfo.spacing[0] : 1;
          const spacingY = volumeInfo.spacing ? volumeInfo.spacing[1] : 1;
          const mmPerPixel = Math.min(spacingX, spacingY);
          // Calculate how many display pixels correspond to one mm in the CT scan
          const widthInMm = volumeInfo.dimensions.width * spacingX;
          pixelsPerMm = displayWidth / widthInMm;
        } else if (currentAxis === 'coronal') {
          const spacingX = volumeInfo.spacing ? volumeInfo.spacing[0] : 1;
          const spacingZ = volumeInfo.spacing ? volumeInfo.spacing[2] : 1;
          const mmPerPixel = Math.min(spacingX, spacingZ);
          // Calculate how many display pixels correspond to one mm in the CT scan
          const widthInMm = volumeInfo.dimensions.width * spacingX;
          pixelsPerMm = displayWidth / widthInMm;
        } else { // sagittal
          const spacingY = volumeInfo.spacing ? volumeInfo.spacing[1] : 1;
          const spacingZ = volumeInfo.spacing ? volumeInfo.spacing[2] : 1;
          const mmPerPixel = Math.min(spacingY, spacingZ);
          // Calculate how many display pixels correspond to one mm in the CT scan
          const heightInMm = volumeInfo.dimensions.height * spacingY;
          pixelsPerMm = displayHeight / heightInMm;
        }
        
        // Calculate the radius in display pixels
        let radiusPixels = radiusMm * pixelsPerMm;
        
        // Make circles more appropriately sized for visualization
        radiusPixels = Math.max(8, radiusPixels * 0.75); // Increase from 0.5 to 0.75 multiplier with larger minimum radius
        
        // Adjust the radius based on how far the nodule is from the current slice
        let distanceFromSlice;
        if (currentAxis === 'axial') {
          distanceFromSlice = Math.abs(nodule.z - slicePos);
        } else if (currentAxis === 'coronal') {
          distanceFromSlice = Math.abs(nodule.y - slicePos);
        } else { // sagittal
          distanceFromSlice = Math.abs(nodule.x - slicePos);
        }
        
        // Calculate the nodule color based on confidence
        // Use a lighter red for better visibility
        const color = '#FF5555'; // Lighter red color for all nodules
        
        // Create circle element for nodule outline
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', xPx);
        circle.setAttribute('cy', yPx);
        circle.setAttribute('r', radiusPixels);
        circle.setAttribute('fill', 'rgba(255, 150, 150, 0.15)'); // Lighter red fill with lower opacity
        circle.setAttribute('stroke', color);
        circle.setAttribute('stroke-width', '1.5'); // Thinner stroke
        circle.setAttribute('data-nodule-id', nodule.id); // Add nodule ID as data attribute
        
        // Make nodules interactive - turn off pointer-events: none on the overlay
        overlay.style.pointerEvents = 'auto';
        
        // Create dot element for center point
        const centerDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        centerDot.setAttribute('cx', xPx);
        centerDot.setAttribute('cy', yPx);
        centerDot.setAttribute('r', '3');
        centerDot.setAttribute('fill', color);
        centerDot.setAttribute('data-nodule-id', nodule.id); // Add nodule ID to center dot too
        
        // Highlight selected nodule with a thicker stroke
        if (selectedNoduleState === nodule.id) {
            circle.setAttribute('stroke-width', '3');  // Still thicker for selected but reduced
            circle.setAttribute('stroke-dasharray', '5,3');
            circle.setAttribute('fill', 'rgba(255, 150, 150, 0.35)'); // Lighter red fill for selected nodule
            centerDot.setAttribute('r', '4'); // Slightly smaller center dot for selected nodule
        }
        
        // Add click event listeners to circle and centerDot
        const handleClick = (event) => {
            event.stopPropagation(); // Prevent click from propagating to container
            const noduleId = event.target.getAttribute('data-nodule-id');
            if (noduleId) {
                // Set selected nodule state locally
                setSelectedNoduleState(noduleId);
                
                // Call the parent's handler if provided
                if (onNoduleClick) {
                    onNoduleClick(noduleId);
                } else {
                    // Otherwise use our own navigation
                    goToNoduleSlice(noduleId);
                }
                
                // Redraw nodules to update selection highlighting
                setTimeout(() => drawNodulesOnImage(), 0);
            }
        };
        
        // Add event listeners
        circle.addEventListener('click', handleClick);
        centerDot.addEventListener('click', handleClick);
        
        // Add tooltip with nodule info
        const tooltip = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        const confidence = nodule.confidence ? (nodule.confidence * 100).toFixed(0) + '%' : 'N/A';
        const malignancy = nodule.malignancy || 'Unknown';
        tooltip.textContent = `Nodule ${nodule.id}\nConfidence: ${confidence}\nMalignancy: ${malignancy}`;
        circle.appendChild(tooltip);
        
        // Add mouseover/mouseout effects for better interaction
        circle.addEventListener('mouseover', () => {
            circle.setAttribute('stroke-width', '2.5');
            circle.setAttribute('fill', 'rgba(255, 150, 150, 0.25)');
            centerDot.setAttribute('r', '4');
        });
        
        circle.addEventListener('mouseout', () => {
            if (selectedNoduleState === nodule.id) {
                circle.setAttribute('stroke-width', '3');
                circle.setAttribute('fill', 'rgba(255, 150, 150, 0.35)');
                centerDot.setAttribute('r', '4');
            } else {
                circle.setAttribute('stroke-width', '1.5');
                circle.setAttribute('fill', 'rgba(255, 150, 150, 0.15)');
                centerDot.setAttribute('r', '3');
            }
        });
        
        centerDot.addEventListener('mouseover', () => {
            circle.setAttribute('stroke-width', '2.5');
            circle.setAttribute('fill', 'rgba(255, 150, 150, 0.25)');
            centerDot.setAttribute('r', '4');
        });
        
        centerDot.addEventListener('mouseout', () => {
            if (selectedNoduleState === nodule.id) {
                circle.setAttribute('stroke-width', '3');
                circle.setAttribute('fill', 'rgba(255, 150, 150, 0.35)');
                centerDot.setAttribute('r', '4');
            } else {
                circle.setAttribute('stroke-width', '1.5');
                circle.setAttribute('fill', 'rgba(255, 150, 150, 0.15)');
                centerDot.setAttribute('r', '3');
            }
        });
        
        svg.appendChild(circle);
        svg.appendChild(centerDot);
      });
      
      overlay.appendChild(svg);
      container.appendChild(overlay);
    };

    drawNodules();
    
    // Cleanup function
    return () => {
      const container = containerRef.current;
      if (container) {
        const overlay = container.querySelector('.nodule-overlay');
        if (overlay) {
          container.removeChild(overlay);
        }
      }
    };
  }, [sliceImage, nodulesInSlice, currentAxis, selectedNoduleState, volumeInfo, resultsPageView]);

  // Get max slice index for current axis
  const getMaxSliceIndex = () => {
    if (!volumeInfo) return 0;
    
    try {
      if (currentAxis === 'axial') {
        return Math.max(0, volumeInfo.dimensions.depth - 1);
      } else if (currentAxis === 'coronal') {
        return Math.max(0, volumeInfo.dimensions.height - 1);
      } else { // sagittal
        return Math.max(0, volumeInfo.dimensions.width - 1);
      }
    } catch (error) {
      console.error('Error determining max slice index:', error);
      return 0;
    }
  };

  const handleAxisChange = (axis) => {
    setCurrentAxis(axis);
    // Reset to middle slice when changing axis
    if (!volumeInfo) return;
    
    let middle;
    if (axis === 'axial') {
      middle = Math.floor(volumeInfo.dimensions.depth / 2);
    } else if (axis === 'coronal') {
      middle = Math.floor(volumeInfo.dimensions.height / 2);
    } else { // sagittal
      middle = Math.floor(volumeInfo.dimensions.width / 2);
    }
    
    // Ensure we're setting a valid number, not NaN
    if (!isNaN(middle) && middle >= 0) {
      setCurrentSlice(middle);
    }
  };

  const handleSliceChange = (e) => {
    const value = parseInt(e.target.value, 10);
    // Validate before setting
    if (!isNaN(value) && value >= 0 && value <= getMaxSliceIndex()) {
      setCurrentSlice(value);
    }
  };

  // Track which slices have nodules to highlight them on the slider
  const slicesWithNodules = () => {
    if (!volumeInfo || !volumeInfo.nodules) return [];
    
    // Don't show nodule indicators on the slider in results page view
    if (resultsPageView) return [];
    
    const noduleSlices = new Set();
    volumeInfo.nodules.forEach(nodule => {
      if (currentAxis === 'axial') {
        // Mark all slices within the nodule's radius
        const start = Math.max(0, Math.floor(nodule.z - nodule.radius));
        const end = Math.min(volumeInfo.dimensions.depth - 1, Math.ceil(nodule.z + nodule.radius));
        for (let i = start; i <= end; i++) {
          noduleSlices.add(i);
        }
      } else if (currentAxis === 'coronal') {
        const start = Math.max(0, Math.floor(nodule.y - nodule.radius));
        const end = Math.min(volumeInfo.dimensions.height - 1, Math.ceil(nodule.y + nodule.radius));
        for (let i = start; i <= end; i++) {
          noduleSlices.add(i);
        }
      } else { // sagittal
        const start = Math.max(0, Math.floor(nodule.x - nodule.radius));
        const end = Math.min(volumeInfo.dimensions.width - 1, Math.ceil(nodule.x + nodule.radius));
        for (let i = start; i <= end; i++) {
          noduleSlices.add(i);
        }
      }
    });
    
    return Array.from(noduleSlices);
  };

  // Nodule colors for different confidence levels
  const getNoduleColor = (confidence) => {
    if (confidence >= 0.8) return 'rgba(220, 38, 38, 0.8)';  // Red for high confidence
    if (confidence >= 0.6) return 'rgba(234, 88, 12, 0.8)';  // Orange for medium confidence
    return 'rgba(234, 179, 8, 0.8)';  // Yellow for lower confidence
  };

  return (
    <div className={`relative w-full h-full ${resultsPageView ? 'results-page-view' : ''}`} ref={containerRef}>
      {/* If in results page mode, only show the image without controls */}
      {resultsPageView ? (
        <>
          {isLoading ? (
            <div className="flex items-center justify-center w-full h-full bg-gray-200">
              <ArrowPathIcon className="h-8 w-8 animate-spin text-gray-400" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center w-full h-full bg-gray-200">
              <div className="text-center p-4">
                <ExclamationCircleIcon className="h-8 w-8 mx-auto mb-2 text-red-500" />
                <p className="text-sm text-gray-600">Error loading scan</p>
              </div>
            </div>
          ) : sliceImage ? (
            <div className="w-full h-full overflow-hidden relative">
              <div className="flex items-center justify-center h-full">
                <img 
                  ref={imageRef}
                  src={sliceImage.data} 
                  alt={`Axial view of scan ${caseId}`}
                  className="object-contain max-h-full max-w-full"
                  style={{ 
                    objectPosition: 'center center',
                    maxHeight: '260px'
                  }}
                />
              </div>
              {sliceImage.isPlaceholder && (
                <div className="absolute bottom-0 left-0 right-0 bg-red-500 bg-opacity-70 text-white text-center py-1 text-sm">
                  Placeholder Image - Scan Data Unavailable
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center w-full h-full bg-gray-200">
              <p className="text-sm text-gray-500">No image available</p>
            </div>
          )}
        </>
      ) : (
        // Original interactive viewer with all controls for the detail page
        <>
          {/* Axis selection */}
          <div className={`absolute top-2 left-2 z-10 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <div className="flex p-1 space-x-1">
              <button
                onClick={() => handleAxisChange('axial')}
                className={`px-3 py-1.5 text-xs font-medium rounded-md ${
                  currentAxis === 'axial' 
                    ? darkMode
                      ? 'bg-indigo-600 text-white'
                      : 'bg-indigo-100 text-indigo-700'
                    : darkMode
                      ? 'text-gray-300 hover:bg-gray-700'
                      : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Axial
              </button>
              <button
                onClick={() => handleAxisChange('coronal')}
                className={`px-3 py-1.5 text-xs font-medium rounded-md ${
                  currentAxis === 'coronal' 
                    ? darkMode
                      ? 'bg-indigo-600 text-white'
                      : 'bg-indigo-100 text-indigo-700'
                    : darkMode
                      ? 'text-gray-300 hover:bg-gray-700'
                      : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Coronal
              </button>
              <button
                onClick={() => handleAxisChange('sagittal')}
                className={`px-3 py-1.5 text-xs font-medium rounded-md ${
                  currentAxis === 'sagittal' 
                    ? darkMode
                      ? 'bg-indigo-600 text-white'
                      : 'bg-indigo-100 text-indigo-700'
                    : darkMode
                      ? 'text-gray-300 hover:bg-gray-700'
                      : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Sagittal
              </button>
            </div>
          </div>
          
          {/* Slice slider */}
          <div className={`absolute bottom-2 left-1/2 transform -translate-x-1/2 z-10 p-2 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`} style={{ width: '90%', maxWidth: '400px' }}>
            <input 
              type="range"
              min="0"
              max={getMaxSliceIndex()}
              value={currentSlice}
              onChange={handleSliceChange}
              className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            />
            <div className="flex justify-between text-xs mt-1">
              <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>Slice: {currentSlice + 1}/{getMaxSliceIndex() + 1}</span>
              <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>
                {slicesWithNodules().includes(currentSlice) ? '🔴 Nodule present' : ''}
                {sliceImage && sliceImage.isPlaceholder ? '⚠️ Placeholder' : ''}
              </span>
            </div>
          </div>
          
          {/* Main image display */}
          <div className="w-full h-full relative">
            {sliceImage ? (
              <div className="relative w-full h-full">
                <img 
                  ref={imageRef}
                  src={sliceImage.data} 
                  alt={`${currentAxis.charAt(0).toUpperCase() + currentAxis.slice(1)} slice of CT scan`}
                  className="w-full h-full object-contain"
                />
                {sliceImage.isPlaceholder && (
                  <div className="absolute bottom-0 left-0 right-0 bg-red-500 bg-opacity-70 text-white text-center py-1 text-sm">
                    Placeholder Image - Scan Data Unavailable
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center w-full h-full bg-gray-200">
                <p className="text-sm text-gray-500">No image available</p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default InteractiveViewer; 