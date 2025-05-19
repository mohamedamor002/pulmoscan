import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_placeholder_slice(width=512, height=512):
    """
    Generate a placeholder CT scan slice when actual scan data is unavailable.
    Returns base64 encoded PNG image.
    """
    # Create a background similar to CT scan
    slice_data = np.ones((height, width), dtype=np.uint8) * 30
    
    # Add some basic structures to simulate a CT scan
    # Create circular outline (body)
    y, x = np.ogrid[-height//2:height//2, -width//2:width//2]
    body_mask = x**2 + y**2 <= (min(width, height)//2 - 10)**2
    slice_data[body_mask] = 90
    
    # Create two circular "lung" areas
    lung_radius = min(width, height) // 6
    lung_y_pos = 0
    
    # Left lung
    left_x_pos = -width // 4
    left_lung_mask = (x - left_x_pos)**2 + (y - lung_y_pos)**2 <= lung_radius**2
    slice_data[left_lung_mask] = 10
    
    # Right lung
    right_x_pos = width // 4
    right_lung_mask = (x - right_x_pos)**2 + (y - lung_y_pos)**2 <= lung_radius**2
    slice_data[right_lung_mask] = 10
    
    # Add spine-like structure
    spine_width = min(width, height) // 10
    spine_y_pos = height // 4
    spine_mask = (x**2 <= (spine_width//2)**2) & (y > spine_y_pos - spine_width//2)
    slice_data[spine_mask] = 180
    
    # Add a "nodule" in one lung (50% of the time in left lung, 50% in right)
    nodule_radius = lung_radius // 5
    if np.random.random() > 0.5:
        # Left lung nodule
        nodule_x = left_x_pos + np.random.randint(-lung_radius//2, lung_radius//2)
        nodule_y = lung_y_pos + np.random.randint(-lung_radius//2, lung_radius//2)
    else:
        # Right lung nodule
        nodule_x = right_x_pos + np.random.randint(-lung_radius//2, lung_radius//2)
        nodule_y = lung_y_pos + np.random.randint(-lung_radius//2, lung_radius//2)
    
    nodule_mask = (x - nodule_x)**2 + (y - nodule_y)**2 <= nodule_radius**2
    slice_data[nodule_mask] = 110
    
    # Add subtle noise to make it look more natural
    noise = np.random.normal(0, 5, slice_data.shape).astype(np.int8)
    slice_data = np.clip(slice_data + noise, 0, 255).astype(np.uint8)
    
    # Generate image
    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Add a watermark text indicating this is a placeholder
    plt.text(width//2, height - 20, "PLACEHOLDER IMAGE - NO SCAN DATA", 
             horizontalalignment='center', color='red', fontsize=12)
    
    # Save to buffer and encode
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8') 

def adaptive_window_level(slice_data, default_window=1500, default_level=-600):
    """
    Apply adaptive window/level based on histogram analysis of the slice.
    
    Args:
        slice_data: 2D numpy array of the CT slice
        default_window: Default window width if histogram analysis fails
        default_level: Default window center if histogram analysis fails
        
    Returns:
        tuple of (window, level) values to use for display
    """
    # Get histogram to analyze the intensity distribution
    try:
        hist, bin_edges = np.histogram(slice_data, bins=100)
        
        # Find the peaks in the histogram (typically air, soft tissue, and bone)
        peak_indices = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
        peak_values = bin_edges[peak_indices]
        
        if len(peak_values) >= 2:
            # If we have at least two peaks, use them to determine window/level
            
            # Sort peaks by intensity
            sorted_peaks = np.sort(peak_values)
            
            # Find lung tissue peak (typically around -700 to -500 HU)
            lung_peak_candidates = [p for p in sorted_peaks if -800 < p < -400]
            
            if lung_peak_candidates:
                # If we found lung peaks, set window/level to enhance that region
                lung_peak = lung_peak_candidates[0]
                
                # Set window width to cover lung tissue range plus some margin
                window = 1600  # Wider than standard lung window for better visibility
                level = lung_peak + 100  # Slightly higher than the lung peak for better contrast
            else:
                # Fallback to standard lung window
                window = default_window
                level = default_level
        else:
            # Not enough peaks found, use default values
            window = default_window
            level = default_level
    except Exception:
        # In case of any errors during histogram analysis, use default values
        window = default_window
        level = default_level
        
    return window, level

def enhance_contrast(image, low_percentile=5, high_percentile=95):
    """
    Enhance contrast using percentile-based normalization.
    
    Args:
        image: Input image as numpy array
        low_percentile: Lower percentile for contrast stretching (default: 5)
        high_percentile: Upper percentile for contrast stretching (default: 95)
        
    Returns:
        Enhanced image as uint8 numpy array
    """
    try:
        # Get percentile values
        low = np.percentile(image, low_percentile)
        high = np.percentile(image, high_percentile)
        
        # Apply contrast stretching
        enhanced = np.clip(image, low, high)
        enhanced = ((enhanced - low) / (high - low) * 255).astype(np.uint8)
        
        return enhanced
    except Exception:
        # In case of any errors, return the original image converted to uint8
        if image.dtype != np.uint8:
            # Try to convert to uint8 range
            if image.max() > 1.0:
                # Assume image is already in 0-255 range
                return np.clip(image, 0, 255).astype(np.uint8)
            else:
                # Assume image is in 0-1 range
                return (image * 255).astype(np.uint8)
        return image 