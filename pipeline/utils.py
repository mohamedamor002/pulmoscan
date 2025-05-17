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