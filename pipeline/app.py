import os
import logging
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, decode_token
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import datetime
import json
import sys
import base64
from pathlib import Path
import traceback
from io import BytesIO
import shutil
import time
import re
import uuid

# Set matplotlib backend to non-interactive 'Agg' to avoid "main thread is not in main loop" errors
import matplotlib
matplotlib.use('Agg')

# Try to import nibabel for NIFTI file support
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("nibabel not available. NIFTI files will not be supported.")

# Add the parent and root directories to the path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\Segmentation\pipeline
root_dir = os.path.dirname(parent_dir)  # D:\Segmentation

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

print(f"Added parent directory to path: {parent_dir}")
print(f"Added root directory to path: {root_dir}")

# First, check if required Python packages are available
try:
    import numpy as np
    import torch
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    print("Required packages are available")
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install the required packages using: pip install -r requirements.txt")
    sys.exit(1)

# First check if combined_pipeline.py exists in the root directory
root_pipeline_path = os.path.join(root_dir, 'combined_pipeline.py')
if os.path.exists(root_pipeline_path):
    # Add the root directory to the beginning of the path
    if root_dir in sys.path:
        sys.path.remove(root_dir)
    sys.path.insert(0, root_dir)
    print(f"Found combined_pipeline.py in root directory, importing from: {root_pipeline_path}")
    
    try:
        # Import directly from root
        from combined_pipeline import (
            process_ct_scan, 
            load_lungs_model, 
            load_nodules_model,
            load_malignant_benign_model
        )
        print("Successfully imported pipeline module from root directory")
    except ImportError as e:
        print(f"Error importing from root directory: {e}")
        # If import from root fails, try other methods
        combined_pipeline_found = False
else:
    # File not found in root, check pipeline directory
    pipeline_path = os.path.join(parent_dir, 'combined_pipeline.py')
    if os.path.exists(pipeline_path):
        print(f"Found combined_pipeline.py in pipeline directory, importing from: {pipeline_path}")
        # Add the pipeline directory to the beginning of the path
        if parent_dir in sys.path:
            sys.path.remove(parent_dir)
        sys.path.insert(0, parent_dir)
        
        try:
            # Import directly from pipeline
            from combined_pipeline import (
                process_ct_scan, 
                load_lungs_model, 
                load_nodules_model,
                load_malignant_benign_model
            )
            print("Successfully imported pipeline module from pipeline directory")
            combined_pipeline_found = True
        except ImportError as e:
            print(f"Error importing from pipeline directory: {e}")
            combined_pipeline_found = False
    else:
        print("combined_pipeline.py not found in expected locations")
        combined_pipeline_found = False

# If importing from expected locations failed, search for the file
if not combined_pipeline_found:
    print("Searching for combined_pipeline.py in any location...")
    found = False
    for search_root, dirs, files in os.walk(root_dir):
        if 'combined_pipeline.py' in files:
            found_path = os.path.join(search_root, 'combined_pipeline.py')
            print(f"Found pipeline file at: {found_path}")
            
            # Add the directory to path
            found_dir = os.path.dirname(found_path)
            if found_dir not in sys.path:
                sys.path.insert(0, found_dir)
            
            try:
                # Dynamically load the module
                import importlib.util
                spec = importlib.util.spec_from_file_location("combined_pipeline", found_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                process_ct_scan = module.process_ct_scan
                load_lungs_model = module.load_lungs_model
                load_nodules_model = module.load_nodules_model
                load_malignant_benign_model = module.load_malignant_benign_model
                
                print("Successfully loaded pipeline module from discovered path")
                found = True
                break
            except ImportError as e:
                print(f"Failed to import from discovered path: {e}")
    
    if not found:
        print("Could not find combined_pipeline.py anywhere in the project")
        print("Please ensure the file exists and is properly structured")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder=os.path.join('static'),
            template_folder=os.path.join('templates'))

# Enable CORS
CORS(app)

# Configure app
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['ALLOWED_EXTENSIONS'] = {'mhd', 'raw', 'nii.gz', 'nii', 'dcm'}
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300 MB limit
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key-for-development-only')  # Change in production!
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=2)  # Extend token lifetime to 2 hours

# Setup JWT
jwt = JWTManager(app)

# Create upload and results directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Mock user database (replace with a real database in production)
users_db = {
    'Pulmoscan': {
        'password': generate_password_hash('Pulmoscan123'),
        'role': 'superadmin',
        'first_name': '',
        'last_name': '',
        'email': 'admin@pulmoscan.com',
        'description': 'Superadmin account',
        'plan': 'subscription',
        'created_at': datetime.datetime.now().isoformat()
    },
    'admin': {
        'password': generate_password_hash('admin123'),
        'role': 'admin',
        'first_name': '',
        'last_name': '',
        'email': '',
        'description': 'Admin account',
        'plan': 'subscription',
        'created_at': datetime.datetime.now().isoformat()
    },
    'doctor': {
        'password': generate_password_hash('doctor123'),
        'role': 'doctor',
        'first_name': '',
        'last_name': '',
        'email': '',
        'description': 'Doctor account',
        'plan': 'usage_based',
        'created_at': datetime.datetime.now().isoformat()
    }
}

# Access control database for results (in production, this would be stored in a database)
access_db = {}  # Format: { 'case_name': ['username1', 'username2', ...] }

# Models
lungs_model = None
nodules_model = None
nodules_seg_model = None
malignant_benign_model = None

def allowed_file(filename):
    """Check if the file has an allowed extension, with special handling for files with dots in the filename."""
    if not filename:
        return False
        
    # For MHD/RAW files, check if the file ends with the extension
    for ext in ['.mhd', '.raw', '.nii.gz', '.nii', '.dcm']:
        if filename.lower().endswith(ext):
            return True
    
    # Fallback to the original method
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in app.config['ALLOWED_EXTENSIONS']

def load_models():
    """Load ML models."""
    global lungs_model, nodules_model, nodules_seg_model, malignant_benign_model
    
    # Check if the required functions are available
    if load_lungs_model is None:
        error_msg = "Missing required function: load_lungs_model"
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    # Clear GPU memory before loading models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory before loading models")
    
    logger.info("Loading models...")
    try:
        lungs_model = load_lungs_model()
        if lungs_model is not None:
            # Ensure model is in evaluation mode
            lungs_model.eval()
            logger.info("Lungs model loaded successfully and set to evaluation mode")
        else:
            logger.error("Failed to load lungs model")
    except Exception as e:
        logger.error(f"Error loading lungs model: {e}")
    
    try:
        nodules_model = load_nodules_model()
        if nodules_model is not None:
            # Ensure model is in evaluation mode
            nodules_model.eval()
            logger.info(f"Nodule detection model loaded successfully with {sum(p.numel() for p in nodules_model.parameters())} parameters")
        else:
            logger.error("Failed to load nodules model")
    except Exception as e:
        logger.error(f"Error loading nodules model: {e}")
        
    # Nodule segmentation functionality has been removed
    logger.info("Nodule segmentation functionality has been removed from the pipeline")
    nodules_seg_model = None
    
    # Load malignant/benign classification model
    try:
        malignant_benign_model = load_malignant_benign_model()
        if malignant_benign_model is not None:
            # Ensure model is in evaluation mode
            malignant_benign_model.eval()
            logger.info("Malignant/benign classification model loaded successfully")
        else:
            logger.error("Failed to load malignant/benign classification model")
    except Exception as e:
        logger.error(f"Error loading malignant/benign classification model: {e}")
        malignant_benign_model = None

def save_file_to_unified_location(file_path, job_id, user=None, move=False):
    """
    Saves a file to a unified location in the results directory.
    
    Args:
        file_path: Path to the file to save
        job_id: Job ID to use for the output directory
        user: Username (optional)
        move: Whether to move the file instead of copying it
        
    Returns:
        Path to the saved file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    # Create job-specific directory in results folder
    job_output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define destination path
    dest_file = job_output_dir / file_path.name
    
    # Copy or move the file
    try:
        import shutil
        if move:
            shutil.move(str(file_path), str(dest_file))
            logger.info(f"Moved file to: {dest_file}")
        else:
            shutil.copy2(str(file_path), str(dest_file))
            logger.info(f"Copied file to: {dest_file}")
            
        # For MHD files, also copy/move the corresponding RAW file
        if file_path.suffix.lower() == '.mhd':
            raw_file = file_path.with_suffix('.raw')
            if raw_file.exists():
                dest_raw = dest_file.with_suffix('.raw')
                if move:
                    shutil.move(str(raw_file), str(dest_raw))
                    logger.info(f"Moved RAW file to: {dest_raw}")
                else:
                    shutil.copy2(str(raw_file), str(dest_raw))
                    logger.info(f"Copied RAW file to: {dest_raw}")
        
        return dest_file
    except Exception as e:
        logger.error(f"Error copying/moving file: {e}")
        return None

# Authentication routes
@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login and get access token."""
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    if not username or not password:
        return jsonify({"msg": "Missing username or password"}), 400
    
    if username not in users_db:
        return jsonify({"msg": "Invalid username or password"}), 401
        
    if check_password_hash(users_db[username]['password'], password):
        # Identity can be any data that is json serializable
        user_claims = {
            'username': username,
            'role': users_db[username]['role']
        }
        access_token = create_access_token(identity=username, additional_claims=user_claims)
        
        # Return user data along with the token
        return jsonify({
            'access_token': access_token,
            'username': username,
            'role': users_db[username]['role'],
            'first_name': users_db[username].get('first_name', ''),
            'last_name': users_db[username].get('last_name', ''),
            'email': users_db[username].get('email', ''),
            'description': users_db[username].get('description', ''),
            'plan': users_db[username].get('plan', 'usage_based')
        }), 200
    else:
        return jsonify({"msg": "Invalid username or password"}), 401

@app.route('/api/auth/protected', methods=['GET'])
@jwt_required()
def protected():
    """Test endpoint for JWT protection."""
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200

@app.route('/api/auth/token', methods=['GET'])
@jwt_required()
def get_auth_token():
    """Get the current user's JWT token for use in direct image URLs."""
    try:
        # Get the JWT token from the request
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No valid token found'}), 400
            
        return jsonify({
            'token': token,
            'username': get_jwt_identity()
        }), 200
    except Exception as e:
        logger.error(f"Error getting token: {e}")
        return jsonify({'error': 'An error occurred retrieving the token'}), 500

# File upload and processing routes
@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_file():
    """Upload a CT scan file for processing."""
    try:
        # Get current user identity for tracking
        current_user = get_jwt_identity()
        logger.info(f"Starting file upload process for user: {current_user}")
        
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'error': 'No file part in the request'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            # Generate a user-specific prefix for the file name
            user_prefix = f"{current_user}_"
            
            # Add the prefix to the original filename to associate with this user
            original_filename = secure_filename(file.filename)
            filename = secure_filename(f"{user_prefix}{original_filename}")
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving primary file: {filename} for user {current_user}")
            file.save(filepath)
            
            # Process MHD and RAW files specially
            raw_filepath = None
            if original_filename.lower().endswith('.mhd') and 'raw_file' in request.files:
                raw_file = request.files['raw_file']
                if raw_file.filename != '':
                    original_raw_filename = secure_filename(raw_file.filename)
                    raw_filename = secure_filename(f"{user_prefix}{original_raw_filename}")
                    raw_filepath = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
                    logger.info(f"Saving associated RAW file: {raw_filename} for user {current_user}")
                    raw_file.save(raw_filepath)
                    
                    # Update the MHD file to reference the RAW file with the user prefix
                    try:
                        logger.info(f"Updating MHD file to reference RAW file: {raw_filename}")
                        with open(filepath, 'r') as f:
                            mhd_content = f.readlines()
                        
                        # Find and update the ElementDataFile line
                        for i, line in enumerate(mhd_content):
                            if line.startswith('ElementDataFile'):
                                # Replace with the new RAW filename (with user prefix)
                                mhd_content[i] = f"ElementDataFile = {raw_filename}\n"
                                break
                        
                        # Write the updated MHD file
                        with open(filepath, 'w') as f:
                            f.writelines(mhd_content)
                            
                        logger.info(f"Successfully updated MHD file references")
                    except Exception as e:
                        logger.error(f"Error updating MHD file: {e}")
                        return jsonify({'error': f'Error processing MHD file: {str(e)}'}), 500
                else:
                    logger.error("RAW file was provided but has no filename")
                    return jsonify({'error': 'Invalid RAW file'}), 400
            
            # Verify that MHD files have their RAW files
            if original_filename.lower().endswith('.mhd'):
                if raw_filepath is None:
                    # Check if RAW file exists with same base name
                    base_name = os.path.splitext(filename)[0]
                    possible_raw = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}.raw")
                    if not os.path.exists(possible_raw):
                        # Check if RAW file exists with the original name (without user prefix)
                        orig_base_name = os.path.splitext(original_filename)[0]
                        possible_raw_orig = os.path.join(app.config['UPLOAD_FOLDER'], f"{orig_base_name}.raw")
                        
                        if not os.path.exists(possible_raw_orig):
                            logger.error(f"MHD file uploaded without associated RAW file: {filename}")
                            return jsonify({
                                'error': 'MHD file requires an associated RAW file',
                                'details': 'Please use the dedicated MHD/RAW upload feature to upload both files together, or ensure the RAW file has the same base name as the MHD file.'
                            }), 400
                        else:
                            # RAW file exists with original name, rename it
                            logger.info(f"Found RAW file with original name: {possible_raw_orig}")
                            raw_filename = f"{base_name}.raw"
                            possible_raw = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
                            os.rename(possible_raw_orig, possible_raw)
                            raw_filepath = possible_raw
                            
                            # Update the MHD file to reference the RAW file with the user prefix
                            try:
                                logger.info(f"Updating MHD file to reference RAW file: {raw_filename}")
                                with open(filepath, 'r') as f:
                                    mhd_content = f.readlines()
                                
                                # Find and update the ElementDataFile line
                                for i, line in enumerate(mhd_content):
                                    if line.startswith('ElementDataFile'):
                                        # Replace with the new RAW filename (with user prefix)
                                        mhd_content[i] = f"ElementDataFile = {raw_filename}\n"
                                        break
                                
                                # Write the updated MHD file
                                with open(filepath, 'w') as f:
                                    f.writelines(mhd_content)
                                    
                                logger.info(f"Successfully updated MHD file references")
                            except Exception as e:
                                logger.error(f"Error updating MHD file: {e}")
                                return jsonify({'error': f'Error processing MHD file: {str(e)}'}), 500
                    else:
                        logger.info(f"Found existing RAW file for MHD: {possible_raw}")
                        raw_filepath = possible_raw
                        
                        # Make sure the MHD file references the correct RAW filename
                        try:
                            raw_filename = os.path.basename(possible_raw)
                            logger.info(f"Checking MHD file references for RAW file: {raw_filename}")
                            
                            with open(filepath, 'r') as f:
                                mhd_content = f.readlines()
                            
                            needs_update = False
                            for i, line in enumerate(mhd_content):
                                if line.startswith('ElementDataFile') and raw_filename not in line:
                                    # Replace with the new RAW filename (with user prefix)
                                    mhd_content[i] = f"ElementDataFile = {raw_filename}\n"
                                    needs_update = True
                                    break
                            
                            if needs_update:
                                # Write the updated MHD file
                                with open(filepath, 'w') as f:
                                    f.writelines(mhd_content)
                                logger.info(f"Updated MHD file references to match existing RAW file")
                        except Exception as e:
                            logger.error(f"Error checking MHD file references: {e}")
                            # Continue processing as the file might still work
            
            # Get processing options from form data
            # Enhanced error handling for confidence threshold
            try:
                confidence_str = request.form.get('confidence', '0.5')
                logger.info(f"Received confidence value: {confidence_str} (type: {type(confidence_str)})")
                
                # Try to convert to float with proper error handling
                confidence_threshold = float(confidence_str)
                # Ensure value is in valid range
                original_confidence = confidence_threshold
                confidence_threshold = max(0.0, min(1.0, confidence_threshold))
                
                if original_confidence != confidence_threshold:
                    logger.warning(f"Confidence threshold adjusted from {original_confidence} to {confidence_threshold} (clamped to [0,1] range)")
                
                logger.info(f"Parsed confidence threshold: {confidence_threshold}")
            except ValueError as e:
                logger.error(f"Error parsing confidence threshold: {e}, using default 0.5")
                confidence_threshold = 0.5
            
            # Process other options
            lungs_only = request.form.get('lungs_only') == 'true'
            
            logger.info(f"Processing options: confidence={confidence_threshold}, lungs_only={lungs_only}")
            
            try:
                # Create a unique job ID
                job_id = f"{os.path.splitext(filename)[0]}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                logger.info(f"Starting processing with job ID: {job_id}")
                
                # Get fast_mode parameter (default to True for better performance)
                fast_mode = request.form.get('fast_mode', 'true') == 'true'
                batch_size = 16  # Default batch size
                
                # If using GPU, increase the batch size
                if torch.cuda.is_available():
                    # Clear GPU memory before processing
                    torch.cuda.empty_cache()
                    logger.info("Cleared GPU memory before processing scan")
                    
                    # Check available VRAM and adjust batch size accordingly
                    try:
                        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                        free_memory_gb = free_memory / (1024**3)  # Convert to GB
                        
                        # Adjust batch size based on available memory
                        if free_memory_gb > 8:
                            batch_size = 32
                        elif free_memory_gb > 4:
                            batch_size = 24
                        logger.info(f"Adjusted batch size to {batch_size} based on {free_memory_gb:.2f}GB available VRAM")
                    except Exception as e:
                        logger.warning(f"Could not check GPU memory: {e}, using default batch size")
                
                # Reset models to ensure no state is retained between scans
                if nodules_model is not None:
                    nodules_model.eval()  # Ensure in evaluation mode
                if lungs_model is not None:
                    lungs_model.eval()  # Ensure in evaluation mode
                
                # Log important processing parameters
                logger.info(f"Processing scan with confidence_threshold={confidence_threshold}, batch_size={batch_size}")
                
                # For MHD files, verify that the ElementDataFile path is correctly set
                if filepath.lower().endswith('.mhd'):
                    try:
                        # Parse the MHD file to extract the ElementDataFile path
                        with open(filepath, 'r') as f:
                            mhd_content = f.read()
                        
                        # Find the ElementDataFile line
                        import re
                        elementdatafile_match = re.search(r'ElementDataFile\s*=\s*(.+)', mhd_content, re.IGNORECASE)
                        
                        if elementdatafile_match:
                            raw_path = elementdatafile_match.group(1).strip()
                            logger.info(f"Found ElementDataFile reference in MHD: {raw_path}")
                            
                            # If the RAW path contains directory separators, it may cause issues
                            if '/' in raw_path or '\\' in raw_path:
                                # Extract just the filename part
                                raw_filename = os.path.basename(raw_path)
                                logger.warning(f"ElementDataFile contains path separators: {raw_path}")
                                logger.info(f"Updating to use just the filename: {raw_filename}")
                                
                                # Update the MHD file to use just the filename
                                updated_content = re.sub(
                                    r'(ElementDataFile\s*=\s*).+', 
                                    f'\\1{raw_filename}', 
                                    mhd_content, 
                                    flags=re.IGNORECASE
                                )
                                
                                # Write the updated content back to the file
                                with open(filepath, 'w') as f:
                                    f.write(updated_content)
                                
                                logger.info("Updated MHD file to use correct RAW reference")
                    except Exception as e:
                        logger.error(f"Error checking/fixing MHD file: {e}")
                        # Continue processing as this is just an enhancement
                        
                # Process the scan with optimized parameters
                result = process_ct_scan(
                    filepath,
                    app.config['OUTPUT_FOLDER'],
                    lungs_model=lungs_model,
                    nodules_model=None if lungs_only else nodules_model,
                    malignant_benign_model=malignant_benign_model,
                    confidence_threshold=confidence_threshold,
                    batch_size=batch_size,
                    fast_mode=fast_mode
                )
                
                processing_time = result.get('processing_time', 0)
                logger.info(f"Processing completed successfully for job: {job_id} in {processing_time:.2f}s")
                
                # Extract the case_name from the result
                case_name = result.get('case_name', os.path.splitext(filename)[0])
                
                # Store the temporary result path in a session variable
                session_key = f"temp_result_{job_id}"
                app.config[session_key] = {
                    'case_name': case_name,
                    'job_id': job_id,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'file_path': filepath,
                    'raw_file_path': raw_filepath,
                    'expires_at': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat(),  # Results expire after 1 hour
                    'nodule_count': len(result.get('nodules', [])),
                    'is_preview': True,
                    'processing_time': processing_time
                }
                
                # Create a preview URL
                preview_url = f"/api/results/preview/{job_id}"
                
                # Return the result for preview, but not yet permanently saved
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and processed successfully. Click the "Results Preview" button to review and save or discard.',
                    'job_id': job_id,
                    'filename': filename,
                    'processing_time': processing_time,
                    'preview_url': preview_url,
                    'result_path': preview_url
                }), 200
                
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                logger.error(traceback.format_exc())  # Log the full traceback
                return jsonify({
                    'error': f'Error processing file: {str(e)}'
                }), 500
        else:
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': f'File type not allowed. Supported formats: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
            
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred during upload'}), 500

@app.route('/api/results', methods=['GET'])
@jwt_required()
def list_results():
    """List all available result files."""
    try:
        logger.info("Fetching list of results")
        results_dir = Path(app.config['OUTPUT_FOLDER'])
        
        # Create the results directory if it doesn't exist
        if not results_dir.exists():
            logger.warning(f"Results directory does not exist: {results_dir}")
            results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created results directory: {results_dir}")
            return jsonify([]), 200
        
        # Get all result files
        result_files = []
        
        # Look for result files with different patterns
        for pattern in ['*_results.png', '*_nodules.png', '*.png']:
            for result_file in results_dir.glob(pattern):
                case_name = result_file.stem
                if '_results' in case_name:
                    case_name = case_name.replace('_results', '')
                if '_nodules' in case_name:
                    case_name = case_name.replace('_nodules', '')
                
                # Avoid duplicate entries
                if not any(r['case_name'] == case_name for r in result_files):
                    # Check for patient information
                    patient_info = None
                    patient_info_path = results_dir / f"{case_name}_patient_info.json"
                    if patient_info_path.exists():
                        try:
                            with open(patient_info_path, 'r') as f:
                                patient_info = json.load(f)
                        except Exception as e:
                            logger.error(f"Error loading patient information for {case_name}: {e}")
                    
                    # Check for details file to extract information if needed
                    details = ""
                    details_path = results_dir / f"{case_name}_results.txt"
                    if details_path.exists():
                        try:
                            with open(details_path, 'r') as f:
                                details = f.read()
                        except Exception as e:
                            logger.error(f"Error loading details for {case_name}: {e}")
                    
                    result_data = {
                        'case_name': case_name,
                        'image_url': f"/api/results/{case_name}/image",
                        'details_url': f"/api/results/{case_name}/details",
                        'timestamp': datetime.datetime.fromtimestamp(result_file.stat().st_mtime).isoformat()
                    }
                    
                    # Add details and patient info if available
                    if details:
                        result_data['details'] = details
                    
                    if patient_info:
                        result_data['patient_info'] = patient_info
                    
                    result_files.append(result_data)
        
        logger.info(f"Found {len(result_files)} result files")
        return jsonify(result_files), 200
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        logger.error(traceback.format_exc())
        return jsonify([]), 500

@app.route('/api/results/<case_name>', methods=['GET', 'DELETE'])
def get_result(case_name):
    """Get or delete a specific result."""
    try:
        # Special handling for DICOM volume jobs which might be accessed 
        # after session expiry when redirecting from the upload page
        is_dicom_volume_job = case_name.startswith(('dicom_volume_', 'zip_volume_'))
        is_dicom_filename = '-' in case_name or case_name.endswith('.dcm')
        
        current_user = None
        jwt_required_exception = None
        job_username = None
        associated_job_id = None  # Add variable to track associated job ID
        
        try:
            # Try to get JWT identity but don't fail immediately if it's not available
            jwt_required()(lambda: None)()
            current_user = get_jwt_identity()
        except Exception as e:
            jwt_required_exception = e
            logger.warning(f"JWT validation failed: {e}")
            
        # For DICOM volume jobs, extract the username from the job ID for permission check
        # even if the JWT has expired
        if is_dicom_volume_job:
            # Extract the username from the job ID format: dicom_volume_username_timestamp
            parts = case_name.split('_')
            if len(parts) >= 3:
                job_username = parts[2]  # username is the third part of the job ID
                
                # If no JWT is available but the URL contains the username in the job ID
                # we can allow access for viewing (but not deleting)
                if current_user is None and request.method == 'GET':
                    logger.info(f"No JWT available but allowing access to {case_name} because it contains username in job ID")
                    current_user = job_username
                    
        # If we don't have a current user at this point, raise the original exception
        if current_user is None:
            if jwt_required_exception:
                raise jwt_required_exception
            else:
                # This should never happen, but just in case
                return jsonify({'error': 'Authentication required'}), 401
        
        # Get current user's role for permission check
        user_role = users_db.get(current_user, {}).get('role', '')
        
        # Skip permissions check for admin role
        is_admin = user_role == 'admin'
        
        # Get current user's previous usernames for continuity
        previous_usernames = users_db.get(current_user, {}).get('previous_usernames', [])
        
        # Always allow access for DICOM volume jobs where the username matches the job ID
        if is_admin:
            # Admins can access everything
            logger.info(f"Admin {current_user} accessing result for {case_name}")
        elif is_dicom_volume_job and job_username and job_username == current_user:
            # User is accessing their own DICOM volume job
            logger.info(f"Allowing access to result for {case_name} for user {current_user} (matches job username)")
        else:
            # Check if this is a DICOM filename (like 1-111) instead of a job ID
            # If so, we need to find the associated job ID for permission checks
            authorized = False
            
            if is_dicom_filename:
                logger.info(f"Case name {case_name} appears to be a DICOM filename, checking job associations")
                
                # Look through the access_db for any jobs that have this user's access
                user_accessible_jobs = [job_id for job_id, users in access_db.items() 
                                     if current_user in users or any(prev in users for prev in previous_usernames)]
                
                # For DICOM volume jobs, check if the slice is part of the job
                for job_id in user_accessible_jobs:
                    if job_id.startswith(('dicom_volume_', 'zip_volume_')) and job_id.split('_')[2] == current_user:
                        # This is a DICOM volume job owned by the current user
                        job_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
                        
                        # Check if this DICOM file is part of this job
                        if job_dir.exists():
                            # Look for the specific DICOM file or any reference to it
                            dicom_file_pattern = f"{case_name}*"
                            matching_files = list(job_dir.glob(dicom_file_pattern))
                            
                            if matching_files:
                                logger.info(f"Found DICOM file {case_name} in job {job_id}, granting access")
                                authorized = True
                                # Store the job ID for later use when finding result files
                                associated_job_id = job_id
                                # Add the mapping to access_db to simplify future checks
                                if case_name not in access_db:
                                    access_db[case_name] = []
                                if current_user not in access_db[case_name]:
                                    access_db[case_name].append(current_user)
                                break
            
            # Now check the standard permission methods if not yet authorized
            if not authorized:
                # Check if user directly has access
                if case_name in access_db and (current_user in access_db[case_name] or 
                                               any(prev in access_db[case_name] for prev in previous_usernames)):
                    logger.info(f"User {current_user} has explicit access to {case_name}")
                    authorized = True
                # Users can access their own cases (cases that start with their username)
                elif case_name.startswith(current_user) or any(case_name.startswith(prev_username) for prev_username in previous_usernames):
                    logger.info(f"User {current_user} has access to {case_name} as case owner")
                    authorized = True
                
                # If not authorized, deny access
                if not authorized:
                    logger.warning(f"Unauthorized access attempt to result {case_name} by user {current_user}")
                    return jsonify({'error': 'Unauthorized access. You can only view your own results.'}), 403
        
        result_dir = Path(app.config['OUTPUT_FOLDER'])
        
        # Handle DELETE request
        if request.method == 'DELETE':
            try:
                # Find all files related to this case
                files_to_delete = list(result_dir.glob(f"{case_name}*"))
                
                if not files_to_delete:
                    logger.warning(f"No files found to delete for case {case_name}")
                    return jsonify({'error': 'Result files not found'}), 404
                
                # Delete all files related to this case
                for file_path in files_to_delete:
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
                
                # Also check for any source files in upload folder that should be deleted
                upload_dir = Path(app.config['UPLOAD_FOLDER'])
                source_files = list(upload_dir.glob(f"{case_name}*"))
                for file_path in source_files:
                    file_path.unlink()
                    logger.info(f"Deleted source file: {file_path}")
                
                logger.info(f"Successfully deleted all files for case: {case_name}")
                return jsonify({'message': f'Successfully deleted case: {case_name}'}), 200
            except Exception as e:
                logger.error(f"Error deleting case {case_name}: {e}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Failed to delete case: {str(e)}'}), 500
        
        # GET request handling
        image_path = None
        details = None
        
        # If we're dealing with a DICOM filename and have found an associated job ID, 
        # look for results in that job's directory
        if is_dicom_filename and associated_job_id:
            logger.info(f"Looking for result files for DICOM {case_name} in job directory {associated_job_id}")
            job_dir = result_dir / associated_job_id
            
            # Look for a results file in the job directory
            png_files = list(job_dir.glob("*_results.png"))
            if png_files:
                image_path = png_files[0]
                logger.info(f"Found result image in job directory: {image_path}")
                
                # Also look for results text file
                results_txt_files = list(job_dir.glob("*_results.txt"))
                if results_txt_files:
                    try:
                        with open(results_txt_files[0], 'r') as f:
                            details = f.read()
                        logger.info(f"Found details file in job directory: {results_txt_files[0]}")
                    except Exception as e:
                        logger.error(f"Error reading details file from job directory: {e}")
                        details = "Error reading details file"
        
        # If no image found in job directory, try the traditional approach
        if image_path is None:
            # Check if result image exists in main directory
            image_path = result_dir / f"{case_name}_results.png"
            
            # If image doesn't exist, the case_name might include a timestamp that's not in the result file
            # Try to find the base filename without the timestamp
            if not image_path.exists():
                logger.info(f"Result not found with exact name: {case_name}, trying to extract base name")
                
                # Check if case_name has timestamp format (user_sometext_YYYYMMDDHHMMSS)
                parts = case_name.split('_')
                if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 14:
                    # Remove timestamp part and reconstruct base name
                    base_case_name = '_'.join(parts[:-1])
                    logger.info(f"Extracted base case name: {base_case_name}")
                    
                    # Check if image exists with base name
                    base_image_path = result_dir / f"{base_case_name}_results.png"
                    if base_image_path.exists():
                        logger.info(f"Found result image with base name: {base_image_path}")
                        case_name = base_case_name
                        image_path = base_image_path
            
            # If still no image found, try searching all job directories
            if not image_path.exists() and is_dicom_filename:
                logger.info(f"Searching all job directories for DICOM {case_name} results")
                for job_dir in result_dir.glob("dicom_volume_*"):
                    if job_dir.is_dir():
                        # Check if this job directory has results for our case
                        png_files = list(job_dir.glob("*_results.png"))
                        if png_files and any(list(job_dir.glob(f"{case_name}*"))):
                            image_path = png_files[0]
                            logger.info(f"Found result image in job directory {job_dir.name}: {image_path}")
                            # Also try to find the details file
                            results_txt_files = list(job_dir.glob("*_results.txt"))
                            if results_txt_files:
                                try:
                                    with open(results_txt_files[0], 'r') as f:
                                        details = f.read()
                                    logger.info(f"Found details file in job directory: {results_txt_files[0]}")
                                except Exception as e:
                                    logger.error(f"Error reading details file from job directory: {e}")
                                    details = "Error reading details file"
                            break
        
        # If still no image found, return 404
        if not image_path or not image_path.exists():
            logger.warning(f"No result image found for case: {case_name}")
            return jsonify({'error': 'Result not found'}), 404
        
        # If we don't have details yet, check for details in main directory
        if details is None:
            details_path = result_dir / f"{case_name}_results.txt"
            if details_path.exists():
                try:
                    with open(details_path, 'r') as f:
                        details = f.read()
                except Exception as e:
                    logger.error(f"Error reading details file for {case_name}: {e}")
                    details = "Error reading details file"
        
        # Check for patient information
        patient_info = None
        patient_info_path = result_dir / f"{case_name}_patient_info.json"
        if patient_info_path.exists():
            try:
                with open(patient_info_path, 'r') as f:
                    patient_info = json.load(f)
            except Exception as e:
                logger.error(f"Error reading patient info file for {case_name}: {e}")
        
        # Return the case details
        result_data = {
            'case_name': case_name,
            'image_url': f"/api/results/{case_name}/image",
            'details': details,
            'patient_info': patient_info
        }
        
        # If no details were found but we have the image,
        # generate a basic detail message to avoid confusion
        if details is None:
            result_data['details'] = "No detailed information available for this result."
        
        return jsonify(result_data), 200
            
    except Exception as e:
        logger.error(f"Error in get_result: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An error occurred retrieving the result: {str(e)}'}), 500

@app.route('/api/results/<case_name>/image', methods=['GET'])
def get_result_image(case_name):
    """Get the result image for a specific case."""
    try:
        # Simplified access - no token validation
        logger.info(f"Image request for {case_name}")
        
        # Check for the image file
        results_dir = Path(app.config['OUTPUT_FOLDER'])
        
        # If case_name is a job ID (starts with dicom_volume_ or zip_volume_), 
        # look in the job-specific subfolder
        if case_name.startswith(('dicom_volume_', 'zip_volume_')):
            # This is a job ID, look in the job-specific subfolder
            job_dir = results_dir / case_name
            if job_dir.exists():
                # Look for any PNG file in the job directory
                png_files = list(job_dir.glob("*_results.png"))
                if png_files:
                    logger.info(f"Serving job-specific image: {png_files[0]}")
                    return send_file(str(png_files[0]), mimetype='image/png')
                
                # If no _results.png files found, try any PNG file
                png_files = list(job_dir.glob("*.png"))
                if png_files:
                    logger.info(f"Serving alternative job image: {png_files[0]}")
                    return send_file(str(png_files[0]), mimetype='image/png')
        
        # Check if this is a DICOM filename (like 1-111)
        is_dicom_filename = '-' in case_name or case_name.endswith('.dcm')
        
        # For DICOM filenames, try to find associated job directories
        if is_dicom_filename:
            logger.info(f"Looking for image for DICOM filename {case_name} in job directories")
            
            # Check if we have any access mappings for this DICOM filename
            associated_jobs = []
            
            # First, look in access_db to see if this DICOM file is associated with any jobs
            for job_id in access_db.keys():
                if job_id.startswith(('dicom_volume_', 'zip_volume_')):
                    job_dir = results_dir / job_id
                    if job_dir.exists():
                        # Check if this job directory has the DICOM file
                        matching_files = list(job_dir.glob(f"{case_name}*"))
                        if matching_files:
                            logger.info(f"Found DICOM file {case_name} in job {job_id}")
                            associated_jobs.append(job_id)
            
            # If no jobs found in access_db, search all dicom_volume job directories
            if not associated_jobs:
                logger.info(f"No mappings in access_db for {case_name}, searching all job directories")
                for job_dir in results_dir.glob("dicom_volume_*"):
                    if job_dir.is_dir():
                        # Check if this job directory has the DICOM file
                        matching_files = list(job_dir.glob(f"{case_name}*"))
                        if matching_files:
                            logger.info(f"Found DICOM file {case_name} in job {job_dir.name}")
                            associated_jobs.append(job_dir.name)
            
            # Check each associated job directory for result images
            for job_id in associated_jobs:
                job_dir = results_dir / job_id
                # Look for any PNG file in the job directory
                png_files = list(job_dir.glob("*_results.png"))
                if png_files:
                    logger.info(f"Serving job-specific image for DICOM {case_name}: {png_files[0]}")
                    return send_file(str(png_files[0]), mimetype='image/png')
                
                # If no _results.png files found, try any PNG file
                png_files = list(job_dir.glob("*.png"))
                if png_files:
                    logger.info(f"Serving alternative job image for DICOM {case_name}: {png_files[0]}")
                    return send_file(str(png_files[0]), mimetype='image/png')
        
        # Traditional case - try different possible image filenames in the main results directory
        possible_filenames = [
            f"{case_name}_results.png",
            f"{case_name}_nodules.png"
        ]
        
        # Try each possible filename
        for filename in possible_filenames:
            image_path = results_dir / filename
            if image_path.exists():
                logger.info(f"Serving image: {image_path}")
                return send_file(str(image_path), mimetype='image/png')
        
        # If no specific filename matches, search for any PNG file with this case name
        png_files = list(results_dir.glob(f"{case_name}*.png"))
        if png_files:
            logger.info(f"Serving found image: {png_files[0]}")
            return send_file(str(png_files[0]), mimetype='image/png')
            
        # If no image found, return 404
        logger.warning(f"No image found for case: {case_name}")
        return jsonify({'error': 'Image not found'}), 404
            
    except Exception as e:
        logger.error(f"Error in get_result_image: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred retrieving the image'}), 500

@app.route('/api/results/<case_name>/details', methods=['GET'])
def get_result_details(case_name):
    """Get the result details for a specific case."""
    try:
        # Special handling for DICOM volume jobs which might be accessed 
        # after session expiry when redirecting from the upload page
        is_dicom_volume_job = case_name.startswith(('dicom_volume_', 'zip_volume_'))
        is_dicom_filename = '-' in case_name or case_name.endswith('.dcm')
        
        current_user = None
        jwt_required_exception = None
        job_username = None
        associated_job_id = None  # Track associated job ID for DICOM filenames
        
        try:
            # Try to get JWT identity but don't fail immediately if it's not available
            jwt_required()(lambda: None)()
            current_user = get_jwt_identity()
        except Exception as e:
            jwt_required_exception = e
            logger.warning(f"JWT validation failed for details: {e}")
            
        # For DICOM volume jobs, extract the username from the job ID for permission check
        # even if the JWT has expired
        if is_dicom_volume_job:
            # Extract the username from the job ID format: dicom_volume_username_timestamp
            parts = case_name.split('_')
            if len(parts) >= 3:
                job_username = parts[2]  # username is the third part of the job ID
                
                # If no JWT is available but the URL contains the username in the job ID
                # we can allow access 
                if current_user is None:
                    logger.info(f"No JWT available but allowing access to details for {case_name} because it contains username in job ID")
                    current_user = job_username
                    
        # If we don't have a current user at this point, raise the original exception
        if current_user is None:
            if jwt_required_exception:
                raise jwt_required_exception
            else:
                # This should never happen, but just in case
                return jsonify({'error': 'Authentication required'}), 401
        
        # Get current user's role for permission check
        user_role = users_db.get(current_user, {}).get('role', '')
        
        # Skip permissions check for admin role
        is_admin = user_role == 'admin'
        
        # Get previous usernames for looking up by previous username
        previous_usernames = users_db.get(current_user, {}).get('previous_usernames', [])
        
        # Always allow access for DICOM volume jobs where the username matches the job ID
        if is_admin:
            # Admins can access everything
            logger.info(f"Admin {current_user} accessing details for {case_name}")
        elif is_dicom_volume_job and job_username and job_username == current_user:
            # User is accessing their own DICOM volume job
            logger.info(f"Allowing access to details for {case_name} for user {current_user} (matches job username)")
        elif is_dicom_filename:
            # Check if this DICOM filename is associated with any of the user's jobs
            logger.info(f"Case name {case_name} appears to be a DICOM filename, checking job associations")
            
            # Look through the access_db for any jobs that have this user's access
            user_accessible_jobs = [job_id for job_id, users in access_db.items() 
                                 if current_user in users or any(prev in users for prev in previous_usernames)]
            
            # For DICOM volume jobs, check if the slice is part of the job
            authorized = False
            for job_id in user_accessible_jobs:
                if job_id.startswith(('dicom_volume_', 'zip_volume_')) and job_id.split('_')[2] == current_user:
                    # This is a DICOM volume job owned by the current user
                    job_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
                    
                    # Check if this DICOM file is part of this job
                    if job_dir.exists():
                        # Look for the specific DICOM file or any reference to it
                        dicom_file_pattern = f"{case_name}*"
                        matching_files = list(job_dir.glob(dicom_file_pattern))
                        
                        if matching_files:
                            logger.info(f"Found DICOM file {case_name} in job {job_id}, granting access")
                            authorized = True
                            associated_job_id = job_id
                            # Add the mapping to access_db to simplify future checks
                            if case_name not in access_db:
                                access_db[case_name] = []
                            if current_user not in access_db[case_name]:
                                access_db[case_name].append(current_user)
                            break
            
            if not authorized:
                logger.warning(f"Unauthorized access attempt to details for DICOM file {case_name} by user {current_user}")
                return jsonify({'error': 'Unauthorized access. You can only view your own results.'}), 403
        else:
            # Regular case name check with standard permission checks
            authorized = False
            # Check if user directly has access to this case
            if case_name in access_db and (current_user in access_db[case_name] or any(prev in access_db[case_name] for prev in previous_usernames)):
                logger.info(f"User {current_user} has explicit access to case {case_name}")
                authorized = True
            # Check if case belongs to the user (based on username prefix)
            elif case_name.startswith(current_user) or any(case_name.startswith(prev) for prev in previous_usernames):
                logger.info(f"User {current_user} has access to own case {case_name}")
                authorized = True
                
            if not authorized:
                logger.warning(f"Unauthorized access attempt to details for {case_name} by user {current_user}")
                return jsonify({'error': 'Unauthorized access. You can only view your own results.'}), 403
            
        # Check for results details
        results_dir = Path(app.config['OUTPUT_FOLDER'])
        
        # First try to find details in the job-specific directory for DICOM filenames
        details = None
        
        if is_dicom_filename and associated_job_id:
            # Look in the associated job directory
            job_dir = results_dir / associated_job_id
            if job_dir.exists():
                # Look for results.txt file in the job directory
                results_txt_files = list(job_dir.glob("*_results.txt"))
                if results_txt_files:
                    try:
                        with open(results_txt_files[0], 'r') as f:
                            details = f.read()
                        logger.info(f"Found details file in job directory: {results_txt_files[0]}")
                    except Exception as e:
                        logger.error(f"Error reading details file from job directory: {e}")
                        details = "Error reading details file"
        
        # If no details found in job directory, search all job directories
        if details is None and is_dicom_filename:
            logger.info(f"Searching all job directories for details for DICOM {case_name}")
            for job_dir in results_dir.glob("dicom_volume_*"):
                if job_dir.is_dir():
                    # Check if this job directory has the DICOM file
                    matching_files = list(job_dir.glob(f"{case_name}*"))
                    if matching_files:
                        # Look for results.txt file
                        results_txt_files = list(job_dir.glob("*_results.txt"))
                        if results_txt_files:
                            try:
                                with open(results_txt_files[0], 'r') as f:
                                    details = f.read()
                                logger.info(f"Found details file in job directory {job_dir.name}: {results_txt_files[0]}")
                                break
                            except Exception as e:
                                logger.error(f"Error reading details file from job directory {job_dir.name}: {e}")
        
        # If still no details found, try the main directory
        if details is None:
            # Try to find details file in the main results directory
            details_path = results_dir / f"{case_name}_results.txt"
            
            if not details_path.exists():
                # Try other possible filenames
                possible_filenames = [
                    f"{case_name}_nodules.txt",
                    f"{case_name}.txt"
                ]
                
                for filename in possible_filenames:
                    test_path = results_dir / filename
                    if test_path.exists():
                        details_path = test_path
                        break
            
            if details_path.exists():
                try:
                    with open(details_path, 'r') as f:
                        details = f.read()
                    logger.info(f"Found details file in main directory: {details_path}")
                except Exception as e:
                    logger.error(f"Error reading details file {details_path}: {e}")
                    details = "Error reading details file"
        
        # If no details found, check if an image exists
        if details is None:
            # Check for image in job directory first
            image_found = False
            
            if is_dicom_filename:
                for job_dir in results_dir.glob("dicom_volume_*"):
                    if job_dir.is_dir():
                        # Check if this job directory has the DICOM file
                        matching_files = list(job_dir.glob(f"{case_name}*"))
                        if matching_files:
                            # Check if there's an image result file
                            result_files = list(job_dir.glob("*_results.png"))
                            if result_files:
                                image_found = True
                                break
            
            if not image_found:
                # Check main directory for image
                image_path = results_dir / f"{case_name}_results.png"
                if image_path.exists():
                    image_found = True
            
            if image_found:
                details = "This result has been generated, but no detailed text information is available."
            else:
                return jsonify({'error': 'No details found for this case'}), 404
        
        # Additional patient details if available
        patient_info = None
        patient_info_path = results_dir / f"{case_name}_patient_info.json"
        if patient_info_path.exists():
            try:
                with open(patient_info_path, 'r') as f:
                    patient_info = json.load(f)
            except Exception as e:
                logger.error(f"Error reading patient info file: {e}")
        
        # Check for nodules data
        nodules_data = None
        nodules_path = results_dir / f"{case_name}_nodules.json"
        
        # If not found in main directory, check job directory
        if not nodules_path.exists() and is_dicom_filename:
            for job_dir in results_dir.glob("dicom_volume_*"):
                if job_dir.is_dir():
                    # Check if this job directory has the DICOM file
                    matching_files = list(job_dir.glob(f"{case_name}*"))
                    if matching_files:
                        # Look for nodules.json file
                        nodules_files = list(job_dir.glob("*_nodules.json"))
                        if nodules_files:
                            nodules_path = nodules_files[0]
                            break
        
        if nodules_path.exists():
            try:
                with open(nodules_path, 'r') as f:
                    nodules_data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading nodules data file: {e}")
        
        # Return the details
        result_data = {
            'details': details,
            'patient_info': patient_info,
            'nodules': nodules_data
        }
        
        return jsonify(result_data), 200
    
    except Exception as e:
        logger.error(f"Error in get_result_details: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An error occurred retrieving the details: {str(e)}'}), 500

# Add API endpoint for interactive slice visualization
@app.route('/api/results/<case_name>/slices', methods=['GET'])
def get_result_slices(case_name):
    """Get interactive slice data for visualization."""
    try:
        # Get axis and index parameters
        axis = request.args.get('axis', 'axial')
        try:
            index = int(request.args.get('index', 0))
        except ValueError:
            index = 0
            
        # Allow token to be passed as a query parameter for direct browser access
        token = request.args.get('token')
        
        # Special handling for DICOM volume jobs which might be accessed 
        # after session expiry when redirecting from the upload page
        is_dicom_volume_job = case_name.startswith(('dicom_volume_', 'zip_volume_'))
        
        current_user = None
        jwt_required_exception = None
        
        # First try to get JWT from Authorization header
        try:
            # Try to get JWT identity but don't fail immediately if it's not available
            jwt_required()(lambda: None)()
            current_user = get_jwt_identity()
            logger.info(f"Authenticated via JWT header: {current_user}")
        except Exception as e:
            jwt_required_exception = e
            logger.warning(f"JWT validation failed for slices: {e}")
            
            # If header auth failed, try token from query parameter
            if token:
                try:
                    # Verify and decode the token
                    decoded_token = decode_token(token)
                    current_user = decoded_token['sub']  # 'sub' contains the identity
                    logger.info(f"Authenticated via token parameter: {current_user}")
                except Exception as token_e:
                    logger.warning(f"Token parameter validation failed: {token_e}")
            
        # For DICOM volume jobs, extract the username from the job ID for permission check
        # even if the JWT has expired
        job_username = None
        if is_dicom_volume_job:
            # Extract the username from the job ID format: dicom_volume_username_timestamp
            parts = case_name.split('_')
            if len(parts) >= 3:
                job_username = parts[2]  # username is the third part of the job ID
                
                # If no JWT is available but the URL contains the username in the job ID
                # we can allow access 
                if current_user is None:
                    logger.info(f"No JWT available but allowing access to slice for {case_name} because it contains username in job ID")
                    current_user = job_username
        
        # If we don't have a current user at this point, try to extract owner from case_name
        if current_user is None and '_' in case_name:
            # Many case names start with username_
            potential_user = case_name.split('_')[0]
            if potential_user in users_db:
                logger.info(f"No auth but allowing access to {case_name} based on case name prefix")
                current_user = potential_user
                
        # If we still don't have a current user at this point, check if this is a preview (public access)
        if current_user is None and (case_name.startswith('preview_') or 'preview' in request.args):
            logger.info(f"Allowing public access to preview: {case_name}")
            # Use a generic user for preview cases
            current_user = "guest"
        
        # Last resort - if we're in development mode, allow access for debugging
        if current_user is None and app.debug:
            logger.warning(f"Allowing debug access without authentication to: {case_name}")
            current_user = "debug"
        
        # If we don't have a current user at this point, raise the original exception
        if current_user is None:
            if jwt_required_exception:
                raise jwt_required_exception
            else:
                # This should never happen, but just in case
                return jsonify({'error': 'Authentication required. Please include a token parameter or Authorization header.'}), 401
        
        # Get current user's role for permission check
        user_role = users_db.get(current_user, {}).get('role', '')
        
        # Skip permissions check for admin role
        is_admin = user_role == 'admin'
        
        # Determine if we have permission for this case
        authorized = False
        
        # Always allow access for admins
        if is_admin:
            # Admins can access everything
            logger.info(f"Admin {current_user} accessing slices for {case_name}")
            authorized = True
        # Allow users to access their own DICOM volume jobs
        elif is_dicom_volume_job and job_username and job_username == current_user:
            # User is accessing their own DICOM volume job
            logger.info(f"Allowing access to {case_name} for user {current_user} (matches job username)")
            authorized = True
        # For preview/debug cases, always allow access
        elif current_user in ["guest", "debug"]:
            logger.info(f"Allowing {current_user} access to {case_name}")
            authorized = True
        else:
            # Get previous usernames for looking up by previous username
            previous_usernames = users_db.get(current_user, {}).get('previous_usernames', [])
            
            # Check if this is a DICOM filename (like 1-111) instead of a job ID
            # If so, we need to find the associated job ID for permission checks
            if not authorized and not is_dicom_volume_job:
                # Check if the case_name is a DICOM filename (simple check if it contains a hyphen)
                is_dicom_filename = '-' in case_name or case_name.endswith('.dcm') or case_name.isdigit() or (case_name.startswith('IM') and case_name[2:].isdigit())
                
                if is_dicom_filename:
                    logger.info(f"Case name {case_name} appears to be a DICOM filename, checking job associations")
                    
                    # Look through the access_db for any jobs that have this user's access
                    user_accessible_jobs = [job_id for job_id, users in access_db.items() 
                                         if current_user in users or any(prev in users for prev in previous_usernames)]
                    
                    # For DICOM volume jobs, check if the slice is part of the job
                    for job_id in user_accessible_jobs:
                        if job_id.startswith(('dicom_volume_', 'zip_volume_')) and job_id.split('_')[2] == current_user:
                            # This is a DICOM volume job owned by the current user
                            job_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
                            
                            # Check if this DICOM file is part of this job
                            if job_dir.exists():
                                # Look for the specific DICOM file or any reference to it
                                dicom_file_pattern = f"{case_name}*"
                                matching_files = list(job_dir.glob(dicom_file_pattern))
                                
                                if matching_files:
                                    logger.info(f"Found DICOM file {case_name} in job {job_id}, granting access")
                                    authorized = True
                                    # Add the mapping to access_db to simplify future checks
                                    if case_name not in access_db:
                                        access_db[case_name] = []
                                    if current_user not in access_db[case_name]:
                                        access_db[case_name].append(current_user)
                                    break
            
            # Now check the standard permission methods if not yet authorized
            if not authorized:
                # Regular case name check with standard permission checks
                # Check if user directly has access
                if case_name in access_db and (current_user in access_db[case_name] or 
                                             any(prev in access_db[case_name] for prev in previous_usernames)):
                    logger.info(f"User {current_user} has explicit access to {case_name}")
                    authorized = True
                # Users can access their own cases (cases that start with their username)
                elif case_name.startswith(current_user) or any(case_name.startswith(prev_username) for prev_username in previous_usernames):
                    logger.info(f"User {current_user} has access to {case_name} as case owner")
                    authorized = True
        
        # For development/testing purposes, allow access if needed
        if not authorized and app.debug and request.args.get('bypass_auth') == 'debug':
            logger.warning(f"DEBUG MODE: Bypassing auth check for {case_name}")
            authorized = True
        
        if not authorized:
            logger.warning(f"Unauthorized access attempt to slice data {case_name} by user {current_user}")
            return jsonify({'error': 'Unauthorized access. You can only view your own results.'}), 403
        
        # Check if this is a DICOM volume job
        is_dicom_volume = case_name.startswith(('dicom_volume_', 'zip_volume_'))
        
        # Check if case_name has timestamp format (user_sometext_YYYYMMDDHHMMSS)
        original_case_name = case_name
        base_case_name = None
        
        parts = case_name.split('_')
        if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 14:
            # Remove timestamp part and reconstruct base name
            base_case_name = '_'.join(parts[:-1])
            logger.info(f"Extracted base case name for slices: {base_case_name}")
        
        # Check if we have the original CT data (it would be in uploads)
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        results_dir = Path(app.config['OUTPUT_FOLDER'])
        
        # Define a function to find source files based on case name
        def find_source_files(case_name):
            # Initialize source_files list to avoid UnboundLocalError
            source_files = []
            
            # First check for direct MHD mapping (new method)
            mhd_mapping_found = False
            
            # For DICOM volume jobs, check mhd_mapping
            if is_dicom_volume and 'mhd_mapping' in app.config and case_name in app.config['mhd_mapping']:
                mhd_file = app.config['mhd_mapping'][case_name]
                logger.info(f"Found direct MHD mapping for DICOM volume job {case_name}: {mhd_file}")
                mhd_path = Path(mhd_file)
                if mhd_path.exists():
                    source_files.append(mhd_path)
                    mhd_mapping_found = True
            
            # For DICOM filenames, check dicom_to_mhd
            elif not is_dicom_volume and ('-' in case_name or case_name.endswith('.dcm') or case_name.isdigit() or (case_name.startswith('IM') and case_name[2:].isdigit())):
                if 'dicom_to_mhd' in app.config and case_name in app.config['dicom_to_mhd']:
                    mhd_file = app.config['dicom_to_mhd'][case_name]
                    logger.info(f"Found direct MHD mapping for DICOM file {case_name}: {mhd_file}")
                    mhd_path = Path(mhd_file)
                    if mhd_path.exists():
                        source_files.append(mhd_path)
                        mhd_mapping_found = True
            
            # Continue with original logic if no mapping found
            if not mhd_mapping_found:
                # Define search locations
                search_locations = [
                    upload_dir,  # Check uploads folder
                    results_dir,  # Check results folder
                    results_dir / case_name  # Check case-specific results folder
                ]
                
                # For DICOM volume jobs, look in the job-specific subfolder
                if is_dicom_volume:
                    job_dir = results_dir / case_name
                    if job_dir.exists():
                        logger.info(f"Using job-specific directory for slices: {job_dir}")
                        
                        # Also try to find the original MHD file in the job directory
                        for ext in ['.mhd', '.nii.gz', '.nii']:
                            job_volume_files = list(job_dir.glob(f"*{ext}"))
                            if job_volume_files:
                                source_files.extend(job_volume_files)
                                logger.info(f"Found volume files in job directory: {job_volume_files}")
                                break
                # For DICOM filenames (like 1-111), look in each user's job directory
                is_dicom_filename = '-' in case_name or case_name.endswith('.dcm') or case_name.isdigit() or (case_name.startswith('IM') and case_name[2:].isdigit())
                associated_job_ids = []
                
                if is_dicom_filename:
                    # This is likely a DICOM filename - first find which job directories contain this file
                    logger.info(f"Searching for DICOM file {case_name} in job directories")
                    
                    # Step 1: Find all job directories that contain this DICOM file
                    for job_dir in results_dir.glob("dicom_volume_*"):
                        if job_dir.is_dir():
                            # Check if this job directory has the DICOM file
                            dicom_files = list(job_dir.glob(f"{case_name}*"))
                            if dicom_files:
                                logger.info(f"Found DICOM file {case_name} in job {job_dir.name}")
                                associated_job_ids.append(job_dir.name)
                                # Add this directory to search locations
                                search_locations.append(job_dir)
                    
                    # Step 2: For each associated job, check for volume files in uploads
                    for job_id in associated_job_ids:
                        # Check if we have a mapping for this job ID
                        if 'mhd_mapping' in app.config and job_id in app.config['mhd_mapping']:
                            mhd_file = app.config['mhd_mapping'][job_id]
                            logger.info(f"Found associated MHD file through mapping: {mhd_file}")
                            mhd_path = Path(mhd_file)
                            if mhd_path.exists():
                                return [str(mhd_path)]
                        
                        # The convention for volume files in uploads is: username_dicom_volume_timestamp_volume.mhd
                        # Extract the important parts from the job ID
                        parts = job_id.split('_')
                        if len(parts) >= 4:
                            username = parts[2]
                            timestamp = parts[3] if len(parts) > 3 else None
                            
                            if username and timestamp:
                                # Check for the associated volume file pattern in uploads
                                volume_pattern = f"{username}_dicom_volume_{timestamp}_volume.mhd"
                                volume_files = list(upload_dir.glob(volume_pattern))
                                if volume_files:
                                    logger.info(f"Found associated volume file for DICOM {case_name}: {volume_files[0]}")
                                    # Return this immediately as it's the most reliable match
                                    return [str(path) for path in volume_files]
                                
                                # Try with wildcard for timestamp (in case of formatting differences)
                                volume_pattern = f"{username}_dicom_volume_*_volume.mhd"
                                volume_files = list(upload_dir.glob(volume_pattern))
                                if volume_files:
                                    logger.info(f"Found volume file using wildcard pattern: {volume_files[0]}")
                                    # Return this immediately as it's still a good match
                                    return [str(path) for path in volume_files]
                                
                                # Try with just username (even less specific)
                                volume_pattern = f"{username}*_volume.mhd"
                                volume_files = list(upload_dir.glob(volume_pattern))
                                if volume_files:
                                    logger.info(f"Found volume file with username pattern: {volume_files[0]}")
                                    # Return this immediately as it's still a reasonable match
                                    return [str(path) for path in volume_files]
                
                # File extensions to search for (in order of preference)
                file_extensions = ['.mhd', '.nii.gz', '.nii', '.dcm']
                
                # Define search patterns based on case name
                search_patterns = [
                    f"{case_name}*",  # Base pattern with wildcard
                    f"{case_name}_volume*",  # Common volume suffix
                    f"{case_name}"  # Exact match
                ]
                
                # For DICOM volume jobs, add more specific patterns
                if is_dicom_volume or is_dicom_filename:
                    if base_case_name:
                        search_patterns.append(f"{base_case_name}*")
                    # Extract timestamp if present for timestamp-based matching
                    timestamp = case_name.split('_')[-1] if '_' in case_name else None
                    if timestamp and timestamp != case_name:
                        search_patterns.append(f"*{timestamp}*_volume*")
                        search_patterns.append(f"*{timestamp}*")
                        
                    # For DICOM volume jobs, add patterns to search for volume files
                    if is_dicom_volume:
                        # Extract username from job ID
                        if '_' in case_name:
                            parts = case_name.split('_')
                            if len(parts) >= 3:
                                username = parts[2]
                                search_patterns.append(f"{username}_dicom_volume_*_volume*")
                
                # Include upload directory subdirectories
                additional_search_locations = []
                for location in search_locations:
                    if location.exists():
                        # Add any dicom_volume_* subdirectories
                        for subdir in location.glob("*_dicom_volume_*"):
                            if subdir.is_dir():
                                additional_search_locations.append(subdir)
                        # Also check deeper - look for any dicom_volume subdirectories 
                        for subdir in location.glob("**/dicom_volume*"):
                            if subdir.is_dir():
                                additional_search_locations.append(subdir)
                
                # Add additional locations to search
                search_locations.extend(additional_search_locations)
                
                # Search all locations with all patterns and extensions
                for location in search_locations:
                    if not location.exists():
                        continue
                        
                    for pattern in search_patterns:
                        for ext in file_extensions:
                            # Handle special case for extended extensions like .nii.gz
                            if '.' in ext:
                                found_files = list(location.glob(f"{pattern}{ext}"))
                            else:
                                found_files = list(location.glob(f"{pattern}{ext}"))
                            
                            if found_files:
                                logger.info(f"Found files matching pattern {pattern}{ext} in {location}: {found_files}")
                                source_files.extend(found_files)
                
                # If no files found with patterns and this is a DICOM filename, search for corresponding volume files
                if not source_files and is_dicom_filename:
                    logger.info("No direct match found, searching for volume files for all job directories")
                    # First check for associated job IDs in the access_db
                    user_jobs = []
                    # Find all dicom_volume job IDs in access_db that mention the current user
                    for job_id, users in access_db.items():
                        if job_id.startswith("dicom_volume_") and current_user in users:
                            user_jobs.append(job_id)
                    
                    # For each user job, look for volume files in uploads directory
                    for job_id in user_jobs:
                        job_parts = job_id.split('_')
                        if len(job_parts) >= 4:
                            username = job_parts[2]
                            timestamp = job_parts[3]
                            # Look for volume file with this timestamp
                            volume_pattern = f"{username}_dicom_volume_{timestamp}*_volume.mhd"
                            volume_files = list(upload_dir.glob(volume_pattern))
                            if volume_files:
                                logger.info(f"Found volume files for job {job_id}: {volume_files}")
                                source_files.extend(volume_files)
                                break  # Found a good match, no need to continue
                    
                    # If still not found, try subdirectories in uploads
                    if not source_files:
                        # Check all dicom_volume_* directories in uploads
                        for dicom_job_dir in upload_dir.glob("*_dicom_volume_*"):
                            if dicom_job_dir.is_dir():
                                for ext in file_extensions:
                                    volume_files = list(dicom_job_dir.glob(f"*_volume{ext}"))
                                    if volume_files:
                                        logger.info(f"Found volume files in uploads subdirectory {dicom_job_dir}: {volume_files}")
                                        source_files.extend(volume_files)
                    
                    # If still not found, do a broader search for all volume files
                    if not source_files:
                        volume_files = list(upload_dir.glob(f"*_volume.mhd"))
                        if volume_files:
                            logger.info(f"Found volume files using broader search: {volume_files}")
                            source_files.extend(volume_files)
                            
                        # Try even broader - any MHD file
                        if not source_files:
                            mhd_files = list(upload_dir.glob("*.mhd"))
                            if mhd_files:
                                logger.info(f"Found MHD files using broadest search: {mhd_files}")
                                source_files.extend(mhd_files)
                
                # If no files found with patterns, try a more general search
                if not source_files:
                    # Extract DICOM identifier if case name has format username_dicom_id
                    if '_' in case_name and not is_dicom_volume:
                        parts = case_name.split('_', 1)
                        if len(parts) > 1:
                            dicom_id = parts[1]
                            for location in search_locations:
                                if location.exists():
                                    for ext in file_extensions:
                                        matching_files = list(location.glob(f"*{dicom_id}*{ext}"))
                                        if matching_files:
                                            logger.info(f"Found files using DICOM ID search: {matching_files}")
                                            source_files.extend(matching_files)
                    
                    # Last resort - try searching upload directory for all .mhd files
                    if not source_files:
                        for location in search_locations[:1]:  # Only check upload folder for this last resort
                            if location.exists():
                                # Use most recent MHD file if available
                                mhd_files = list(location.glob("*.mhd"))
                                if mhd_files:
                                    # Sort by last modified time (newest first)
                                    mhd_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                                    logger.info(f"Last resort: using most recent MHD file: {mhd_files[0]}")
                                    source_files = [mhd_files[0]]
                
                # Sort files to ensure consistent results
                source_files.sort()
            
            # Convert all Path objects to strings before returning
            return [str(path) for path in source_files]
        
        # Get axis and index parameters
        axis = request.args.get('axis', 'axial')
        try:
            index = int(request.args.get('index', 0))
        except ValueError:
            index = 0
        
        # Find all potential source files
        source_files = find_source_files(case_name)
        
        if source_files:
            logger.info(f"Found source files: {source_files}")
            source_file = source_files[0]  # Use the first file found
            
            # Check and fix MHD file if needed
            if str(source_file).lower().endswith('.mhd'):
                logger.info(f"Checking and fixing MHD file before loading: {source_file}")
                if check_and_fix_mhd_file(source_file):
                    logger.info(f"Successfully checked/fixed MHD file: {source_file}")
                else:
                    logger.warning(f"Could not fix MHD file, but will attempt to load: {source_file}")
            
            try:
                # Load the volume data
                file_path = str(source_file)
                
                # Load volume based on file type
                if file_path.endswith(('.nii', '.nii.gz')):
                    # NIfTI handling code remains the same...
                    pass
                else:
                    # MHD loading code remains the same...
                    reader = sitk.ImageFileReader()
                    
                    try:
                        logger.info(f"Setting filename to: {str(source_file)}")
                        reader.SetFileName(str(source_file))
                        
                        # Try to read just the metadata first
                        try:
                            logger.info("Reading image information...")
                            reader.ReadImageInformation()
                            file_size = reader.GetSize()
                            pixel_type = reader.GetPixelID()
                            logger.info(f"Image info: size={file_size}, pixel type={pixel_type}")
                        except Exception as info_e:
                            logger.error(f"Failed to read image information: {info_e}")
                        
                        logger.info("Executing SimpleITK reader...")
                        image = reader.Execute()
                        logger.info(f"Image dimensions: {image.GetSize()}, pixel type: {image.GetPixelIDTypeAsString()}")
                        
                        # Get array dimensions before conversion
                        volume = sitk.GetArrayFromImage(image)
                        logger.info(f"Loaded volume with shape: {volume.shape}, dtype: {volume.dtype}, min: {volume.min()}, max: {volume.max()}")
                        
                        # Check for NaN or extreme values
                        if np.isnan(volume).any():
                            logger.warning("Volume contains NaN values!")
                            volume = np.nan_to_num(volume, nan=-1000)
                            
                        if np.isinf(volume).any():
                            logger.warning("Volume contains infinite values!")
                            volume = np.nan_to_num(volume, posinf=3000, neginf=-1000)
                        
                        logger.info(f"Loaded volume with shape: {volume.shape}")
                    except Exception as e:
                        # Error handling for loading volume remains the same...
                        raise
                
                # If no index was requested, just return volume information
                if 'index' not in request.args and 'axis' not in request.args:
                    # Get image spacing from SimpleITK
                    spacing = [1.0, 1.0, 1.0]  # Default spacing
                    try:
                        if 'image' in locals():
                            sitk_spacing = image.GetSpacing()
                            spacing = [float(s) for s in sitk_spacing]
                    except Exception as spacing_error:
                        logger.error(f"Error getting spacing: {spacing_error}")
                    
                    # Get dimensions from volume shape
                    dimensions = {
                        'depth': int(volume.shape[0]),
                        'height': int(volume.shape[1]),
                        'width': int(volume.shape[2])
                    }
                    
                    # Return volume metadata
                    return jsonify({
                        'volume_info': {
                            'dimensions': dimensions,
                            'spacing': spacing,
                            'min_value': float(volume.min()),
                            'max_value': float(volume.max()),
                            'source_file': str(source_file),
                            'nodules': load_nodules_for_case(case_name)  # Load nodules data
                        }
                    })
                
                # Process volume based on requested axis
                try:
                    # Make sure the volume has 3 dimensions
                    if len(volume.shape) != 3:
                        logger.error(f"Expected 3D volume but got shape: {volume.shape}")
                        if len(volume.shape) == 2:
                            logger.warning("Converting 2D volume to 3D by adding dimension")
                            volume = volume.reshape(1, volume.shape[0], volume.shape[1])
                        elif len(volume.shape) == 4 and volume.shape[3] == 1:
                            logger.warning("Converting 4D volume to 3D by removing channel dimension")
                            volume = volume[:, :, :, 0]
                        else:
                            raise ValueError(f"Cannot process volume with shape {volume.shape}")
                    
                    # Verify the volume is not empty
                    if 0 in volume.shape:
                        raise ValueError(f"Volume has zero-sized dimension: {volume.shape}")
                
                    # Make sure index is valid for each axis
                    if axis == 'axial':
                        max_index = volume.shape[0] - 1
                        if max_index < 0:
                            raise ValueError(f"Invalid volume shape for axial view: {volume.shape}")
                        
                        if index > max_index:
                            index = max_index
                        elif index < 0:
                            index = 0
                        
                        # Get axial slice
                        slice_data = volume[index].astype(np.float32)
                        logger.info(f"Generated axial slice at index {index} with shape {slice_data.shape}")
                        
                        # Create a highlight mask for overlays (if needed)
                        highlight_mask = np.zeros_like(slice_data, dtype=np.uint8)
                        
                    elif axis == 'coronal':
                        # For coronal view
                        max_index = volume.shape[1] - 1
                        if max_index < 0:
                            raise ValueError(f"Invalid volume shape for coronal view: {volume.shape}")
                        
                        if index > max_index:
                            index = max_index
                        elif index < 0:
                            index = 0
                        
                        # Get coronal slice
                        slice_data = volume[:, index, :].astype(np.float32)
                        logger.info(f"Generated coronal slice at index {index} with shape {slice_data.shape}")
                        
                        # Create a highlight mask for overlays (if needed)
                        highlight_mask = np.zeros_like(slice_data, dtype=np.uint8)
                        
                    elif axis == 'sagittal':
                        # For sagittal view
                        max_index = volume.shape[2] - 1
                        if max_index < 0:
                            raise ValueError(f"Invalid volume shape for sagittal view: {volume.shape}")
                        
                        if index > max_index:
                            index = max_index
                        elif index < 0:
                            index = 0
                        
                        # Get sagittal slice
                        slice_data = volume[:, :, index].astype(np.float32)
                        logger.info(f"Generated sagittal slice at index {index} with shape {slice_data.shape}")
                        
                        # Create a highlight mask for overlays (if needed)
                        highlight_mask = np.zeros_like(slice_data, dtype=np.uint8)
                        
                    else:
                        raise ValueError(f"Invalid axis: {axis}")
                    
                    # Rest of the function remains the same...
                    
                    # Create a figure to render the slice
                    plt.figure(figsize=(8, 8))
                    
                    # Clear any existing plots to avoid "Argument must be an image or collection in this Axes" error
                    plt.clf()
                    
                    # Create new axes for the image
                    ax = plt.gca()
                    
                    # Calculate appropriate aspect ratio based on pixel spacing
                    aspect_ratio = 1.0  # Default to 1.0 if spacing information is not available
                    if hasattr(image, 'GetSpacing'):
                        try:
                            spacing = image.GetSpacing()
                            if len(spacing) >= 2:
                                aspect_ratio = spacing[1] / spacing[0]
                        except Exception as e:
                            logger.warning(f"Could not determine aspect ratio from spacing: {e}")
                    
                    # Display the image with the correct aspect ratio inside the square figure
                    ax.imshow(slice_data, cmap='gray', aspect=aspect_ratio)
                    
                    # Check if there are nodules to highlight
                    try:
                        # Load nodules for this case
                        nodules = load_nodules_for_case(case_name)
                        
                        # Create highlight mask for nodules in this slice
                        if nodules:
                            # Define threshold for nodule visibility (how many slices away)
                            visibility_threshold = 5  # Nodules within 5 slices will show
                            
                            for nodule in nodules:
                                # Ensure nodule has all needed coordinates
                                if not all(coord in nodule for coord in ['x', 'y', 'z']):
                                    continue
                                    
                                # Get nodule position based on current view
                                if axis == 'axial':
                                    # For axial view, check if nodule z is close to current slice
                                    if abs(nodule['z'] - index) <= visibility_threshold:
                                        # Draw circle at (x, y) coordinates
                                        nodule_x, nodule_y = int(nodule['x']), int(nodule['y'])
                                        
                                        # Calculate radius in pixels (default to 10 if not specified)
                                        radius_pixels = int(nodule.get('radius', 10))
                                        
                                        # Make circles more visible by increasing radius
                                        radius_pixels = int(radius_pixels * 1.5)  # Increase radius by 50%
                                        
                                        # Create a circle on the highlight mask
                                        y_indices, x_indices = np.ogrid[:slice_data.shape[0], :slice_data.shape[1]]
                                        dist = np.sqrt((x_indices - nodule_x) ** 2 + (y_indices - nodule_y) ** 2)
                                        
                                        # Create a hollow circle (ring) for better visibility
                                        # Make the ring thinner for clarity
                                        ring_width = max(2, radius_pixels // 5)  # Thinner ring
                                        
                                        # Outer circle
                                        outer_circle = dist <= radius_pixels
                                        # Inner circle (smaller by ring_width)
                                        inner_circle = dist <= (radius_pixels - ring_width)
                                        # Ring is outer circle minus inner circle
                                        ring = outer_circle & ~inner_circle
                                        
                                        # Set intensity - use a lighter color for better visibility (200 instead of 255)
                                        highlight_mask[ring] = 200
                                
                                elif axis == 'coronal':
                                    # For coronal view, check if nodule y is close to current slice
                                    if abs(nodule['y'] - index) <= visibility_threshold:
                                        # Draw circle at (x, z) coordinates
                                        nodule_x, nodule_z = int(nodule['x']), int(nodule['z'])
                                        
                                        # Make sure coordinates are within bounds
                                        if 0 <= nodule_z < slice_data.shape[0] and 0 <= nodule_x < slice_data.shape[1]:
                                            # Calculate radius in pixels (default to 10 if not specified)
                                            radius_pixels = int(nodule.get('radius', 10))
                                            
                                            # Make circles more visible by increasing radius
                                            radius_pixels = int(radius_pixels * 1.5)  # Increase radius by 50%
                                            
                                            # Create a circle on the highlight mask
                                            z_indices, x_indices = np.ogrid[:slice_data.shape[0], :slice_data.shape[1]]
                                            dist = np.sqrt((x_indices - nodule_x) ** 2 + (z_indices - nodule_z) ** 2)
                                            
                                            # Create a hollow circle (ring) for better visibility
                                            # Make the ring thinner for clarity
                                            ring_width = max(2, radius_pixels // 5)  # Thinner ring
                                            
                                            # Outer circle
                                            outer_circle = dist <= radius_pixels
                                            # Inner circle (smaller by ring_width)
                                            inner_circle = dist <= (radius_pixels - ring_width)
                                            # Ring is outer circle minus inner circle
                                            ring = outer_circle & ~inner_circle
                                            
                                            # Set intensity - use a lighter color for better visibility (200 instead of 255)
                                            highlight_mask[ring] = 200
                                
                                elif axis == 'sagittal':
                                    # For sagittal view, check if nodule x is close to current slice
                                    if abs(nodule['x'] - index) <= visibility_threshold:
                                        # Draw circle at (y, z) coordinates
                                        nodule_y, nodule_z = int(nodule['y']), int(nodule['z'])
                                        
                                        # Make sure coordinates are within bounds
                                        if 0 <= nodule_z < slice_data.shape[0] and 0 <= nodule_y < slice_data.shape[1]:
                                            # Calculate radius in pixels (default to 10 if not specified)
                                            radius_pixels = int(nodule.get('radius', 10))
                                            
                                            # Make circles more visible by increasing radius
                                            radius_pixels = int(radius_pixels * 1.5)  # Increase radius by 50%
                                            
                                            # Create a circle on the highlight mask
                                            z_indices, y_indices = np.ogrid[:slice_data.shape[0], :slice_data.shape[1]]
                                            dist = np.sqrt((y_indices - nodule_y) ** 2 + (z_indices - nodule_z) ** 2)
                                            
                                            # Create a hollow circle (ring) for better visibility
                                            # Make the ring thinner for clarity
                                            ring_width = max(2, radius_pixels // 5)  # Thinner ring
                                            
                                            # Outer circle
                                            outer_circle = dist <= radius_pixels
                                            # Inner circle (smaller by ring_width)
                                            inner_circle = dist <= (radius_pixels - ring_width)
                                            # Ring is outer circle minus inner circle
                                            ring = outer_circle & ~inner_circle
                                            
                                            # Set intensity - use a lighter color for better visibility (200 instead of 255)
                                            highlight_mask[ring] = 200
                            
                            logger.info(f"Created highlight mask with rings for {len(nodules)} nodules in {axis} view at index {index}")
                    except Exception as e:
                        logger.warning(f"Could not create nodule highlights: {e}")
                        logger.warning(traceback.format_exc())
                    
                    # Overlay highlights for nodules if any exist
                    if np.any(highlight_mask > 0):
                        highlight_overlay = np.ma.masked_where(highlight_mask == 0, highlight_mask)
                        ax.imshow(highlight_overlay, cmap='Reds_r', alpha=0.8)  # Use Reds_r colormap for lighter red appearance
                    
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    
                    # Save the figure to a bytes buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.close('all')  # Ensure all figures are closed
                    
                    # Encode the image as base64
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    
                    # Get dimensions for all axes to help the UI
                    all_dimensions = {
                        'axial': volume.shape[0],
                        'coronal': volume.shape[1],
                        'sagittal': volume.shape[2]
                    }
                    
                    # Return image data with complete information
                    return jsonify({
                        'slice_data': img_base64,
                        'max_index': max_index,
                        'current_index': index,
                        'source_file': str(source_file),
                        'all_dimensions': all_dimensions,
                        'dimensions': {
                            'width': slice_data.shape[1],
                            'height': slice_data.shape[0]
                        }
                    })
                
                except Exception as e:
                    logger.error(f"Error extracting slice: {e}")
                    logger.error(traceback.format_exc())
                    # Return a meaningful error response
                    return jsonify({
                        'error': f'Error extracting slice: {str(e)}',
                        'volume_shape': volume.shape if 'volume' in locals() else 'unknown',
                        'axis': axis,
                        'index': index
                    }), 500
                
            except Exception as e:
                logger.error(f"Error in get_slice: {e}")
                logger.error(traceback.format_exc())
                
                try:
                    # Create a placeholder image for errors
                    plt.close('all')  # Close any existing figures
                    plt.figure(figsize=(8, 8))
                    plt.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center')
                    plt.axis('off')
                    
                    # Save to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.close('all')
                    
                    # Encode
                    buf.seek(0)
                    placeholder_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    
                    # Return placeholder with error message and basic dimensions
                    return jsonify({
                        'slice_data': placeholder_base64,
                        'max_index': 0,  # Only one slice available in error case
                        'current_index': 0,
                        'is_placeholder': True,
                        'error': str(e),
                        'message': f"Error loading slice: {str(e)}",
                        'dimensions': {'width': 512, 'height': 512},  # Default dimensions
                        'all_dimensions': {'axial': 1, 'coronal': 1, 'sagittal': 1}  # Default dimensions
                    })
                except Exception as placeholder_error:
                    logger.error(f"Error creating placeholder: {placeholder_error}")
                    # If even placeholder creation fails, return a simple error message
                    return jsonify({
                        'error': str(e),
                        'message': f"Error loading slice: {str(e)}",
                        'is_placeholder': True,
                        'max_index': 0,
                        'dimensions': {'width': 512, 'height': 512},
                        'all_dimensions': {'axial': 1, 'coronal': 1, 'sagittal': 1}
                    }), 500
        else:
            # No source files found, return placeholder
            logger.warning(f"No source files found for case {case_name}")
            
            try:
                # Create a placeholder image
                plt.close('all')
                plt.figure(figsize=(8, 8))
                plt.text(0.5, 0.5, f"No source files found for {case_name}", 
                         horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                
                # Save to buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close('all')
                
                # Encode
                buf.seek(0)
                placeholder_base64 = base64.b64encode(buf.read()).decode('utf-8')
                
                return jsonify({
                    'slice_data': placeholder_base64,
                    'max_index': 0,
                    'current_index': 0,
                    'is_placeholder': True,
                    'message': f"No source files found for {case_name}",
                    'dimensions': {'width': 512, 'height': 512},
                    'all_dimensions': {'axial': 1, 'coronal': 1, 'sagittal': 1}
                })
            except Exception as placeholder_error:
                logger.error(f"Error creating placeholder: {placeholder_error}")
                return jsonify({
                    'error': str(placeholder_error),
                    'message': f"No source files found for {case_name}",
                    'is_placeholder': True,
                    'max_index': 0,
                    'dimensions': {'width': 512, 'height': 512},
                    'all_dimensions': {'axial': 1, 'coronal': 1, 'sagittal': 1}
                }), 500
    
    except Exception as e:
        logger.error(f"Error in get_result_slices: {e}")
        logger.error(traceback.format_exc())
        
        # Try to create a placeholder
        try:
            plt.close('all')
            plt.figure(figsize=(8, 8))
            plt.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close('all')
            
            buf.seek(0)
            placeholder_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return jsonify({
                'slice_data': placeholder_base64,
                'max_index': 0,
                'current_index': 0,
                'is_placeholder': True,
                'error': str(e),
                'message': f"Error processing request: {str(e)}",
                'dimensions': {'width': 512, 'height': 512},
                'all_dimensions': {'axial': 1, 'coronal': 1, 'sagittal': 1}
            })
        except:
            return jsonify({
                'error': str(e),
                'message': f"Error processing request: {str(e)}",
                'is_placeholder': True,
                'max_index': 0,
                'dimensions': {'width': 512, 'height': 512},
                'all_dimensions': {'axial': 1, 'coronal': 1, 'sagittal': 1}
            }), 500

# Add a health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the server is running."""
    try:
        # Check if models are loaded
        models_status = {
            'lungs_model': lungs_model is not None,
            'nodules_model': nodules_model is not None,
            'nodules_seg_model': nodules_seg_model is not None,
            'malignant_benign_model': malignant_benign_model is not None
        }
        
        # Check if required directories exist
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        results_dir = Path(app.config['OUTPUT_FOLDER'])
        
        dirs_status = {
            'upload_dir': upload_dir.exists(),
            'results_dir': results_dir.exists()
        }
        
        # Create any missing directories
        if not upload_dir.exists():
            upload_dir.mkdir(parents=True, exist_ok=True)
            dirs_status['upload_dir'] = "created"
            
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
            dirs_status['results_dir'] = "created"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.datetime.now().isoformat(),
            'models': models_status,
            'directories': dirs_status,
            'version': '1.0'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

# Serve React app and landing page
@app.route('/dashboard', defaults={'path': ''})
@app.route('/dashboard/<path:path>')
def serve_dashboard(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/users', methods=['GET'])
@jwt_required()
def get_users():
    """Get list of users (admin only)."""
    # Get current user identity and role from JWT
    current_user = get_jwt_identity()
    user_role = users_db.get(current_user, {}).get('role', '')
    
    # Check if user is admin
    if user_role not in ['admin', 'superadmin']:
        return jsonify({'error': 'Unauthorized. Admin access required.'}), 403
    
    # Return list of users (exclude passwords)
    user_list = []
    for username, data in users_db.items():
        user_data = {
            'username': username, 
            'role': data['role'],
            'first_name': data.get('first_name', ''),
            'last_name': data.get('last_name', ''),
            'email': data.get('email', ''),
            'description': data.get('description', ''),
            'plan': data.get('plan', 'usage_based'),
            'created_at': data.get('created_at', '')
        }
        user_list.append(user_data)
    
    return jsonify(user_list), 200

@app.route('/api/users/<username>', methods=['PUT'])
@jwt_required()
def update_user(username):
    """Update a user (admin can update anyone, users can only update themselves)."""
    # Get current user identity and role from JWT
    current_user = get_jwt_identity()
    user_role = users_db.get(current_user, {}).get('role', '')
    
    # Check if the user is updating their own profile or is an admin
    if current_user != username and user_role not in ['admin', 'superadmin']:
        return jsonify({'error': 'Unauthorized. You can only update your own profile unless you are an admin.'}), 403
    
    if not request.is_json:
        return jsonify({'error': 'Missing JSON in request'}), 400

    # Check if user exists
    if username not in users_db:
        return jsonify({'error': 'User not found'}), 404

    # Get update fields
    data = request.get_json()
    
    # Update user fields
    if 'email' in data:
        users_db[username]['email'] = data['email']
    if 'description' in data:
        users_db[username]['description'] = data['description']
    if 'plan' in data:
        users_db[username]['plan'] = data['plan']
    if 'password' in data:
        users_db[username]['password'] = generate_password_hash(data['password'])

    # Role changes require admin privileges
    if 'role' in data:
        if user_role != 'admin' and user_role != 'superadmin':
            return jsonify({'error': 'Only admins can change user roles'}), 403
        
        # Only superadmins can create superadmin accounts
        if data['role'] == 'superadmin' and user_role != 'superadmin':
            return jsonify({'error': 'Only superadmins can create superadmin accounts'}), 403
            
        users_db[username]['role'] = data['role']

    # Return updated user data
    return jsonify({
        'success': True,
        'message': 'User updated successfully',
        'user': {
            'username': username,
            'role': users_db[username]['role'],
            'email': users_db[username].get('email', ''),
            'description': users_db[username].get('description', ''),
            'plan': users_db[username].get('plan', 'usage_based')
        }
    }), 200

@app.route('/api/results/user', methods=['GET'])
@jwt_required()
def list_user_results():
    """List results associated with the current user (or all results for admins)."""
    try:
        # Get current user identity and role
        current_user = get_jwt_identity()
        user_role = users_db.get(current_user, {}).get('role', '')
        
        logger.info(f"Fetching results for user: {current_user} with role: {user_role}")
        results_dir = Path(app.config['OUTPUT_FOLDER'])
        
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
            return jsonify([]), 200
        
        # Get current user's previous usernames (for accessing old results)
        previous_usernames = users_db.get(current_user, {}).get('previous_usernames', [])
        logger.info(f"User {current_user} has previous usernames: {previous_usernames}")
        
        # Combine current and previous usernames to check for results
        usernames_to_check = [current_user] + previous_usernames
        
        # Get all result files
        result_files = []
        
        for pattern in ['*_results.png', '*_nodules.png', '*.png']:
            for result_file in results_dir.glob(pattern):
                case_name = result_file.stem
                if '_results' in case_name:
                    case_name = case_name.replace('_results', '')
                if '_nodules' in case_name:
                    case_name = case_name.replace('_nodules', '')
                
                # Avoid duplicate entries
                if not any(r['case_name'] == case_name for r in result_files):
                    # Check if the result belongs to any of the user's identities or if they're an admin
                    is_users_result = any(case_name.startswith(username) for username in usernames_to_check)
                    
                    if user_role == 'admin' or is_users_result:
                        # Check for patient information
                        patient_info = None
                        patient_info_path = results_dir / f"{case_name}_patient_info.json"
                        if patient_info_path.exists():
                            try:
                                with open(patient_info_path, 'r') as f:
                                    patient_info = json.load(f)
                            except Exception as e:
                                logger.error(f"Error loading patient information for {case_name}: {e}")
                        
                        # Check for details file to extract information if needed
                        details = ""
                        details_path = results_dir / f"{case_name}_results.txt"
                        if details_path.exists():
                            try:
                                with open(details_path, 'r') as f:
                                    details = f.read()
                            except Exception as e:
                                logger.error(f"Error loading details for {case_name}: {e}")
                        
                        result_data = {
                            'case_name': case_name,
                            'image_url': f"/api/results/{case_name}/image",
                            'details_url': f"/api/results/{case_name}/details",
                            'timestamp': datetime.datetime.fromtimestamp(result_file.stat().st_mtime).isoformat(),
                            'user': current_user
                        }
                        
                        # Add details and patient info if available
                        if details:
                            result_data['details'] = details
                        
                        if patient_info:
                            result_data['patient_info'] = patient_info
                            
                        result_files.append(result_data)
        
        logger.info(f"Found {len(result_files)} result files for user {current_user}")
        return jsonify(result_files), 200
    except Exception as e:
        logger.error(f"Error fetching results for user {current_user}: {e}")
        logger.error(traceback.format_exc())
        return jsonify([]), 500

# New route to handle saving or discarding the results
@app.route('/api/results/action/<job_id>', methods=['POST'])
@jwt_required()
def save_or_discard_result(job_id):
    """Permanently save or discard processed results."""
    try:
        # Get current user identity for tracking
        current_user = get_jwt_identity()
        
        # Check if the job_id exists in the temporary storage
        session_key = f"temp_result_{job_id}"
        if session_key not in app.config:
            logger.error(f"No temporary results found for job ID: {job_id}")
            return jsonify({'error': 'No temporary results found for this job ID'}), 404
        
        # Get the action parameter (save or discard)
        action = request.json.get('action', '')
        if action not in ['save', 'discard']:
            return jsonify({'error': 'Invalid action. Must be "save" or "discard"'}), 400
        
        # Get the temporary result data
        temp_result = app.config[session_key]
        case_name = temp_result['case_name']
        
        # Check if this is a DICOM volume job
        is_dicom_volume = job_id.startswith(('dicom_volume_', 'zip_volume_'))
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        
        if is_dicom_volume:
            # For DICOM volume jobs, use the job-specific subfolder
            job_dir = output_dir / job_id
            logger.info(f"Working with DICOM volume job directory: {job_dir}")
        
        if action == 'discard':
            try:
                if is_dicom_volume:
                    # Delete the entire job directory for DICOM volume jobs
                    if job_dir.exists():
                        import shutil
                        shutil.rmtree(job_dir)
                        logger.info(f"Deleted job directory: {job_dir}")
                else:
                    # For regular jobs, delete the individual files
                    case_files = list(output_dir.glob(f"{case_name}*.*"))
                    for file_path in case_files:
                        os.remove(file_path)
                        logger.info(f"Deleted temporary result file: {file_path}")
                    
                    # Also delete the uploaded files to clean up space
                    if 'file_path' in temp_result and temp_result['file_path'] and os.path.exists(temp_result['file_path']):
                        os.remove(temp_result['file_path'])
                        logger.info(f"Deleted uploaded file: {temp_result['file_path']}")
                    
                    if 'raw_file_path' in temp_result and temp_result['raw_file_path'] and os.path.exists(temp_result['raw_file_path']):
                        os.remove(temp_result['raw_file_path'])
                        logger.info(f"Deleted uploaded RAW file: {temp_result['raw_file_path']}")
                
                # Remove the temporary result from the session
                del app.config[session_key]
                
                return jsonify({
                    'success': True,
                    'message': 'Results discarded successfully'
                }), 200
                
            except Exception as e:
                logger.error(f"Error discarding results: {e}")
                return jsonify({'error': f'Error discarding results: {str(e)}'}), 500
        
        elif action == 'save':
            # Get patient information if provided
            patient_info = request.json.get('patient_info', {})
            
            # Add patient information to the results
            if patient_info:
                try:
                    if is_dicom_volume:
                        # Save patient info to the job directory
                        patient_info_file = job_dir / f"{case_name}_patient_info.json"
                        with open(patient_info_file, 'w') as f:
                            json.dump({
                                'name': patient_info.get('name', ''),
                                'age': patient_info.get('age', ''),
                                'description': patient_info.get('description', ''),
                                'timestamp': datetime.datetime.now().isoformat(),
                                'user': current_user
                            }, f, indent=2)
                        
                        logger.info(f"Saved patient information for DICOM volume job: {job_id}")
                        
                        # Find the details file in the job directory
                        details_files = list(job_dir.glob("*_results.txt"))
                        if details_files:
                            details_file = details_files[0]
                            with open(details_file, 'a') as f:
                                f.write("\n\n----- Patient Information -----\n")
                                f.write(f"Name: {patient_info.get('name', 'Not provided')}\n")
                                f.write(f"Age: {patient_info.get('age', 'Not provided')}\n")
                                if patient_info.get('description'):
                                    f.write(f"Clinical Notes: {patient_info.get('description')}\n")
                                f.write(f"Added by: {current_user}\n")
                                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    else:
                        # Traditional file structure
                        patient_info_file = output_dir / f"{case_name}_patient_info.json"
                        with open(patient_info_file, 'w') as f:
                            json.dump({
                                'name': patient_info.get('name', ''),
                                'age': patient_info.get('age', ''),
                                'description': patient_info.get('description', ''),
                                'timestamp': datetime.datetime.now().isoformat(),
                                'user': current_user
                            }, f, indent=2)
                        
                        logger.info(f"Saved patient information for case: {case_name}")
                        
                        # Also append patient info to the details text file if it exists
                        details_file = output_dir / f"{case_name}_results.txt"
                        if details_file.exists():
                            with open(details_file, 'a') as f:
                                f.write("\n\n----- Patient Information -----\n")
                                f.write(f"Name: {patient_info.get('name', 'Not provided')}\n")
                                f.write(f"Age: {patient_info.get('age', 'Not provided')}\n")
                                if patient_info.get('description'):
                                    f.write(f"Clinical Notes: {patient_info.get('description')}\n")
                                f.write(f"Added by: {current_user}\n")
                                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        
                except Exception as e:
                    logger.error(f"Error saving patient information: {e}")
                    # Continue with saving even if patient info fails
            
            # For DICOM volume jobs, make sure we copy the result files to the main directory
            # so they can be found by the API endpoints
            if is_dicom_volume:
                try:
                    import shutil
                    
                    # Copy the main results image
                    results_images = list(job_dir.glob("*_results.png"))
                    if results_images:
                        src_image = results_images[0]
                        dest_image = output_dir / f"{job_id}_results.png"
                        shutil.copy2(src_image, dest_image)
                        logger.info(f"Copied result image from {src_image} to {dest_image}")
                    else:
                        # Try any PNG file if no specific results.png found
                        png_files = list(job_dir.glob("*.png"))
                        if png_files:
                            src_image = png_files[0]
                            dest_image = output_dir / f"{job_id}_results.png"
                            shutil.copy2(src_image, dest_image)
                            logger.info(f"Copied alternative image from {src_image} to {dest_image}")
                        else:
                            logger.warning(f"No result image found to copy for job {job_id}")
                    
                    # Copy the results text file
                    results_texts = list(job_dir.glob("*_results.txt"))
                    if results_texts:
                        src_text = results_texts[0]
                        dest_text = output_dir / f"{job_id}_results.txt"
                        shutil.copy2(src_text, dest_text)
                        logger.info(f"Copied result text from {src_text} to {dest_text}")
                    
                    # Copy the nodules JSON file if it exists
                    nodules_jsons = list(job_dir.glob("*_nodules.json"))
                    if nodules_jsons:
                        src_json = nodules_jsons[0]
                        dest_json = output_dir / f"{job_id}_nodules.json"
                        shutil.copy2(src_json, dest_json)
                        logger.info(f"Copied nodules JSON from {src_json} to {dest_json}")
                    
                    # Copy MHD/RAW volume files for visualization
                    mhd_files = list(job_dir.glob("*.mhd"))
                    if mhd_files:
                        # Get the first MHD file
                        src_mhd = mhd_files[0]
                        dest_mhd = output_dir / f"{job_id}_volume.mhd"
                        
                        # Copy MHD file
                        shutil.copy2(src_mhd, dest_mhd)
                        logger.info(f"Copied MHD file from {src_mhd} to {dest_mhd}")
                        
                        # Also copy the corresponding RAW file if it exists
                        src_raw = src_mhd.with_suffix('.raw')
                        if src_raw.exists():
                            dest_raw = dest_mhd.with_suffix('.raw')
                            shutil.copy2(src_raw, dest_raw)
                            logger.info(f"Copied RAW file from {src_raw} to {dest_raw}")
                    
                    # Copy the patient info JSON if it exists
                    if patient_info_file.exists():
                        dest_patient_info = output_dir / f"{job_id}_patient_info.json"
                        shutil.copy2(patient_info_file, dest_patient_info)
                        logger.info(f"Copied patient info from {patient_info_file} to {dest_patient_info}")
                        
                except Exception as e:
                    logger.error(f"Error copying result files to main directory: {e}")
                    # Continue even if copying fails
            
            # Remove the temporary result from the session
            del app.config[session_key]
            
            # Response based on the action taken
            return jsonify({
                'success': True,
                'message': 'Results saved successfully',
                'case_name': case_name
            }), 200
    
    except Exception as e:
        logger.error(f"Error in save_or_discard_result: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


# Add a new route to get preview results for a specific job ID
@app.route('/api/results/preview/<job_id>', methods=['GET'])
def get_preview_result(job_id):  # Remove @jwt_required() to make this publicly accessible
    """Get preview results for a specific job ID."""
    try:
        # Check if the job_id exists in the temporary storage
        session_key = f"temp_result_{job_id}"
        if session_key not in app.config:
            logger.error(f"No temporary results found for job ID: {job_id}")
            return jsonify({'error': 'No temporary results found for this job ID'}), 404
        
        # Get the temporary result data
        temp_result = app.config[session_key]
        
        # Get case_name with a fallback to the job_id if not present
        case_name = temp_result.get('case_name', job_id)
        
        # Get result details from the job-specific subfolder
        job_output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        
        # Check if there's a job-specific output directory
        if not job_output_dir.exists():
            # For MHD/RAW uploads and regular file uploads, results might be in the main directory
            output_dir = Path(app.config['OUTPUT_FOLDER'])
            
            # Look for result files in the main output directory with the case name
            image_files = list(output_dir.glob(f"{case_name}_results.png"))
            if not image_files:
                image_files = list(output_dir.glob(f"{case_name}*.png"))
                
            if not image_files:
                logger.error(f"Preview results image not found for job ID: {job_id} (case: {case_name})")
                return jsonify({'error': 'Preview results not found. Please try uploading again.'}), 404
            
            image_file = image_files[0]
            logger.info(f"Found preview image in main directory: {image_file}")
            
            # Get the result details text file
            details_files = list(output_dir.glob(f"{case_name}*_results.txt"))
            details = ""
            if details_files:
                details_file = details_files[0]
                try:
                    with open(details_file, 'r') as f:
                        details = f.read()
                    logger.info(f"Loaded details from {details_file}")
                except Exception as e:
                    logger.error(f"Error reading details file: {e}")
        else:
            # Job-specific directory exists, look for files there
            logger.info(f"Looking for preview results in job directory: {job_output_dir}")
            
            # Look for result files in the job subfolder
            image_files = list(job_output_dir.glob("*_results.png"))
            if not image_files:
                image_files = list(job_output_dir.glob("*.png"))
                
            if not image_files:
                logger.error(f"Preview results image not found for job ID: {job_id} in {job_output_dir}")
                return jsonify({'error': 'Preview results not found'}), 404
            
            # Use the first image file found
            image_file = image_files[0]
            logger.info(f"Found preview image in job directory: {image_file}")
            
            # Get the result details text file
            details_files = list(job_output_dir.glob("*_results.txt"))
            details = ""
            if details_files:
                details_file = details_files[0]
                try:
                    with open(details_file, 'r') as f:
                        details = f.read()
                    logger.info(f"Loaded details from {details_file}")
                except Exception as e:
                    logger.error(f"Error reading details file: {e}")
        
        # Check for nodules info
        nodule_info = {}
        nodules_json_files = list(job_output_dir.glob("*_nodules.json"))
        if not nodules_json_files:
            # Check in the main directory
            nodules_json_files = list(Path(app.config['OUTPUT_FOLDER']).glob(f"{case_name}*_nodules.json"))
            
        if nodules_json_files:
            try:
                with open(nodules_json_files[0], 'r') as f:
                    nodule_info = json.load(f)
                logger.info(f"Loaded nodule info from {nodules_json_files[0]}")
            except Exception as e:
                logger.error(f"Error reading nodules JSON: {e}")
        
        # Read the processing_info.json file for status
        status = 'completed'
        try:
            # First check job-specific directory
            if job_output_dir.exists():
                processing_info_path = job_output_dir / 'processing_info.json'
                if processing_info_path.exists():
                    with open(processing_info_path, 'r') as f:
                        processing_info = json.load(f)
                        status = processing_info.get('status', 'completed')
        except Exception as e:
            logger.error(f"Error reading processing info: {e}")
        
        # Determine image path for frontend use
        # Use the session_key as the parameter to allow direct access without authentication
        image_path = f"/api/results/preview/image/{job_id}"
        
        # Create a result object with all available data
        result = {
            'case_name': case_name,
            'details': details,
            'job_id': job_id,
            'image_url': image_path,
            'timestamp': temp_result.get('timestamp', datetime.datetime.now().isoformat()),
            'is_preview': True,
            'status': status,
            'nodule_count': temp_result.get('nodule_count', nodule_info.get('nodule_count', 0)),
            'action_url': f"/api/results/action/{job_id}",
            'has_lung_segmentation': True,  # This is always true since we always do lung segmentation
            'filename': os.path.basename(temp_result.get('file_path', '')) if 'file_path' in temp_result else '',
            'processing_time': temp_result.get('processing_time', 0)
        }
        
        # Add nodule details if available
        if nodule_info and 'nodules' in nodule_info:
            result['nodules'] = nodule_info['nodules']
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting preview results: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error getting preview results: {str(e)}'}), 500

@app.route('/api/upload/dicom-volume', methods=['POST'])
@jwt_required()
def upload_dicom_volume():
    """Upload multiple DICOM files to be processed as a 3D volume."""
    try:
        # Get current user identity for tracking
        current_user = get_jwt_identity()
        logger.info(f"Starting DICOM volume upload process for user: {current_user}")
        
        if 'files[]' not in request.files:
            logger.error("No files part in the request")
            return jsonify({'error': 'No files part in the request'}), 400
            
        files = request.files.getlist('files[]')
        
        if len(files) == 0:
            logger.error("No selected files")
            return jsonify({'error': 'No selected files'}), 400
        
        if len(files) == 1 and files[0].filename.lower().endswith('.zip'):
            # Handle zip file containing multiple DICOM files
            return handle_zip_upload(files[0], current_user)
            
        # Create a temporary directory to store the DICOM files
        import tempfile
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        folder_name = os.path.basename(temp_dir)
        
        # Generate a user-specific prefix for the folder name
        user_prefix = f"{current_user}_dicom_volume_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        dicom_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_prefix)
        os.makedirs(dicom_dir, exist_ok=True)
        
        # Save all DICOM files to the temporary directory
        file_paths = []
        for file in files:
            if file.filename.lower().endswith('.dcm'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(dicom_dir, filename)
                logger.info(f"Saving DICOM file: {filename} for user {current_user}")
                file.save(filepath)
                file_paths.append(filepath)
            else:
                logger.warning(f"Skipping non-DICOM file: {file.filename}")
        
        if len(file_paths) == 0:
            logger.error("No valid DICOM files found")
            return jsonify({'error': 'No valid DICOM files found'}), 400
        
        logger.info(f"Saved {len(file_paths)} DICOM files to {dicom_dir}")
        
        # Now we need to convert the DICOM files to a 3D volume that our pipeline can process
        try:
            # Import SimpleITK for DICOM processing
            import SimpleITK as sitk
            
            # Create a DICOM series reader
            reader = sitk.ImageSeriesReader()
            
            # Get the DICOM series IDs
            series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
            
            if not series_IDs:
                logger.error("No DICOM series found in the uploaded files")
                return jsonify({'error': 'No valid DICOM series found in the uploaded files'}), 400
            
            # Use the first series ID
            series_ID = series_IDs[0]
            logger.info(f"Found DICOM series ID: {series_ID}")
            
            # Get the DICOM file names for this series
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ID)
            
            # Set the file names and load the DICOM series
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            # Save as a temporary MHD/RAW file that our pipeline can process
            output_basename = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_prefix}_volume")
            mhd_path = f"{output_basename}.mhd"
            raw_path = f"{output_basename}.raw"
            
            # Write the image
            sitk.WriteImage(image, mhd_path)
            logger.info(f"Successfully converted DICOM series to MHD/RAW: {mhd_path}")
            
            # Get processing options from form data
            try:
                confidence_str = request.form.get('confidence', '0.5')
                confidence_threshold = float(confidence_str)
                confidence_threshold = max(0.0, min(1.0, confidence_threshold))
                logger.info(f"Parsed confidence threshold: {confidence_threshold}")
            except ValueError as e:
                logger.error(f"Error parsing confidence threshold: {e}, using default 0.5")
                confidence_threshold = 0.5
            
            lungs_only = request.form.get('lungs_only') == 'true'
            logger.info(f"Processing options: confidence={confidence_threshold}, lungs_only={lungs_only}")
            
            # Create a unique job ID
            job_id = f"dicom_volume_{user_prefix}"
            logger.info(f"Starting processing with job ID: {job_id}")
            
            # Get fast_mode parameter (default to True for better performance)
            fast_mode = request.form.get('fast_mode', 'true') == 'true'
            batch_size = 16  # Default batch size
            
            # If using GPU, increase the batch size
            if torch.cuda.is_available():
                # Clear GPU memory before processing
                torch.cuda.empty_cache()
                logger.info("Cleared GPU memory before processing scan")
                
                # Check available VRAM and adjust batch size accordingly
                try:
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                    free_memory_gb = free_memory / (1024**3)  # Convert to GB
                    
                    # Adjust batch size based on available memory
                    if free_memory_gb > 8:
                        batch_size = 32
                    elif free_memory_gb > 4:
                        batch_size = 24
                    logger.info(f"Adjusted batch size to {batch_size} based on {free_memory_gb:.2f}GB available VRAM")
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}, using default batch size")
            
            # Reset models to ensure no state is retained between scans
            if nodules_model is not None:
                nodules_model.eval()  # Ensure in evaluation mode
            if lungs_model is not None:
                lungs_model.eval()  # Ensure in evaluation mode
            
            # Log the current models being used
            logger.info(f"Using lungs model: {lungs_model is not None}")
            logger.info(f"Using nodules model: {nodules_model is not None}")
            logger.info(f"Using malignant/benign model: {malignant_benign_model is not None}")
            
            # Prepare output directory
            output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process in a separate thread to not block the response
            import threading
            
            def process_scan_task():
                """Process the uploaded CT scan in a background thread."""
                # Define session key at the beginning to prevent NameError in error handling
                session_key = f"temp_result_{job_id}"
                
                # Initialize the session entry in app.config if it doesn't exist
                if session_key not in app.config:
                    app.config[session_key] = {
                        'status': 'initializing',
                        'job_id': job_id,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'case_name': job_id,  # Set initial case_name to job_id for default
                        'nodule_count': 0     # Initialize with 0 nodules
                    }
                
                try:
                    logger.info(f"Starting background processing task for job ID: {job_id}")
                    # Create a dedicated output directory for this job
                    job_output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
                    job_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Created job-specific output directory: {job_output_dir}")
                    
                    # Process the CT scan
                    logger.info(f"Processing {dicom_dir} with confidence threshold {confidence_threshold}")
                    start_time = time.time()
                    
                    # If no models loaded, load them now
                    if lungs_model is None or nodules_model is None:
                        logger.info("Models not loaded. Loading models now...")
                        load_models()
                    
                    # Additional check in case loading failed
                    if lungs_model is None:
                        logger.error("Lung segmentation model is not available. Cannot process scan.")
                        # Use app.config to store error for front-end
                        app.config[session_key].update({
                            'status': 'error',
                            'error_message': 'Lung segmentation model is not available. Cannot process scan.'
                        })
                        return
                    
                    # Run pipeline
                    try:
                        start_time = time.time()
                        
                        # Use different models based on options
                        if lungs_only:
                            logger.info("Running lung segmentation only (skipping nodule detection)")
                            result = process_ct_scan(
                                mhd_path,  # Use the MHD file path that was created from the DICOM series
                                job_output_dir,  # Use the job-specific output directory
                                lungs_model,
                                None,  # Skip nodule detection
                                None,  # Skip malignant/benign classification
                                None,  # Skip cancer type classification
                                confidence_threshold=confidence_threshold,
                                batch_size=batch_size,
                                fast_mode=fast_mode
                            )
                        else:
                            logger.info("Running full pipeline with nodule detection")
                            result = process_ct_scan(
                                mhd_path,  # Use the MHD file path that was created from the DICOM series
                                job_output_dir,  # Use the job-specific output directory
                                lungs_model,
                                nodules_model,
                                malignant_benign_model,
                                None,  # Skip cancer type classification
                                confidence_threshold=confidence_threshold,
                                batch_size=batch_size,
                                fast_mode=fast_mode
                            )
                        
                        if isinstance(result, dict):
                            processing_time = result.get('processing_time', time.time() - start_time)
                            nodule_count = result.get('nodule_count', 0)
                            logger.info(f"Processing completed in {processing_time} seconds")
                            
                            if nodule_count > 0:
                                logger.info(f"Detected {nodule_count} nodules")
                            else:
                                logger.info("No nodules detected")
                            
                            # Save case name for reference
                            case_name = result.get('case_name', os.path.basename(mhd_path))
                            
                            # Copy the source file to the job output directory for easier access
                            try:
                                # Use the unified file saving helper
                                dest_file = save_file_to_unified_location(mhd_path, job_id, current_user)
                                if dest_file:
                                    logger.info(f"Copied source file to job directory: {dest_file}")
                            except Exception as e:
                                logger.error(f"Error copying source file: {e}")
                            
                            # Grant access to the current user
                            if job_id not in access_db:
                                access_db[job_id] = []
                            if current_user not in access_db[job_id]:
                                access_db[job_id].append(current_user)
                                logger.info(f"Granted access to {job_id} for user {current_user}")
                            
                            # Copy result image to main results directory for easier access
                            result_image = job_output_dir / f"{case_name}_results.png"
                            if result_image.exists():
                                import shutil  # Add local import to fix the UnboundLocalError
                                main_result_image = Path(app.config['OUTPUT_FOLDER']) / f"{job_id}_results.png"
                                shutil.copy2(result_image, main_result_image)
                                logger.info(f"Copied result image to main directory: {main_result_image}")
                            
                            # Copy result text to main results directory
                            result_text = job_output_dir / f"{case_name}_results.txt"
                            if result_text.exists():
                                main_result_text = Path(app.config['OUTPUT_FOLDER']) / f"{job_id}_results.txt"
                                shutil.copy2(result_text, main_result_text)
                                logger.info(f"Copied result text to main directory: {main_result_text}")
                            
                            # Copy nodules JSON to main results directory
                            nodules_json = job_output_dir / f"{case_name}_nodules.json"
                            if nodules_json.exists():
                                main_nodules_json = Path(app.config['OUTPUT_FOLDER']) / f"{job_id}_nodules.json"
                                shutil.copy2(nodules_json, main_nodules_json)
                                logger.info(f"Copied nodules JSON to main directory: {main_nodules_json}")
                        
                        # Processing complete, update the session data
                        elapsed_time = time.time() - start_time
                        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
                        
                        # Update session with the result information
                        app.config[session_key].update({
                            'status': 'complete',
                            'elapsed_time': elapsed_time,
                            'case_name': case_name,
                            'job_output_dir': str(job_output_dir),
                            'has_nodules': bool(result.get('nodules', []))
                        })
                        
                        # Count nodules and add summary
                        nodules = result.get('nodules', [])
                        nodule_count = len(nodules)
                        if nodule_count > 0:
                            # Get counts of malignant and benign nodules
                            malignant_count = sum(1 for n in nodules if n.get('malignancy') == 'Malignant')
                            benign_count = sum(1 for n in nodules if n.get('malignancy') == 'Benign')
                            unknown_count = nodule_count - malignant_count - benign_count
                            
                            # Get the highest confidence nodule
                            if nodules:
                                sorted_nodules = sorted(nodules, key=lambda x: x.get('confidence', 0), reverse=True)
                                top_nodule = sorted_nodules[0]
                                top_confidence = top_nodule.get('confidence', 0)
                                top_confidence_str = f"{top_confidence:.2f}" if top_confidence else "N/A"
                            else:
                                top_confidence_str = "N/A"
                            
                            app.config[session_key].update({
                                'nodule_count': nodule_count,
                                'malignant_count': malignant_count,
                                'benign_count': benign_count,
                                'unknown_count': unknown_count,
                                'top_confidence': top_confidence_str
                            })
                            
                            logger.info(f"Detected {nodule_count} nodules: {malignant_count} malignant, {benign_count} benign, {unknown_count} unknown")
                        else:
                            logger.info("No nodules detected")
                            app.config[session_key].update({
                                'nodule_count': 0,
                                'malignant_count': 0,
                                'benign_count': 0,
                                'unknown_count': 0,
                                'top_confidence': "N/A"
                            })
                        
                        # Additional access control - grant access to the user who uploaded the scan
                        access_db[job_id] = [current_user]
                        access_db[case_name] = [current_user]
                        logger.info(f"Granted access to {job_id} for user {current_user}")
                        
                        # Also copy the key result files to the main output directory for compatibility
                        try:
                            import shutil
                            
                            # Define source files in job directory
                            result_image = job_output_dir / f"{case_name}_results.png"
                            result_txt = job_output_dir / f"{case_name}_results.txt"
                            nodules_json = job_output_dir / f"{case_name}_nodules.json"
                            
                            # Define target files in main output directory using job_id
                            # This ensures unique names and links them to this specific job
                            main_output_dir = Path(app.config['OUTPUT_FOLDER'])
                            main_result_image = main_output_dir / f"{job_id}_results.png"
                            main_result_txt = main_output_dir / f"{job_id}_results.txt"
                            main_nodules_json = main_output_dir / f"{job_id}_nodules.json"
                            
                            # Copy the files if they exist
                            if result_image.exists():
                                shutil.copy2(result_image, main_result_image)
                                logger.info(f"Copied result image to main directory: {main_result_image}")
                            
                            if result_txt.exists():
                                shutil.copy2(result_txt, main_result_txt)
                                logger.info(f"Copied result text to main directory: {main_result_txt}")
                            
                            if nodules_json.exists():
                                shutil.copy2(nodules_json, main_nodules_json)
                                logger.info(f"Copied nodules JSON to main directory: {main_nodules_json}")
                        except Exception as e:
                            logger.error(f"Error copying result files to main directory: {e}")
                            # Continue even if copying fails
                        
                        # Update DICOM file access permissions after processing
                        try:
                            logger.info("Updating DICOM file access permissions...")
                            setup_dicom_access_permissions()
                        except Exception as e:
                            logger.error(f"Error updating DICOM access permissions: {e}")
                    
                    except Exception as e:
                        logger.error(f"Error in processing task: {e}")
                        logger.error(traceback.format_exc())
                        # Store error in session
                        app.config[session_key].update({
                            'status': 'error',
                            'error_message': f"Processing error: {str(e)}",
                            'file_path': mhd_path  # Use mhd_path instead of filepath
                        })
                        
                except Exception as e:
                    logger.error(f"Unhandled error in background processing task: {e}")
                    logger.error(traceback.format_exc())
                    # Make sure the session is updated even in case of unexpected errors
                    app.config[session_key].update({
                        'status': 'error',
                        'error_message': f"Unhandled error: {str(e)}"
                    })
                    
                finally:
                    # Clear GPU memory after processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared GPU memory after processing")
            
            # Start processing thread
            processing_thread = threading.Thread(target=process_scan_task)
            processing_thread.daemon = True
            processing_thread.start()
            
            return jsonify({
                'message': 'DICOM volume upload successful. Processing started.',
                'job_id': job_id
            }), 200
                
        except Exception as e:
            logger.error(f"Error processing DICOM files: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing DICOM files: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error uploading DICOM volume: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error uploading DICOM volume: {str(e)}'}), 500

def handle_zip_upload(zip_file, current_user):
    """Handle a zip file containing multiple DICOM files."""
    try:
        import zipfile
        import tempfile
        
        # Create a temporary directory for extraction
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        
        # Generate a user-specific prefix for the folder name
        user_prefix = f"{current_user}_zip_volume_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        dicom_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_prefix)
        os.makedirs(dicom_dir, exist_ok=True)
        
        # Save the zip file
        zip_path = os.path.join(temp_dir, secure_filename(zip_file.filename))
        zip_file.save(zip_path)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dicom_dir)
        
        # Delete the zip file
        os.remove(zip_path)
        
        # Find all DICOM files in the extracted directory
        dicom_files = []
        for root, dirs, files in os.walk(dicom_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if len(dicom_files) == 0:
            logger.error("No DICOM files found in the zip archive")
            return jsonify({'error': 'No DICOM files found in the zip archive'}), 400
        
        logger.info(f"Found {len(dicom_files)} DICOM files in the zip archive")
        
        # Now we need to convert the DICOM files to a 3D volume that our pipeline can process
        try:
            # Import SimpleITK for DICOM processing
            import SimpleITK as sitk
            
            # Create a DICOM series reader
            reader = sitk.ImageSeriesReader()
            
            # Get the DICOM series IDs
            series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
            
            if not series_IDs:
                logger.error("No DICOM series found in the uploaded files")
                return jsonify({'error': 'No valid DICOM series found in the uploaded files'}), 400
            
            # Use the first series ID
            series_ID = series_IDs[0]
            logger.info(f"Found DICOM series ID: {series_ID}")
            
            # Get the DICOM file names for this series
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ID)
            
            # Set the file names and load the DICOM series
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            # Save as a temporary MHD/RAW file that our pipeline can process
            output_basename = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_prefix}_volume")
            mhd_path = f"{output_basename}.mhd"
            raw_path = f"{output_basename}.raw"
            
            # Write the image
            sitk.WriteImage(image, mhd_path)
            logger.info(f"Successfully converted DICOM series to MHD/RAW: {mhd_path}")
            
            # Process the same way as the direct DICOM upload
            # Create a unique job ID
            job_id = f"zip_volume_{user_prefix}"
            logger.info(f"Starting processing with job ID: {job_id}")
            
            # Get processing options
            confidence_threshold = float(request.form.get('confidence', '0.5'))
            confidence_threshold = max(0.0, min(1.0, confidence_threshold))
            lungs_only = request.form.get('lungs_only') == 'true'
            fast_mode = request.form.get('fast_mode', 'true') == 'true'
            batch_size = 16
            
            # Prepare output directory
            output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process in a separate thread
            import threading
            
            def process_scan_task():
                try:
                    logger.info(f"Starting pipeline processing for job {job_id}")
                    
                    # Define temp result key
                    temp_result_key = f"temp_result_{job_id}"
                    
                    # Initialize the result in app.config if it doesn't exist
                    if temp_result_key not in app.config:
                        app.config[temp_result_key] = {
                            'status': 'processing',
                            'job_id': job_id,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'case_name': job_id,  # Set initial case_name to job_id for default
                            'nodule_count': 0     # Initialize with 0 nodules
                        }
                    
                    # Store processing info
                    processing_info = {
                        'status': 'processing',
                        'job_id': job_id,
                        'user': current_user,
                        'file': f"{user_prefix}_volume.mhd", 
                        'timestamp': datetime.datetime.now().isoformat(),
                        'confidence': confidence_threshold,
                        'lungs_only': lungs_only,
                        'patient_info': None
                    }
                    
                    # Save processing info
                    with open(os.path.join(output_dir, 'processing_info.json'), 'w') as f:
                        json.dump(processing_info, f, indent=2)
                    
                    # Process the scan
                    results = process_ct_scan(
                        mhd_path,
                        output_dir,
                        lungs_model,
                        None if lungs_only else nodules_model,
                        None if lungs_only else malignant_benign_model,
                        None,  # No cancer type model for now
                        confidence_threshold=confidence_threshold,
                        batch_size=batch_size,
                        fast_mode=fast_mode
                    )
                    
                    # Save the source file to the job output directory for easier access
                    try:
                        # Use the unified file saving helper
                        dest_file = save_file_to_unified_location(mhd_path, job_id, current_user)
                        if dest_file:
                            logger.info(f"Copied source file to job directory: {dest_file}")
                        else:
                            logger.warning(f"Failed to copy source file to job directory")
                    except Exception as e:
                        logger.error(f"Error copying source file to job directory: {e}")
                    
                    logger.info(f"Processing completed for job {job_id}")
                    
                    # Store the result in app.config for the preview endpoint to access
                    # Must match the format expected by the frontend
                    case_name = f"{job_id}"
                    
                    # Extract nodule information
                    nodules = results.get('nodules', [])
                    nodule_count = len(nodules)
                    malignant_count = sum(1 for n in nodules if n.get('malignancy') == 'Malignant')
                    
                    temp_result = {
                        'case_name': case_name,
                        'job_id': job_id,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'status': 'completed',
                        'nodule_count': nodule_count,
                        'malignant_count': malignant_count,
                        'processing_time': results.get('processing_time', 0),
                        'output_dir': output_dir
                    }
                    
                    # Store in app config for retrieval
                    app.config[temp_result_key] = temp_result
                    logger.info(f"Stored temporary result in app.config with key: {temp_result_key}")
                    
                    # Update DICOM file access permissions after processing
                    try:
                        logger.info("Updating DICOM file access permissions...")
                        setup_dicom_access_permissions()
                    except Exception as e:
                        logger.error(f"Error updating DICOM access permissions: {e}")
                    
                    # Clean up GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared GPU memory after processing scan")
                    
                except Exception as e:
                    logger.error(f"Error processing scan for job {job_id}: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Define temp result key if not done earlier
                    temp_result_key = f"temp_result_{job_id}"
                    
                    # Initialize or update the error status
                    if temp_result_key not in app.config:
                        app.config[temp_result_key] = {
                            'status': 'error',
                            'error_message': str(e),
                            'job_id': job_id,
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                    else:
                        app.config[temp_result_key].update({
                            'status': 'error',
                            'error_message': str(e)
                        })
                    
                    # Update processing info with error
                    processing_info = {
                        'status': 'error',
                        'job_id': job_id,
                        'user': current_user,
                        'file': f"{user_prefix}_volume.mhd",
                        'timestamp': datetime.datetime.now().isoformat(),
                        'error': str(e)
                    }
                    
                    # Save error info
                    with open(os.path.join(output_dir, 'processing_info.json'), 'w') as f:
                        json.dump(processing_info, f, indent=2)
                    
                    # Clear GPU memory after error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared GPU memory after processing error")
            
            # Start processing thread
            processing_thread = threading.Thread(target=process_scan_task)
            processing_thread.daemon = True
            processing_thread.start()
            
            return jsonify({
                'message': 'ZIP file upload successful. Processing started.',
                'job_id': job_id
            }), 200
                
        except Exception as e:
            logger.error(f"Error processing DICOM files from ZIP: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing DICOM files from ZIP: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error processing ZIP file: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing ZIP file: {str(e)}'}), 500

def adaptive_window_level(slice_data, default_window=1500, default_level=-600):
    """Apply adaptive window/level based on histogram analysis"""
    try:
        # Get histogram to analyze the intensity distribution
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
            
        return window, level
    except Exception as e:
        logger.error(f"Error in adaptive windowing: {e}")
        return default_window, default_level

def enhance_contrast(image, low_percentile=5, high_percentile=95):
    """Enhance contrast using percentile-based normalization"""
    try:
        # Get percentile values
        low = np.percentile(image, low_percentile)
        high = np.percentile(image, high_percentile)
        
        # Apply contrast stretching
        enhanced = np.clip(image, low, high)
        enhanced = ((enhanced - low) / (high - low) * 255).astype(np.uint8)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing contrast: {e}")
        return image.astype(np.uint8)

def setup_dicom_access_permissions():
    """
    Scan the results directory and build access_db mappings for DICOM files.
    This allows users to access DICOM files by their original filenames (e.g., 1-111)
    instead of just by job IDs.
    
    Also creates a mapping between DICOM job IDs and their corresponding MHD/RAW files.
    """
    global access_db
    
    logger.info("Setting up DICOM access permissions...")
    
    # Get all job directories in the results folder
    results_dir = Path(app.config['OUTPUT_FOLDER'])
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    job_dirs = [d for d in results_dir.iterdir() if d.is_dir() and 
                (d.name.startswith('dicom_volume_') or d.name.startswith('zip_volume_'))]
    
    # Count of mappings added
    mappings_added = 0
    mhd_mappings_added = 0
    
    # Create a mapping between job IDs and MHD files
    if 'mhd_mapping' not in app.config:
        app.config['mhd_mapping'] = {}
    
    for job_dir in job_dirs:
        try:
            # Extract username from job ID
            parts = job_dir.name.split('_')
            if len(parts) >= 3:
                username = parts[2]  # Format: dicom_volume_username_timestamp
                timestamp = parts[3] if len(parts) > 3 else None
                
                # Add the job to access_db to ensure the owner has access
                if job_dir.name not in access_db:
                    access_db[job_dir.name] = []
                if username not in access_db[job_dir.name]:
                    access_db[job_dir.name].append(username)
                
                # Find associated MHD file in the uploads directory
                # Format for MHD files created from DICOM: username_dicom_volume_timestamp_volume.mhd
                mhd_pattern = f"{username}_dicom_volume_{timestamp}_volume.mhd"
                mhd_files = list(upload_dir.glob(mhd_pattern))
                
                # If no exact match, try more flexible patterns
                if not mhd_files and timestamp:
                    mhd_pattern = f"{username}_dicom_volume_*_volume.mhd"
                    mhd_files = list(upload_dir.glob(mhd_pattern))
                
                # Try even more generic pattern as last resort
                if not mhd_files:
                    mhd_pattern = f"{username}_*volume*.mhd"
                    mhd_files = list(upload_dir.glob(mhd_pattern))
                
                # If an MHD file is found, map it to the job ID
                if mhd_files:
                    mhd_file = str(mhd_files[0])
                    app.config['mhd_mapping'][job_dir.name] = mhd_file
                    logger.info(f"Mapped job {job_dir.name} to MHD file: {mhd_file}")
                    mhd_mappings_added += 1
                else:
                    # Try to find a .mhd file inside the job directory itself
                    job_mhd_files = list(job_dir.glob("*.mhd"))
                    if job_mhd_files:
                        mhd_file = str(job_mhd_files[0])
                        app.config['mhd_mapping'][job_dir.name] = mhd_file
                        logger.info(f"Mapped job {job_dir.name} to internal MHD file: {mhd_file}")
                        mhd_mappings_added += 1
                
                # Find all files in this job directory
                all_files = list(job_dir.glob('*'))
                
                # Look for any files that could be DICOM files or slice files
                # This includes files with pattern like 1-111, 2-222, files with .dcm extension,
                # and any slice image files
                potential_dicom_files = []
                
                for f in all_files:
                    # Consider typical DICOM naming patterns
                    if ('-' in f.name or                       # Standard DICOM series naming (1-111)
                        f.name.endswith('.dcm') or             # DICOM file extension
                        f.stem.isdigit() or                     # Simple numeric file
                        (f.name.startswith('IM') and f.stem[2:].isdigit())   # IM0001 format
                    ):
                        potential_dicom_files.append(f)
                    # Also check for slice image outputs which indicate a valid DICOM
                    elif ('slice' in f.name.lower() and f.suffix in ['.png', '.jpg', '.jpeg']):
                        potential_dicom_files.append(f)
                
                # Add each potential DICOM file to access_db
                for dicom_file in potential_dicom_files:
                    # Use the base filename as the key
                    base_name = dicom_file.stem if dicom_file.suffix else dicom_file.name
                    
                    # Add to access_db
                    if base_name not in access_db:
                        access_db[base_name] = []
                    if username not in access_db[base_name]:
                        access_db[base_name].append(username)
                        mappings_added += 1
                        
                    # If this file is associated with an MHD file, store that mapping as well
                    if job_dir.name in app.config['mhd_mapping']:
                        mhd_file = app.config['mhd_mapping'][job_dir.name]
                        if 'dicom_to_mhd' not in app.config:
                            app.config['dicom_to_mhd'] = {}
                        app.config['dicom_to_mhd'][base_name] = mhd_file
                        
        except Exception as e:
            logger.error(f"Error processing job directory {job_dir}: {e}")
    
    # Log counts for debugging
    logger.info(f"DICOM access permission setup complete:")
    logger.info(f"  - Added {mappings_added} DICOM file mappings")
    logger.info(f"  - Added {mhd_mappings_added} MHD file mappings")
    
    # Print some example mappings for debugging
    if 'mhd_mapping' in app.config and app.config['mhd_mapping']:
        examples = list(app.config['mhd_mapping'].items())[:3]  # Show up to 3 examples
        logger.info(f"Example MHD mappings: {examples}")
    
    if 'dicom_to_mhd' in app.config and app.config['dicom_to_mhd']:
        examples = list(app.config['dicom_to_mhd'].items())[:3]  # Show up to 3 examples
        logger.info(f"Example DICOM to MHD mappings: {examples}")

# Initialize DICOM access permissions when starting the app
setup_dicom_access_permissions()

@app.route('/api/upload/mhd-raw', methods=['POST'])
@jwt_required()
def upload_mhd_raw():
    """Upload MHD/RAW file pair for processing."""
    try:
        # Get current user identity for tracking
        current_user = get_jwt_identity()
        logger.info(f"Starting MHD/RAW file upload process for user: {current_user}")
        
        # Check for MHD file
        if 'mhd_file' not in request.files:
            logger.error("No MHD file part in the request")
            return jsonify({'error': 'No MHD file part in the request'}), 400
            
        mhd_file = request.files['mhd_file']
        
        if mhd_file.filename == '':
            logger.error("No selected MHD file")
            return jsonify({'error': 'No selected MHD file'}), 400
            
        if not mhd_file.filename.lower().endswith('.mhd'):
            logger.error(f"File is not an MHD file: {mhd_file.filename}")
            return jsonify({'error': 'The first file must be an MHD file'}), 400
        
        # Check for RAW file
        if 'raw_file' not in request.files:
            logger.error("No RAW file part in the request")
            return jsonify({'error': 'No RAW file part in the request'}), 400
            
        raw_file = request.files['raw_file']
        
        if raw_file.filename == '':
            logger.error("No selected RAW file")
            return jsonify({'error': 'No selected RAW file'}), 400
            
        if not raw_file.filename.lower().endswith('.raw'):
            logger.error(f"File is not a RAW file: {raw_file.filename}")
            return jsonify({'error': 'The second file must be a RAW file'}), 400
        
        # Generate a user-specific prefix for the file names
        user_prefix = f"{current_user}_"
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Create a unique base name for both files
        base_name = secure_filename(f"{user_prefix}scan_{timestamp}")
        
        # Save the MHD file
        mhd_filename = f"{base_name}.mhd"
        mhd_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mhd_filename)
        logger.info(f"Saving MHD file: {mhd_filename}")
        mhd_file.save(mhd_filepath)
        
        # Save the RAW file
        raw_filename = f"{base_name}.raw"
        raw_filepath = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
        logger.info(f"Saving RAW file: {raw_filename}")
        raw_file.save(raw_filepath)
        
        # Update the MHD file to reference the RAW file with the correct name
        try:
            logger.info(f"Updating MHD file to reference RAW file: {raw_filename}")
            with open(mhd_filepath, 'r') as f:
                mhd_content = f.readlines()
            
            # Find and update the ElementDataFile line
            updated = False
            for i, line in enumerate(mhd_content):
                if line.lower().startswith('elementdatafile'):
                    # Replace with the new RAW filename
                    mhd_content[i] = f"ElementDataFile = {raw_filename}\n"
                    updated = True
                    break
            
            # If ElementDataFile line was not found, add it
            if not updated:
                mhd_content.append(f"ElementDataFile = {raw_filename}\n")
            
            # Write the updated MHD file
            with open(mhd_filepath, 'w') as f:
                f.writelines(mhd_content)
                
            logger.info(f"Successfully updated MHD file references")
        except Exception as e:
            logger.error(f"Error updating MHD file: {e}")
            return jsonify({'error': f'Error processing MHD file: {str(e)}'}), 500
        
        # Get processing options from form data
        try:
            confidence_str = request.form.get('confidence', '0.5')
            confidence_threshold = float(confidence_str)
            confidence_threshold = max(0.0, min(1.0, confidence_threshold))
            logger.info(f"Parsed confidence threshold: {confidence_threshold}")
        except ValueError as e:
            logger.error(f"Error parsing confidence threshold: {e}, using default 0.5")
            confidence_threshold = 0.5
        
        lungs_only = request.form.get('lungs_only') == 'true'
        fast_mode = request.form.get('fast_mode', 'true') == 'true'
        
        logger.info(f"Processing options: confidence={confidence_threshold}, lungs_only={lungs_only}, fast_mode={fast_mode}")
        
        # Create a unique job ID
        job_id = f"mhdraw_{base_name}"
        logger.info(f"Starting processing with job ID: {job_id}")
        
        # Determine batch size based on available GPU memory
        batch_size = 16  # Default batch size
        
        if torch.cuda.is_available():
            # Clear GPU memory before processing
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory before processing scan")
            
            # Check available VRAM and adjust batch size
            try:
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                free_memory_gb = free_memory / (1024**3)  # Convert to GB
                
                # Adjust batch size based on available memory
                if free_memory_gb > 8:
                    batch_size = 32
                elif free_memory_gb > 4:
                    batch_size = 24
                logger.info(f"Adjusted batch size to {batch_size} based on {free_memory_gb:.2f}GB available VRAM")
            except Exception as e:
                logger.warning(f"Could not check GPU memory: {e}, using default batch size")
        
        # Reset models to ensure no state is retained between scans
        if nodules_model is not None:
            nodules_model.eval()  # Ensure in evaluation mode
        if lungs_model is not None:
            lungs_model.eval()  # Ensure in evaluation mode
        
        # Process the scan with optimized parameters
        try:
            logger.info(f"Processing scan with confidence_threshold={confidence_threshold}, batch_size={batch_size}")
            
            # Double-check that the ElementDataFile path is correct before processing
            try:
                with open(mhd_filepath, 'r') as f:
                    mhd_content = f.read()
                
                # Find the ElementDataFile line
                elementdatafile_match = re.search(r'ElementDataFile\s*=\s*(.+)', mhd_content, re.IGNORECASE)
                
                if elementdatafile_match:
                    raw_path = elementdatafile_match.group(1).strip()
                    logger.info(f"Found ElementDataFile reference in MHD: {raw_path}")
                    
                    # If the RAW path is not exactly the raw_filename, update it
                    if raw_path != raw_filename:
                        logger.warning(f"ElementDataFile reference doesn't match our saved RAW file: {raw_path} vs {raw_filename}")
                        
                        # Update the MHD file to use our generated raw_filename
                        updated_content = re.sub(
                            r'(ElementDataFile\s*=\s*).+', 
                            f'\\1{raw_filename}', 
                            mhd_content, 
                            flags=re.IGNORECASE
                        )
                        
                        # Write the updated content back to the file
                        with open(mhd_filepath, 'w') as f:
                            f.write(updated_content)
                        
                        logger.info("Updated MHD file to use correct RAW reference")
            except Exception as e:
                logger.error(f"Error checking/fixing MHD file: {e}")
                # Continue processing as this is just an enhancement
             
            result = process_ct_scan(
                mhd_filepath,
                app.config['OUTPUT_FOLDER'],
                lungs_model=lungs_model,
                nodules_model=None if lungs_only else nodules_model,
                malignant_benign_model=malignant_benign_model,
                confidence_threshold=confidence_threshold,
                batch_size=batch_size,
                fast_mode=fast_mode
            )
            
            processing_time = result.get('processing_time', 0)
            logger.info(f"Processing completed successfully for job: {job_id} in {processing_time:.2f}s")
            
            # Extract the case_name from the result
            case_name = result.get('case_name', base_name)
            
            # Store the temporary result path in a session variable
            session_key = f"temp_result_{job_id}"
            app.config[session_key] = {
                'case_name': case_name,
                'job_id': job_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'file_path': mhd_filepath,
                'raw_file_path': raw_filepath,
                'expires_at': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat(),  # Results expire after 1 hour
                'nodule_count': len(result.get('nodules', [])),
                'is_preview': True,
                'processing_time': processing_time
            }
            
            # Create a preview URL
            preview_url = f"/api/results/preview/{job_id}"
            
            # Return the result for preview with clear instructions
            return jsonify({
                'success': True,
                'message': 'MHD/RAW files uploaded and processed successfully. Click the "Results Preview" button to review and save or discard.',
                'job_id': job_id,
                'mhd_filename': mhd_filename,
                'raw_filename': raw_filename,
                'processing_time': processing_time,
                'preview_url': preview_url,
                'result_path': preview_url
            }), 200
            
        except Exception as e:
            logger.error(f"Error processing MHD/RAW files: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f'Error processing MHD/RAW files: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in upload_mhd_raw: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred during MHD/RAW upload'}), 500

@app.route('/api/results/preview/image/<job_id>', methods=['GET'])
def get_preview_image(job_id):
    """Get the preview image for a specific job without requiring authentication."""
    try:
        # Check if the job_id exists in the temporary storage
        session_key = f"temp_result_{job_id}"
        if session_key not in app.config:
            logger.error(f"No temporary results found for job ID: {job_id}")
            return jsonify({'error': 'No temporary results found for this job ID'}), 404
        
        # Get the temporary result data
        temp_result = app.config[session_key]
        
        # Get case_name with a fallback to the job_id if not present
        case_name = temp_result.get('case_name', job_id)
        
        # Get result details from the job-specific subfolder
        job_output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        
        # Prioritize job-specific directory if it exists
        if job_output_dir.exists():
            # Look for result files in the job subfolder
            image_files = list(job_output_dir.glob("*_results.png"))
            if not image_files:
                image_files = list(job_output_dir.glob("*.png"))
                
            if image_files:
                image_file = image_files[0]
                logger.info(f"Serving preview image from job directory: {image_file}")
                return send_file(str(image_file), mimetype='image/png')
        
        # If no job directory or no image found, check the main output directory
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        
        # Look for result files in the main output directory with the case name
        image_files = list(output_dir.glob(f"{case_name}_results.png"))
        if not image_files:
            image_files = list(output_dir.glob(f"{case_name}*.png"))
            
        if image_files:
            image_file = image_files[0]
            logger.info(f"Serving preview image from main directory: {image_file}")
            return send_file(str(image_file), mimetype='image/png')
            
        # If still no image found, return a 404
        logger.error(f"No preview image found for job ID: {job_id} (case: {case_name})")
        return jsonify({'error': 'Preview image not found'}), 404
        
    except Exception as e:
        logger.error(f"Error getting preview image: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error getting preview image: {str(e)}'}), 500

def check_and_fix_mhd_file(mhd_path):
    """
    Check an MHD file for common issues and fix them if possible.
    
    Args:
        mhd_path: Path to the MHD file
        
    Returns:
        bool: True if the file is valid or was successfully fixed, False otherwise
    """
    try:
        mhd_path = Path(mhd_path)
        logger.info(f"Checking MHD file: {mhd_path}")
        
        if not mhd_path.exists():
            logger.error(f"MHD file does not exist: {mhd_path}")
            return False
            
        # Read the MHD file
        with open(mhd_path, 'r') as f:
            mhd_content = f.read()
            
        # Check for ElementDataFile reference
        elementdatafile_match = re.search(r'ElementDataFile\s*=\s*(.+)', mhd_content, re.IGNORECASE)
        if not elementdatafile_match:
            logger.error(f"No ElementDataFile reference found in {mhd_path}")
            # Try to infer the RAW filename and add it to the MHD file
            inferred_raw = mhd_path.with_suffix('.raw')
            if inferred_raw.exists():
                logger.info(f"Adding inferred ElementDataFile: {inferred_raw.name}")
                # Add ElementDataFile line at the end of the file
                with open(mhd_path, 'a') as f:
                    f.write(f"\nElementDataFile = {inferred_raw.name}\n")
                return True
            else:
                return False
            
        raw_filename = elementdatafile_match.group(1).strip()
        logger.info(f"Found RAW reference: {raw_filename}")
        
        # Possible locations for the RAW file
        raw_locations = [
            mhd_path.parent / raw_filename,  # Same directory with exact name
            mhd_path.parent / Path(raw_filename).name,  # Just the filename without path
            mhd_path.with_suffix('.raw'),  # Same name as MHD but with .raw extension
        ]
        
        # Add variations with lowercase/uppercase .raw extension
        base_name = mhd_path.stem
        raw_locations.extend([
            mhd_path.parent / f"{base_name}.raw",
            mhd_path.parent / f"{base_name}.RAW"
        ])
        
        # Check each possible location
        found_raw = None
        for raw_path in raw_locations:
            if raw_path.exists():
                found_raw = raw_path
                logger.info(f"Found RAW file at: {found_raw}")
                break
                
        if not found_raw:
            # If RAW not found, look in parent directories and subdirectories
            logger.warning(f"Could not find RAW file for {mhd_path} in immediate directory. Searching deeper...")
            
            # Check parent directories up to 2 levels
            parent_dir = mhd_path.parent
            for _ in range(2):
                if parent_dir.parent != parent_dir:  # Avoid infinite loop at root
                    parent_dir = parent_dir.parent
                    parent_raw = parent_dir / raw_filename
                    if parent_raw.exists():
                        found_raw = parent_raw
                        logger.info(f"Found RAW file in parent directory: {found_raw}")
                        break
            
            # If still not found, check subdirectories
            if not found_raw:
                for subdir in mhd_path.parent.glob('**/'):
                    if subdir != mhd_path.parent:  # Skip current directory as we've already checked it
                        subdir_raw = subdir / raw_filename
                        if subdir_raw.exists():
                            found_raw = subdir_raw
                            logger.info(f"Found RAW file in subdirectory: {found_raw}")
                            break
            
            # If still not found, look for any .raw file with similar name
            if not found_raw:
                for raw_path in mhd_path.parent.glob(f"{mhd_path.stem}*.raw"):
                    found_raw = raw_path
                    logger.info(f"Found RAW file with similar name: {found_raw}")
                    break
            
            if not found_raw:
                # If still not found, check if the raw file exists with a different case (uppercase/lowercase)
                for raw_path in mhd_path.parent.glob(f"{mhd_path.stem.lower()}*.raw") or mhd_path.parent.glob(f"{mhd_path.stem.upper()}*.raw"):
                    found_raw = raw_path
                    logger.info(f"Found RAW file with different case: {found_raw}")
                    break
            
            if not found_raw:
                logger.error(f"Could not find RAW file for {mhd_path} after extensive search")
                return False
            
        # Check if the reference in the MHD file needs to be updated
        # First verify if the found_raw path is relative or absolute
        found_raw_str = str(found_raw)
        mhd_dir_str = str(mhd_path.parent)
        
        # Update to use relative or absolute path based on current reference and what works
        if str(found_raw.name) != raw_filename:
            logger.info(f"Updating ElementDataFile reference from {raw_filename} to {found_raw.name}")
            
            # Update the reference in the MHD file to use the filename only
            updated_content = re.sub(
                r'(ElementDataFile\s*=\s*).+',
                r'\1' + found_raw.name,
                mhd_content,
                flags=re.IGNORECASE
            )
            
            # Write the updated content back to the MHD file
            with open(mhd_path, 'w') as f:
                f.write(updated_content)
            
            # Verify that the RAW file is accessible with this reference by doing a test read
            try:
                logger.info("Testing MHD file after update to ensure RAW file is accessible")
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(mhd_path))
                # Only read metadata to verify file is readable without loading full image
                reader.ReadImageInformation()
                logger.info("MHD file verified - RAW file is accessible")
            except Exception as e:
                logger.warning(f"MHD reference still not working after update: {e}")
                
                # Try absolute path as a fallback
                try:
                    logger.info(f"Trying absolute path to RAW file: {found_raw_str}")
                    updated_content = re.sub(
                        r'(ElementDataFile\s*=\s*).+',
                        r'\1' + found_raw_str.replace('\\', '/'),  # SimpleITK prefers forward slashes
                        mhd_content,
                        flags=re.IGNORECASE
                    )
                    
                    # Write the updated content with absolute path
                    with open(mhd_path, 'w') as f:
                        f.write(updated_content)
                        
                    # Test again
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(str(mhd_path))
                    reader.ReadImageInformation()
                    logger.info("MHD file verified with absolute path - RAW file is accessible")
                except Exception as e2:
                    logger.error(f"Failed to fix MHD file even with absolute path: {e2}")
                    return False
        
        # Verify ElementType is present (required by SimpleITK)
        if "ElementType" not in mhd_content:
            logger.warning("ElementType missing from MHD file - adding default (MET_SHORT)")
            with open(mhd_path, 'a') as f:
                f.write("\nElementType = MET_SHORT\n")
        
        return True
            
    except Exception as e:
        logger.error(f"Error checking/fixing MHD file {mhd_path}: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/api/diagnostic/check-mhd-files', methods=['GET'])
def diagnostic_check_mhd_files():
    """Check all MHD files in the uploads directory and report their status."""
    try:
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        results = []
        
        # Find all MHD files in the uploads directory
        mhd_files = list(upload_dir.glob('*.mhd'))
        
        for mhd_path in mhd_files:
            mhd_info = {
                'mhd_file': str(mhd_path),
                'exists': mhd_path.exists(),
                'size': mhd_path.stat().st_size if mhd_path.exists() else 0,
                'raw_file': None,
                'raw_exists': False,
                'raw_size': 0,
                'status': 'unknown',
                'error': None
            }
            
            try:
                # Read the MHD file content
                with open(mhd_path, 'r') as f:
                    mhd_content = f.read()
                
                # Check for ElementDataFile reference
                elementdatafile_match = re.search(r'ElementDataFile\s*=\s*(.+)', mhd_content, re.IGNORECASE)
                if not elementdatafile_match:
                    mhd_info['error'] = "No ElementDataFile reference found"
                    mhd_info['status'] = 'error'
                else:
                    raw_filename = elementdatafile_match.group(1).strip()
                    mhd_info['raw_file'] = raw_filename
                    
                    # Check if the RAW file exists
                    raw_path = mhd_path.parent / raw_filename
                    if raw_path.exists():
                        mhd_info['raw_exists'] = True
                        mhd_info['raw_size'] = raw_path.stat().st_size
                        mhd_info['status'] = 'ok'
                    else:
                        # Try alternative paths
                        alt_raw_path = mhd_path.with_suffix('.raw')
                        if alt_raw_path.exists():
                            mhd_info['raw_exists'] = True
                            mhd_info['raw_size'] = alt_raw_path.stat().st_size
                            mhd_info['status'] = 'ok_alt_path'
                            mhd_info['error'] = f"RAW file found at alternative path: {alt_raw_path}"
                        else:
                            mhd_info['status'] = 'error'
                            mhd_info['error'] = f"RAW file not found: {raw_path}"
                
                # Try to load the MHD file with SimpleITK to verify it works
                try:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(str(mhd_path))
                    reader.ReadImageInformation()
                    mhd_info['can_read_metadata'] = True
                    
                    # Try to load the actual image data
                    image = reader.Execute()
                    volume = sitk.GetArrayFromImage(image)
                    mhd_info['can_read_image'] = True
                    mhd_info['volume_shape'] = list(volume.shape)
                    mhd_info['status'] = 'fully_readable'
                except Exception as e:
                    mhd_info['can_read_metadata'] = False
                    mhd_info['can_read_image'] = False
                    mhd_info['status'] = 'not_readable'
                    mhd_info['error'] = f"Failed to read with SimpleITK: {str(e)}"
                
            except Exception as e:
                mhd_info['status'] = 'error'
                mhd_info['error'] = str(e)
            
            results.append(mhd_info)
        
        # Check if we found any MHD files
        if not results:
            return jsonify({
                'status': 'warning',
                'message': 'No MHD files found in uploads directory',
                'upload_dir': str(upload_dir),
                'files_in_dir': [str(f) for f in upload_dir.glob('*') if f.is_file()]
            })
        
        return jsonify({
            'status': 'success',
            'mhd_files': results,
            'upload_dir': str(upload_dir),
            'total_files': len(results),
            'readable_files': sum(1 for r in results if r['status'] == 'fully_readable')
        })
    
    except Exception as e:
        logger.error(f"Error in diagnostic endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Helper function to load nodules data for a case
def load_nodules_for_case(case_name):
    """Load nodules data for a case from results files or database."""
    try:
        # Define places to look for nodule data
        results_dir = Path(app.config['OUTPUT_FOLDER'])
        
        # First try to find nodules in case-specific results file
        nodules_path = results_dir / f"{case_name}_nodules.json"
        
        # If no specific nodules file exists, check results file
        if not nodules_path.exists():
            # Look for results.json file
            results_path = results_dir / f"{case_name}_results.json"
            if results_path.exists():
                nodules_path = results_path
            else:
                # Also check in case directory
                case_dir = results_dir / case_name
                if case_dir.exists():
                    for path in case_dir.glob("*_nodules.json"):
                        nodules_path = path
                        break
                    
                    if not nodules_path.exists():
                        for path in case_dir.glob("*_results.json"):
                            nodules_path = path
                            break
        
        # Load nodules data from file if it exists
        nodules = []
        if nodules_path.exists():
            try:
                with open(nodules_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract nodules array from data structure
                if isinstance(data, dict):
                    nodules = data.get('nodules', [])
                elif isinstance(data, list):
                    nodules = data
                else:
                    nodules = []
                
                # Make sure each nodule has required fields
                for nodule in nodules:
                    # Generate ID if not present
                    if 'id' not in nodule:
                        nodule['id'] = str(uuid.uuid4())
                    
                    # Special handling for coordinates
                    if 'coordinates' in nodule and isinstance(nodule['coordinates'], list) and len(nodule['coordinates']) >= 3:
                        # Handle case where coordinates are stored as [z, y, x] array
                        coords = nodule['coordinates']
                        nodule['z'] = int(coords[0])
                        nodule['y'] = int(coords[1])
                        nodule['x'] = int(coords[2])
                    else:
                        # Set default coordinates if not present
                        for coord in ['x', 'y', 'z']:
                            if coord not in nodule:
                                nodule[coord] = 0
                            
                    # Set default confidence and radius
                    if 'confidence' not in nodule:
                        nodule['confidence'] = 0.5
                    if 'radius' not in nodule and 'radius_mm' in nodule:
                        nodule['radius'] = nodule['radius_mm']
                    elif 'radius' not in nodule:
                        nodule['radius'] = 5
                
                logger.info(f"Loaded {len(nodules)} nodules for case {case_name} from {nodules_path}")
                return nodules
            except Exception as e:
                logger.error(f"Error loading nodules from {nodules_path}: {e}")
                # Fall through to other methods
        
        # Parse nodules from the text results file
        nodules_from_text = parse_nodules_from_text(case_name)
        if nodules_from_text:
            return nodules_from_text
            
        # If we got this far but still have no nodules, try to check if there's MHD file
        # in uploads that matches this case name - often nodules are stored in the volume metadata
        if not nodules:
            try:
                upload_dir = Path(app.config['UPLOAD_FOLDER'])
                mhd_file = upload_dir / f"{case_name}.mhd"
                
                if mhd_file.exists():
                    logger.info(f"Checking if MHD file contains nodule data: {mhd_file}")
                    
                    # Try to check for metadata in MHD file that might indicate nodules
                    try:
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(str(mhd_file))
                        reader.ReadImageInformation()
                        
                        # Check if this MHD has any nodule annotations
                        if reader.HasMetaDataKey('NumberOfNodules'):
                            num_nodules = int(reader.GetMetaData('NumberOfNodules'))
                            logger.info(f"Found {num_nodules} nodules in MHD metadata")
                            
                            # Extract nodule data from metadata
                            for i in range(num_nodules):
                                try:
                                    prefix = f"Nodule{i+1}"
                                    coords_key = f"{prefix}Coords"
                                    
                                    if reader.HasMetaDataKey(coords_key):
                                        coords_str = reader.GetMetaData(coords_key)
                                        coords = [float(c) for c in coords_str.split(',')]
                                        
                                        radius_key = f"{prefix}Radius"
                                        radius = float(reader.GetMetaData(radius_key)) if reader.HasMetaDataKey(radius_key) else 5.0
                                        
                                        nodules.append({
                                            'id': f"Nodule {i+1}",
                                            'x': int(coords[0]),
                                            'y': int(coords[1]),
                                            'z': int(coords[2]),
                                            'radius': radius,
                                            'confidence': 0.9,
                                            'malignancy': "Unknown"
                                        })
                                except Exception as nodule_err:
                                    logger.warning(f"Error extracting nodule {i} from MHD: {nodule_err}")
                    except Exception as e:
                        logger.warning(f"Error checking MHD for nodule data: {e}")
            except Exception as mhd_err:
                logger.warning(f"Error checking for MHD file: {mhd_err}")
        
        # If no nodules found yet and in debug mode, create sample nodules
        if not nodules and app.debug:
            # For testing only - create sample nodules in debug mode
            logger.info(f"No nodules file found for {case_name}, creating 2 sample nodules for testing")
            return [
                {
                    'id': f"Nodule 1",
                    'x': 256,  # Assuming standard 512x512 CT size
                    'y': 256,
                    'z': 63,   # Middle slice for visibility
                    'radius': 15,
                    'confidence': 0.9,
                    'malignancy': "Benign"
                },
                {
                    'id': f"Nodule 2",
                    'x': 170,
                    'y': 170,
                    'z': 55,
                    'radius': 10,
                    'confidence': 0.7,
                    'malignancy': "Unknown"
                }
            ]
        
        return nodules
    except Exception as e:
        logger.error(f"Error loading nodules for case {case_name}: {e}")
        return []

def parse_nodules_from_text(case_name):
    """Extract nodule information from the results text file if JSON is not available."""
    try:
        results_dir = Path(app.config['OUTPUT_FOLDER'])
        
        # Look for the text results file
        results_path = results_dir / f"{case_name}_results.txt"
        
        if not results_path.exists():
            # Try in case directory
            case_dir = results_dir / case_name
            if case_dir.exists():
                for path in case_dir.glob("*_results.txt"):
                    results_path = path
                    break
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                content = f.read()
                
            # Parse the content
            nodules = []
            current_nodule = None
            
            for line in content.split('\n'):
                if line.startswith('Nodule '):
                    if current_nodule is not None:
                        nodules.append(current_nodule)
                    
                    nodule_id = line.replace(':', '').strip()
                    current_nodule = {
                        'id': nodule_id,
                        'x': 0,
                        'y': 0,
                        'z': 0,
                        'radius': 5,  # Default radius
                        'confidence': 0.5  # Default confidence
                    }
                elif current_nodule is not None:
                    if 'Coordinates' in line:
                        # Extract coordinates from format like "(45, 256, 189)"
                        coords_match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', line)
                        if coords_match:
                            current_nodule['z'] = int(coords_match.group(1))
                            current_nodule['y'] = int(coords_match.group(2))
                            current_nodule['x'] = int(coords_match.group(3))
                    elif 'Radius' in line:
                        # Extract radius value
                        radius_match = re.search(r'Radius:\s*(\d+\.?\d*)', line)
                        if radius_match:
                            current_nodule['radius'] = float(radius_match.group(1))
                    elif 'Confidence' in line and not 'Cancer Type Confidence' in line:
                        # Extract confidence value
                        conf_match = re.search(r'Confidence:\s*(\d+\.?\d*)', line)
                        if conf_match:
                            current_nodule['confidence'] = float(conf_match.group(1))
                    elif 'Malignancy:' in line and not 'Score' in line:
                        # Extract malignancy classification
                        malignancy = line.split(':')[1].strip()
                        current_nodule['malignancy'] = malignancy
            
            # Add the last nodule
            if current_nodule is not None:
                nodules.append(current_nodule)
                
            if nodules:
                logger.info(f"Extracted {len(nodules)} nodules from text file for case {case_name}")
            
            return nodules
    except Exception as e:
        logger.error(f"Error parsing nodules from text for case {case_name}: {e}")
    
    return []

@app.route('/api/setup/initialize', methods=['POST'])
def initialize_database():
    """Initialize the database with a superadmin user if no users exist."""
    try:
        # Check if any users exist
        if users_db:
            return jsonify({'message': 'Database is already initialized with users'}), 400
        
        # Create users directly in the dictionary
        users_db['Pulmoscan'] = {
            'password': generate_password_hash('Pulmoscan123'),
            'role': 'superadmin',
            'email': 'admin@pulmoscan.com',
            'description': 'Initial superadmin account',
            'plan': 'subscription',
            'created_at': datetime.datetime.now().isoformat()
        }
        
        users_db['admin'] = {
            'password': generate_password_hash('admin123'),
            'role': 'admin',
            'email': 'admin@example.com',
            'description': 'Admin account',
            'plan': 'subscription',
            'created_at': datetime.datetime.now().isoformat()
        }
        
        users_db['doctor'] = {
            'password': generate_password_hash('doctor123'),
            'role': 'doctor',
            'email': 'doctor@example.com',
            'description': 'Doctor account',
            'plan': 'usage_based',
            'created_at': datetime.datetime.now().isoformat()
        }
        
        return jsonify({
            'message': 'Database initialized with default users',
            'users': [
                {'username': 'Pulmoscan', 'role': 'superadmin', 'password': 'Pulmoscan123'},
                {'username': 'admin', 'role': 'admin', 'password': 'admin123'},
                {'username': 'doctor', 'role': 'doctor', 'password': 'doctor123'}
            ]
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a new route for the landing page
@app.route('/landing-page')
def landing_page():
    """Serve the landing page."""
    return send_from_directory(app.static_folder, 'landing.html')

# Serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Load models before starting the app
    try:
        # Set PyTorch to non-deterministic mode to avoid identical outputs for different inputs
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            logger.info("Set PyTorch to non-deterministic mode for better performance and variability")
        
        # Use a random seed based on time to avoid deterministic behavior
        import random
        import time
        random_seed = int(time.time()) % 10000
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        logger.info(f"Set random seed to {random_seed} based on current time")
        
        # Load models
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        print(f"Error: Failed to load models: {e}")
        sys.exit(1)
        
    app.run(debug=True, host='0.0.0.0', port=3000) 