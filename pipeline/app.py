import os
import logging
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import datetime
import json
import sys
import base64
from pathlib import Path
import traceback
from io import BytesIO

# Set matplotlib backend to non-interactive 'Agg' to avoid "main thread is not in main loop" errors
import matplotlib
matplotlib.use('Agg')

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
app.config['JWT_SECRET_KEY'] = 'super-secret-key'  # Change this in production!
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=1)

# Setup JWT
jwt = JWTManager(app)

# Create upload and results directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Mock user database (replace with a real database in production)
users_db = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'role': 'admin'
    },
    'doctor': {
        'password': generate_password_hash('doctor123'),
        'role': 'doctor'
    }
}

# Models
lungs_model = None
nodules_model = None
nodules_seg_model = None
malignant_benign_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
            'email': users_db[username].get('email', '')
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
                            return jsonify({'error': 'MHD file requires an associated RAW file'}), 400
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
                    'expires_at': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat()  # Results expire after 1 hour
                }
                
                # Return the result for preview, but not yet permanently saved
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and processed successfully. Review to save or discard.',
                    'job_id': job_id,
                    'filename': filename,
                    'processing_time': processing_time,
                    'result_path': f"/api/results/preview/{job_id}"
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
@jwt_required()
def get_result(case_name):
    """Get or delete a specific case result."""
    # Get current user identity and role for permission check
    current_user = get_jwt_identity()
    user_role = users_db.get(current_user, {}).get('role', '')
    
    # Get current user's previous usernames for continuity
    previous_usernames = users_db.get(current_user, {}).get('previous_usernames', [])
    
    # Check if user has permission to access this result
    # Admins can access all results, others can only access their own or from previous usernames
    can_access = (user_role == 'admin' or 
                 case_name.startswith(current_user) or 
                 any(case_name.startswith(prev_username) for prev_username in previous_usernames))
    
    if not can_access:
        logger.warning(f"Unauthorized access attempt to result {case_name} by user {current_user}")
        return jsonify({'error': 'Unauthorized access. You can only view or delete your own results.'}), 403
    
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
    
    # GET request handling - check for results with the job ID format
    
    # Check if result image exists
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
    
    # If still no image found, return 404
    if not image_path.exists():
        logger.warning(f"No result image found for case: {case_name}")
        return jsonify({'error': 'Result not found'}), 404
    
    # Check if result details exist
    details_path = result_dir / f"{case_name}_results.txt"
    details = None
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
                logger.info(f"Loaded patient information for case: {case_name}")
        except Exception as e:
            logger.error(f"Error loading patient information for case {case_name}: {e}")
            # Continue without patient info if there's an error
    
    # Prepare basic response data
    response_data = {
        'case_name': case_name,
        'image_url': f"/api/results/{case_name}/image",
        'timestamp': datetime.datetime.fromtimestamp(image_path.stat().st_mtime).isoformat()
    }
    
    # Add details if available
    if details:
        response_data['details'] = details
    else:
        response_data['details'] = 'No details available'
    
    # Add patient info if available
    if patient_info:
        response_data['patient_info'] = patient_info
    
    return jsonify(response_data), 200

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
@jwt_required()
def get_result_details(case_name):
    """Get the result details for a specific case."""
    # Get current user identity and role for permission check
    current_user = get_jwt_identity()
    user_role = users_db.get(current_user, {}).get('role', '')
    
    # Get current user's previous usernames for continuity
    previous_usernames = users_db.get(current_user, {}).get('previous_usernames', [])
    
    # Handle DICOM volume job IDs specially
    if case_name.startswith(('dicom_volume_', 'zip_volume_')):
        # Extract the username from the job ID format: {prefix}_{username}_something_timestamp
        parts = case_name.split('_')
        if len(parts) >= 4:  # At minimum: dicom/zip_volume_username_timestamp
            username_from_job = parts[2]  # The username should be the 3rd part
            logger.info(f"Extracted username '{username_from_job}' from job ID {case_name} for details")
            
            # Check permissions based on the extracted username
            if user_role != 'admin' and current_user != username_from_job:
                logger.warning(f"Unauthorized access attempt to details for job {case_name} by user {current_user}")
                return jsonify({'error': 'Unauthorized access. You can only view your own results.'}), 403
    else:
        # Regular case name check with standard permission checks
        # Admins can access all results, others can only access their own or from previous usernames
        can_access = (user_role == 'admin' or 
                    case_name.startswith(current_user) or 
                    any(case_name.startswith(prev_username) for prev_username in previous_usernames))
        
        if not can_access:
            logger.warning(f"Unauthorized access attempt to result details {case_name} by user {current_user}")
            return jsonify({'error': 'Unauthorized access. You can only view your own results.'}), 403
    
    # Check if this is a DICOM volume job
    is_dicom_volume = case_name.startswith(('dicom_volume_', 'zip_volume_'))
    
    logger.info(f"Fetching details for case: {case_name}")
    result_dir = Path(app.config['OUTPUT_FOLDER'])
    
    # For DICOM volume jobs, look in the job-specific subfolder
    if is_dicom_volume:
        job_dir = result_dir / case_name
        if job_dir.exists():
            logger.info(f"Using job-specific directory for details: {job_dir}")
            
            # Look for details files in the job directory first
            possible_filenames = list(job_dir.glob("*_results.txt"))
            if possible_filenames:
                found_file = possible_filenames[0]
                logger.info(f"Found details file in job directory: {found_file}")
                
                try:
                    # Read the details file
                    with open(found_file, 'r') as f:
                        details = f.read()
                
                    # Return the details
                    return jsonify({'details': details}), 200
                except Exception as e:
                    logger.error(f"Error reading details file for job {case_name}: {e}")
                    return jsonify({
                        'error': 'Error reading details file',
                        'details': f"An error occurred while reading the details for job: {case_name}"
                    }), 500
    
    # Continue with traditional file structure if not a job ID or if no file found in job directory
    # Check if case_name has timestamp format (user_sometext_YYYYMMDDHHMMSS)
    # First try with the original case name, then with extracted base name if needed
    original_case_name = case_name
    base_case_name = None
    
    parts = case_name.split('_')
    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 14:
        # Remove timestamp part and reconstruct base name
        base_case_name = '_'.join(parts[:-1])
        logger.info(f"Extracted base case name: {base_case_name}")
    
    # Try different possible filenames for the details text file
    possible_filenames = [
        f"{case_name}_results.txt",  # Standard results file
        f"{case_name}_nodules.txt",  # Alternative file used by some versions
        f"{case_name}.txt"           # Simplified filename
    ]
    
    # If we have a base case name, add those possibilities too
    if base_case_name:
        possible_filenames.extend([
            f"{base_case_name}_results.txt",
            f"{base_case_name}_nodules.txt",
            f"{base_case_name}.txt"
        ])
    
    found_file = None
    for filename in possible_filenames:
        file_path = result_dir / filename
        if file_path.exists():
            found_file = file_path
            # If we found a file with the base case name, update case_name
            if base_case_name and base_case_name in str(file_path):
                case_name = base_case_name
            break
    
    if not found_file:
        logger.warning(f"No details file found for case {case_name}. Checking for image file to return default message.")
        
        # Check if at least the image file exists with either case name format
        image_paths = [
            result_dir / f"{original_case_name}_results.png"
        ]
        
        if base_case_name:
            image_paths.append(result_dir / f"{base_case_name}_results.png")
        
        found_image = False
        for image_path in image_paths:
            if image_path.exists():
                found_image = True
                # If we found an image with the base case name, update case_name
                if base_case_name and base_case_name in str(image_path):
                    case_name = base_case_name
                break
                
        if found_image:
            # Return a default message since the image exists but no details file
            return jsonify({
                'details': f"Results for case: {case_name}\n\nNo detailed information available for this scan."
            }), 200
        else:
            logger.error(f"No results found for case {case_name}")
            return jsonify({
                'error': 'Results not found',
                'details': f"No results found for case: {case_name}"
            }), 404
    
    try:
        # Read the details file
        with open(found_file, 'r') as f:
            details = f.read()
    
        # Return the details
        return jsonify({'details': details}), 200
    except Exception as e:
        logger.error(f"Error reading details file for case {case_name}: {e}")
        return jsonify({
            'error': 'Error reading details file',
            'details': f"An error occurred while reading the details for case: {case_name}"
        }), 500

# Add API endpoint for interactive slice visualization
@app.route('/api/results/<case_name>/slices', methods=['GET'])
@jwt_required()
def get_result_slices(case_name):
    """Get interactive slice data for visualization."""
    try:
        # Get current user identity and role for permission check
        current_user = get_jwt_identity()
        user_role = users_db.get(current_user, {}).get('role', '')
        
        # Handle DICOM volume job IDs specially
        if case_name.startswith(('dicom_volume_', 'zip_volume_')):
            # Extract the username from the job ID format: {prefix}_{username}_something_timestamp
            parts = case_name.split('_')
            if len(parts) >= 4:  # At minimum: dicom/zip_volume_username_timestamp
                username_from_job = parts[2]  # The username should be the 3rd part
                logger.info(f"Extracted username '{username_from_job}' from job ID {case_name}")
                
                # Check permissions based on the extracted username
                if user_role != 'admin' and current_user != username_from_job:
                    logger.warning(f"Unauthorized access attempt to job slices for {case_name} by user {current_user}")
                    return jsonify({'error': 'Unauthorized access. You can only view your own results.'}), 403
        else:
            # Regular case name check
            if user_role != 'admin' and not case_name.startswith(current_user):
                logger.warning(f"Unauthorized access attempt to slices for {case_name} by user {current_user}")
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
        
        # For DICOM volume jobs, look in the job-specific subfolder
        if is_dicom_volume:
            job_dir = results_dir / case_name
            if job_dir.exists():
                logger.info(f"Using job-specific directory for slices: {job_dir}")
                results_dir = job_dir
        
        # Try different possible source files
        source_files = []
        
        # For DICOM volume jobs, try to find the source MHD file
        if is_dicom_volume:
            # Extract the prefix from the job ID (e.g., 'doctor_dicom_volume_20250517124451')
            # Assuming format is 'dicom_volume_username_dicom_volume_timestamp'
            parts = case_name.split('_')
            if len(parts) >= 5:  # At minimum: dicom_volume_username_dicom_volume_timestamp
                # Reconstruct MHD filename pattern
                mhd_prefix = '_'.join(parts[2:])  # e.g., 'username_dicom_volume_timestamp'
                
                # Look for MHD files in uploads
                mhd_patterns = [
                    f"{mhd_prefix}_volume.mhd",
                    f"{mhd_prefix}*.mhd",
                    f"*{parts[-1]}_volume.mhd"  # Match by timestamp
                ]
                
                for pattern in mhd_patterns:
                    matching_files = list(upload_dir.glob(pattern))
                    if matching_files:
                        source_files.extend(matching_files)
                        logger.info(f"Found DICOM volume source files: {matching_files}")
                        break
        
        # If no source files found yet, try traditional patterns
        if not source_files:
            # Try with original case name
            for ext in ['.mhd', '.nii.gz', '.nii', '.dcm']:
                source_files.extend(list(upload_dir.glob(f"{original_case_name}*{ext}")))
            
            # If we have a base case name and didn't find files, also try with that
            if base_case_name and not source_files:
                for ext in ['.mhd', '.nii.gz', '.nii', '.dcm']:
                    source_files.extend(list(upload_dir.glob(f"{base_case_name}*{ext}")))
                    
                # If we found files with the base name, update case_name
                if source_files:
                    case_name = base_case_name
        
        # Get nodule information from results
        nodule_data = []
        possible_detail_files = []
        
        # For DICOM volume jobs, look in the job directory first
        if is_dicom_volume and 'job_dir' in locals() and job_dir.exists():
            # First check in the job directory
            possible_detail_files.extend([
                job_dir / f"{original_case_name}_results.txt",
                job_dir / f"{original_case_name}_nodules.txt",
                job_dir / f"*_results.txt",  # Wildcard pattern for any results
                job_dir / f"*_nodules.txt"   # Wildcard pattern for any nodules
            ])
            
            # Use glob to expand the wildcard patterns
            for pattern in list(possible_detail_files):
                if '*' in str(pattern):
                    # This is a wildcard pattern, use glob to find matching files
                    matching_files = list(job_dir.glob(pattern.name))
                    if matching_files:
                        # Remove the pattern and add the actual files
                        possible_detail_files.remove(pattern)
                        possible_detail_files.extend(matching_files)
        else:
            # Traditional case - look in main results directory
            possible_detail_files = [
                results_dir / f"{original_case_name}_results.txt",
                results_dir / f"{original_case_name}_nodules.txt",
                results_dir / f"{original_case_name}.txt"
            ]
            
            # If we have a base case name, also check those files
            if base_case_name:
                possible_detail_files.extend([
                    results_dir / f"{base_case_name}_results.txt",
                    results_dir / f"{base_case_name}_nodules.txt",
                    results_dir / f"{base_case_name}.txt"
                ])
        
        details_file = next((f for f in possible_detail_files if f.exists()), None)
        
        if details_file:
            with open(details_file, 'r') as f:
                details = f.read()
                
            # Parse nodule data from the details
            lines = details.split('\n')
            current_nodule = None
            
            for line in lines:
                if line.startswith('Nodule '):
                    if current_nodule:
                        nodule_data.append(current_nodule)
                    current_nodule = {'id': line.replace(':', '').strip()}
                elif current_nodule:
                    if line.strip():
                        if 'Coordinates' in line:
                            coords_str = line.split(':')[1].strip()
                            # Convert from string to actual coordinates
                            try:
                                coords = [float(c.strip()) for c in coords_str.split(',')]
                                if len(coords) == 3:
                                    current_nodule['z'] = coords[0]
                                    current_nodule['y'] = coords[1]
                                    current_nodule['x'] = coords[2]
                            except:
                                logger.warning(f"Could not parse coordinates: {coords_str}")
                        elif 'Radius' in line:
                            try:
                                current_nodule['radius'] = float(line.split(':')[1].strip().split()[0])
                            except:
                                logger.warning(f"Could not parse radius: {line}")
                        elif 'Confidence' in line:
                            try:
                                current_nodule['confidence'] = float(line.split(':')[1].strip())
                            except:
                                logger.warning(f"Could not parse confidence: {line}")
            
            # Add the last nodule if present
            if current_nodule:
                nodule_data.append(current_nodule)
        
        # If we have source files, try to read the actual image data
        if source_files:
            source_file = source_files[0]
            
            # Try to import SimpleITK for medical image processing
            try:
                import SimpleITK as sitk
                import numpy as np
                
                # Read the image file
                image = sitk.ReadImage(str(source_file))
                volume = sitk.GetArrayFromImage(image)
                spacing = image.GetSpacing()
                
                # Get image dimensions
                depth, height, width = volume.shape
                
                # Normalize for web display (0-255)
                min_val = np.min(volume)
                max_val = np.max(volume)
                
                # Only send metadata for the volume, not the actual pixel data
                # The actual slices will be fetched individually
                volume_info = {
                    'dimensions': {
                        'depth': depth,
                        'height': height,
                        'width': width
                    },
                    'spacing': spacing,
                    'intensityRange': {
                        'min': float(min_val),
                        'max': float(max_val)
                    },
                    'nodules': nodule_data
                }
                
                return jsonify({
                    'case_name': case_name,
                    'volume_info': volume_info,
                    'source_file': source_file.name,
                    'using_placeholder': False
                }), 200
            
            except ImportError:
                logger.error("SimpleITK not available for image processing")
                return jsonify({
                    'error': 'Image processing library not available',
                    'message': 'The server cannot process the medical images'
                }), 500
            except Exception as e:
                logger.error(f"Error reading image file: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'error': 'Failed to process image file',
                    'message': str(e)
                }), 500
        else:
            # No source files found, return a standard placeholder volume info
            logger.warning(f"No source files found for case {case_name}, using placeholder volume info")
            
            # Default values for a standard CT scan
            default_depth = 100
            default_height = 512
            default_width = 512
            default_spacing = (1.0, 1.0, 1.0)  # Standard 1mm spacing
            
            volume_info = {
                'dimensions': {
                    'depth': default_depth,
                    'height': default_height,
                    'width': default_width
                },
                'spacing': default_spacing,
                'intensityRange': {
                    'min': -1000.0,  # Standard CT HU range
                    'max': 1000.0
                },
                'nodules': nodule_data
            }
            
            return jsonify({
                'case_name': case_name,
                'volume_info': volume_info,
                'using_placeholder': True,
                'message': 'Source CT data not found, using placeholder images'
            }), 200
    
    except Exception as e:
        logger.error(f"Error getting slice data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An unexpected error occurred',
            'message': str(e)
        }), 500

# Add endpoint to get a specific slice
@app.route('/api/results/<case_name>/slices/<axis>/<int:index>', methods=['GET'])
@jwt_required()
def get_slice(case_name, axis, index):
    """Get a specific slice from the volume."""
    try:
        # Get current user identity and role for permission check
        current_user = get_jwt_identity()
        user_role = users_db.get(current_user, {}).get('role', '')
        
        # Handle DICOM volume job IDs specially
        if case_name.startswith(('dicom_volume_', 'zip_volume_')):
            # Extract the username from the job ID format: {prefix}_{username}_something_timestamp
            parts = case_name.split('_')
            if len(parts) >= 4:  # At minimum: dicom/zip_volume_username_timestamp
                username_from_job = parts[2]  # The username should be the 3rd part
                logger.info(f"Extracted username '{username_from_job}' from job ID {case_name} for slice access")
                
                # Check permissions based on the extracted username
                if user_role != 'admin' and current_user != username_from_job:
                    logger.warning(f"Unauthorized access attempt to job slice {axis}/{index} for {case_name} by user {current_user}")
                    return jsonify({'error': 'Unauthorized access. You can only view your own results.'}), 403
        else:
            # Regular case name check
            if user_role != 'admin' and not case_name.startswith(current_user):
                logger.warning(f"Unauthorized access attempt to slice {axis}/{index} for {case_name} by user {current_user}")
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
            logger.info(f"Extracted base case name for slice: {base_case_name}")
        
        # Find the source file
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        
        # Try different possible source files with both name formats
        source_files = []
        
        # For DICOM volume jobs, try to find the source MHD file
        if is_dicom_volume:
            # Extract the prefix from the job ID (e.g., 'doctor_dicom_volume_20250517124451')
            # Assuming format is 'dicom_volume_username_dicom_volume_timestamp'
            parts = case_name.split('_')
            if len(parts) >= 5:  # At minimum: dicom_volume_username_dicom_volume_timestamp
                # Reconstruct MHD filename pattern
                mhd_prefix = '_'.join(parts[2:])  # e.g., 'username_dicom_volume_timestamp'
                
                # Look for MHD files in uploads
                mhd_patterns = [
                    f"{mhd_prefix}_volume.mhd",
                    f"{mhd_prefix}*.mhd",
                    f"*{parts[-1]}_volume.mhd"  # Match by timestamp
                ]
                
                for pattern in mhd_patterns:
                    matching_files = list(upload_dir.glob(pattern))
                    if matching_files:
                        source_files.extend(matching_files)
                        logger.info(f"Found DICOM volume source files for slice: {matching_files}")
                        break
        
        # If no source files found yet, try traditional patterns
        if not source_files:
            # Try with original case name
            for ext in ['.mhd', '.nii.gz', '.nii', '.dcm']:
                source_files.extend(list(upload_dir.glob(f"{original_case_name}*{ext}")))
            
            # If we have a base case name, also try with that
            if base_case_name and not source_files:
                for ext in ['.mhd', '.nii.gz', '.nii', '.dcm']:
                    source_files.extend(list(upload_dir.glob(f"{base_case_name}*{ext}")))
                    
                # If we found files with the base name, update case_name
                if source_files:
                    case_name = base_case_name
        
        if not source_files:
            # Source file not found, use placeholder instead
            logger.warning(f"Source file not found for case {case_name}, using placeholder image")
            
            # Import the placeholder generator function
            from utils import generate_placeholder_slice
            
            # Generate placeholder slice
            slice_data = generate_placeholder_slice()
            
            # Return placeholder with some reasonable max dimensions
            max_indices = {
                'axial': 100,
                'coronal': 100,
                'sagittal': 100
            }
            
            return jsonify({
                'slice_data': slice_data,
                'max_index': max_indices[axis],
                'is_placeholder': True
            })
        
        source_file = source_files[0]
        
        # Read the volume
        if source_file.suffix == '.mhd':
            # SimpleITK for MHD/RAW
            img = sitk.ReadImage(str(source_file))
            volume = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            spacing = (spacing[2], spacing[1], spacing[0])  # Reorder to match array order
        else:
            # NiBabel for NIFTI
            img = nib.load(str(source_file))
            volume = img.get_fdata().transpose(2, 1, 0)  # ZYX order for consistency
            spacing = img.header.get_zooms()[::-1]
            
        # Apply adaptive windowing and contrast enhancement
        def adaptive_window_level(slice_data, default_window=1500, default_level=-600):
            """Apply adaptive window/level based on histogram analysis"""
            # Get histogram to analyze the intensity distribution
            hist, bin_edges = np.histogram(slice_data, bins=100)
            
            # Find the peaks in the histogram (typically air, soft tissue, and bone)
            peak_indices = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
            peak_values = bin_edges[peak_indices]
            
            if len(peak_values) >= 2:
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
        
        def enhance_contrast(image, low_percentile=5, high_percentile=95):
            """Enhance contrast using percentile-based normalization"""
            # Get percentile values
            low = np.percentile(image, low_percentile)
            high = np.percentile(image, high_percentile)
            
            # Apply contrast stretching
            enhanced = np.clip(image, low, high)
            enhanced = ((enhanced - low) / (high - low) * 255).astype(np.uint8)
            
            return enhanced
            
        # Get the slice based on axis
        if axis == 'axial':
            if index >= volume.shape[0]:
                return jsonify({'error': 'Slice index out of range'}), 400
            slice_data = volume[index].copy()
            aspect_ratio = spacing[1] / spacing[2]  # Y/X aspect ratio
        elif axis == 'coronal':
            if index >= volume.shape[1]:
                return jsonify({'error': 'Slice index out of range'}), 400
            slice_data = volume[:, index, :].T
            # Rotate the coronal slice 90 degrees counterclockwise for better orientation
            slice_data = np.rot90(slice_data)
            aspect_ratio = spacing[0] / spacing[2]  # Z/X aspect ratio
        elif axis == 'sagittal':
            if index >= volume.shape[2]:
                return jsonify({'error': 'Slice index out of range'}), 400
            # Get the sagittal slice and rotate it for proper orientation
            slice_data = volume[:, :, index].T
            # Rotate the sagittal slice for better orientation
            slice_data = np.rot90(slice_data)
            aspect_ratio = spacing[0] / spacing[1]  # Z/Y aspect ratio
        else:
            return jsonify({'error': 'Invalid axis. Choose from: axial, coronal, sagittal'}), 400
        
        # Apply adaptive windowing
        window, level = adaptive_window_level(slice_data)
        window_min = level - (window / 2)
        window_max = level + (window / 2)
        
        # Apply window/level with better contrast
        normalized_slice = np.clip(slice_data, window_min, window_max)
        normalized_slice = ((normalized_slice - window_min) / (window_max - window_min) * 255).astype(np.uint8)
        
        # Apply contrast enhancement
        normalized_slice = enhance_contrast(normalized_slice)
        
        # Load nodule information to highlight on the slice
        try:
            results_folder = Path(app.config['OUTPUT_FOLDER'])
            nodule_file = results_folder / f"{case_name}_nodules.json"
            
            highlight_mask = np.zeros_like(normalized_slice, dtype=np.float32)
            
            if nodule_file.exists():
                with open(nodule_file, 'r') as f:
                    nodule_data = json.load(f)
                    
                # Find nodules near this slice
                if 'nodules' in nodule_data:
                    nodules = nodule_data['nodules']
                    
                    for nodule in nodules:
                        z, y, x = [int(round(coord)) for coord in nodule['coordinates']]
                        r = int(round(nodule['radius_mm'] * 1.5))  # Slightly larger highlight region
                        
                        # Determine if the nodule is visible in this slice
                        if axis == 'axial' and abs(z - index) < r + 3:
                            # Create circular highlight in the slice
                            y_indices, x_indices = np.ogrid[-r:r+1, -r:r+1]
                            mask = x_indices**2 + y_indices**2 <= r**2
                            
                            # Calculate z-distance factor (1.0 at center, decreasing toward edges)
                            z_factor = 1.0 - (abs(z - index) / (r + 3))
                            
                            # Get bounds for the highlight region
                            y_min, y_max = max(0, y-r), min(normalized_slice.shape[0], y+r+1)
                            x_min, x_max = max(0, x-r), min(normalized_slice.shape[1], x+r+1)
                            
                            # Calculate mask bounds
                            mask_y_min = max(0, r - y) if y < r else 0
                            mask_y_max = mask_y_min + (y_max - y_min)
                            mask_x_min = max(0, r - x) if x < r else 0
                            mask_x_max = mask_x_min + (x_max - x_min)
                            
                            # Apply the highlight with distance-based intensity
                            try:
                                valid_mask = mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
                                highlight_mask[y_min:y_max, x_min:x_max][valid_mask] = 0.3 * z_factor
                                
                                # Get malignancy information
                                malignancy = nodule.get('malignancy', 'Unknown')
                                malignancy_score = nodule.get('malignancy_score', 0.0)
                                nodule['label'] = f"{malignancy} ({malignancy_score:.2f})"
                                
                                # Set color based on malignancy
                                if malignancy == 'Malignant':
                                    nodule['color'] = 'red'
                                elif malignancy == 'Benign':
                                    nodule['color'] = 'green'
                                else:
                                    nodule['color'] = 'yellow'
                            except (IndexError, ValueError):
                                # Skip if indices are out of bounds
                                continue
                                
                        elif axis == 'coronal' and abs(y - index) < r + 3:
                            # For coronal view (rotated)
                            # Apply rotation to coordinates
                            rotated_x = z
                            rotated_y = volume.shape[2] - x
                            
                            # Calculate y-distance factor (1.0 at center, decreasing toward edges)
                            y_factor = 1.0 - (abs(y - index) / (r + 3))
                            
                            # Create circular highlight in the slice
                            y_indices, x_indices = np.ogrid[-r:r+1, -r:r+1]
                            mask = x_indices**2 + y_indices**2 <= r**2
                            
                            # Get bounds for the highlight region
                            y_min, y_max = max(0, rotated_y-r), min(normalized_slice.shape[0], rotated_y+r+1)
                            x_min, x_max = max(0, rotated_x-r), min(normalized_slice.shape[1], rotated_x+r+1)
                            
                            # Calculate mask bounds
                            mask_y_min = max(0, r - rotated_y) if rotated_y < r else 0
                            mask_y_max = mask_y_min + (y_max - y_min)
                            mask_x_min = max(0, r - rotated_x) if rotated_x < r else 0
                            mask_x_max = mask_x_min + (x_max - x_min)
                            
                            # Apply the highlight with distance-based intensity
                            try:
                                valid_mask = mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
                                highlight_mask[y_min:y_max, x_min:x_max][valid_mask] = 0.3 * y_factor
                            except (IndexError, ValueError):
                                # Skip if indices are out of bounds
                                continue
                                
                        elif axis == 'sagittal' and abs(x - index) < r + 3:
                            # For sagittal view (rotated)
                            # Apply rotation to coordinates for consistency with the rotated view
                            rotated_x = z
                            rotated_y = y
                            
                            # Calculate x-distance factor
                            x_factor = 1.0 - (abs(x - index) / (r + 3))
                            
                            # Same logic for masking as above
                            y_indices, x_indices = np.ogrid[-r:r+1, -r:r+1]
                            mask = x_indices**2 + y_indices**2 <= r**2
                            
                            y_min, y_max = max(0, rotated_y-r), min(normalized_slice.shape[0], rotated_y+r+1)
                            x_min, x_max = max(0, rotated_x-r), min(normalized_slice.shape[1], rotated_x+r+1)
                            
                            mask_y_min = max(0, r - rotated_y) if rotated_y < r else 0
                            mask_y_max = mask_y_min + (y_max - y_min)
                            mask_x_min = max(0, r - rotated_x) if rotated_x < r else 0
                            mask_x_max = mask_x_min + (x_max - x_min)
                            
                            try:
                                valid_mask = mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
                                highlight_mask[y_min:y_max, x_min:x_max][valid_mask] = 0.3 * x_factor
                            except (IndexError, ValueError):
                                # Skip if indices are out of bounds
                                continue
        except Exception as e:
            # If there's an error loading nodules, just continue without highlights
            logger.error(f"Error loading nodules for highlighting: {e}")
            pass
        
        # Calculate figure size with proper aspect ratio
        # Always use a square figure
        fig_size = 8  # 8x8 inches
        plt.figure(figsize=(fig_size, fig_size), dpi=100)
        
        # Display the image with the correct aspect ratio inside the square figure
        plt.imshow(normalized_slice, cmap='gray', aspect=aspect_ratio)
        
        # Overlay highlights for nodules if any exist
        if np.any(highlight_mask > 0):
            highlight_overlay = np.ma.masked_where(highlight_mask == 0, highlight_mask)
            plt.imshow(highlight_overlay, cmap='Reds', alpha=0.3)
            
        plt.axis('off')
        # Remove padding/margins to maximize image size in the square
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save the figure to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buffer.seek(0)
        
        slice_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Return slice and volume info
        max_indices = {
            'axial': volume.shape[0] - 1,
            'coronal': volume.shape[1] - 1,
            'sagittal': volume.shape[2] - 1
        }
        
        return jsonify({
            'slice_data': slice_data,
            'max_index': max_indices[axis],
            'is_placeholder': False
        })
        
    except Exception as e:
        logger.error(f"Error in get_slice: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Attempt to provide a placeholder image even in case of other errors
        try:
            from utils import generate_placeholder_slice
            slice_data = generate_placeholder_slice()
            
            return jsonify({
                'slice_data': slice_data,
                'max_index': 100,  # Arbitrary reasonable value
                'is_placeholder': True,
                'error_message': str(e)
            })
        except Exception as placeholder_error:
            logger.error(f"Failed to generate placeholder: {placeholder_error}")
            return jsonify({'error': 'An error occurred while retrieving the slice'}), 500

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

# Serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
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
    if user_role != 'admin':
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
            'created_at': data.get('created_at', '')
        }
        user_list.append(user_data)
    
    return jsonify(user_list), 200

@app.route('/api/users', methods=['POST'])
@jwt_required()
def create_user():
    """Create a new user (admin only)."""
    # Get current user identity and role from JWT
    current_user = get_jwt_identity()
    user_role = users_db.get(current_user, {}).get('role', '')
    
    # Check if user is admin
    if user_role != 'admin':
        return jsonify({'error': 'Unauthorized. Admin access required.'}), 403
    
    if not request.is_json:
        return jsonify({'error': 'Missing JSON in request'}), 400
        
    # Required fields
    username = request.json.get('username')
    password = request.json.get('password')
    
    # Optional fields with defaults
    role = request.json.get('role', 'doctor')
    first_name = request.json.get('first_name', '')
    last_name = request.json.get('last_name', '')
    email = request.json.get('email', '')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
        
    if username in users_db:
        return jsonify({'error': 'Username already exists'}), 400
        
    if role not in ['admin', 'doctor']:
        return jsonify({'error': 'Invalid role. Must be admin or doctor'}), 400
    
    # Validate email format if provided
    if email and '@' not in email:
        return jsonify({'error': 'Invalid email format'}), 400
    
    # Add user to database
    users_db[username] = {
        'password': generate_password_hash(password),
        'role': role,
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'created_at': datetime.datetime.now().isoformat()
    }
    
    # In a real application, you would persist this to a database
    
    return jsonify({
        'success': True,
        'message': f'User {username} created successfully',
        'user': {
            'username': username, 
            'role': role,
            'first_name': first_name,
            'last_name': last_name,
            'email': email
        }
    }), 201

@app.route('/api/users/<username>', methods=['DELETE'])
@jwt_required()
def delete_user(username):
    """Delete a user (admin only)."""
    # Get current user identity and role from JWT
    current_user = get_jwt_identity()
    user_role = users_db.get(current_user, {}).get('role', '')
    
    # Check if user is admin
    if user_role != 'admin':
        return jsonify({'error': 'Unauthorized. Admin access required.'}), 403
    
    # Check if user exists
    if username not in users_db:
        return jsonify({'error': 'User not found'}), 404
        
    # Prevent self-deletion
    if username == current_user:
        return jsonify({'error': 'Cannot delete your own account for security reasons'}), 400
    
    # Delete the user
    del users_db[username]
    
    # In a real application, you would persist this to a database
    
    return jsonify({
        'success': True,
        'message': f'User {username} deleted successfully'
    }), 200

@app.route('/api/users/<username>', methods=['PUT'])
@jwt_required()
def update_user(username):
    """Update a user (admin can update anyone, users can only update themselves)."""
    # Get current user identity and role from JWT
    current_user = get_jwt_identity()
    user_role = users_db.get(current_user, {}).get('role', '')
    
    # Check if the user is updating their own profile or is an admin
    if current_user != username and user_role != 'admin':
        return jsonify({'error': 'Unauthorized. You can only update your own profile unless you are an admin.'}), 403
    
    if not request.is_json:
        return jsonify({'error': 'Missing JSON in request'}), 400
        
    # Check if user exists
    if username not in users_db:
        return jsonify({'error': 'User not found'}), 404
    
    # Get update fields
    new_username = request.json.get('username')
    new_password = request.json.get('password')
    new_role = request.json.get('role')
    new_first_name = request.json.get('first_name')
    new_last_name = request.json.get('last_name')
    new_email = request.json.get('email')
    current_password = request.json.get('current_password')
    
    # Check if username is being changed
    username_changed = new_username and new_username != username
    
    # If changing username, make sure the new username doesn't already exist
    if username_changed and new_username in users_db:
        return jsonify({'error': f'Username {new_username} is already taken'}), 400
    
    # Only admins can change roles
    if new_role and user_role != 'admin':
        return jsonify({'error': 'Changing roles requires admin privileges'}), 403
    
    # Non-admins must provide current password to change their password
    if new_password and user_role != 'admin' and current_user == username:
        if not current_password:
            return jsonify({'error': 'Current password is required to set a new password'}), 400
        
        # Verify current password
        if not check_password_hash(users_db[username]['password'], current_password):
            return jsonify({'error': 'Current password is incorrect'}), 401
    
    if new_role and new_role not in ['admin', 'doctor']:
        return jsonify({'error': 'Invalid role. Must be admin or doctor'}), 400
    
    # Validate email format if provided
    if new_email and '@' not in new_email:
        return jsonify({'error': 'Invalid email format'}), 400
    
    # Prevent admins from downgrading their own role
    if username == current_user and new_role and new_role != 'admin' and user_role == 'admin':
        return jsonify({'error': 'You cannot downgrade your own admin role for security reasons'}), 400
    
    # Handle username change by creating a new user entry
    if username_changed:
        # Copy existing user data to new username
        users_db[new_username] = users_db[username].copy()
        
        # Keep track of previous usernames for continuity
        # Initialize previous_usernames list if it doesn't exist
        if 'previous_usernames' not in users_db[new_username]:
            users_db[new_username]['previous_usernames'] = []
        
        # Add the old username to the previous_usernames list
        users_db[new_username]['previous_usernames'].append(username)
        
        # Apply updates to the new user entry
        if new_password:
            users_db[new_username]['password'] = generate_password_hash(new_password)
        
        if new_role and user_role == 'admin':
            users_db[new_username]['role'] = new_role
        
        if new_first_name is not None:
            users_db[new_username]['first_name'] = new_first_name
        
        if new_last_name is not None:
            users_db[new_username]['last_name'] = new_last_name
        
        if new_email is not None:
            users_db[new_username]['email'] = new_email
        
        # Delete the old username entry if user is updating their own profile
        # (Admin might want to keep the old account when changing someone else's username)
        if current_user == username:
            del users_db[username]
        
        # Create a user object to return
        user_data = {
            'username': new_username,
            'role': users_db[new_username]['role'],
            'first_name': users_db[new_username].get('first_name', ''),
            'last_name': users_db[new_username].get('last_name', ''),
            'email': users_db[new_username].get('email', ''),
            'previous_usernames': users_db[new_username].get('previous_usernames', [])
        }
    else:
        # No username change - update the existing user
        if new_password:
            users_db[username]['password'] = generate_password_hash(new_password)
            
        if new_role and user_role == 'admin':
            users_db[username]['role'] = new_role
            
        if new_first_name is not None:
            users_db[username]['first_name'] = new_first_name
            
        if new_last_name is not None:
            users_db[username]['last_name'] = new_last_name
            
        if new_email is not None:
            users_db[username]['email'] = new_email
        
        # Create a user object to return
        user_data = {
            'username': username,
            'role': users_db[username]['role'],
            'first_name': users_db[username].get('first_name', ''),
            'last_name': users_db[username].get('last_name', ''),
            'email': users_db[username].get('email', ''),
            'previous_usernames': users_db[username].get('previous_usernames', [])
        }
    
    # In a real application, you would persist this to a database
    
    return jsonify({
        'success': True,
        'message': f'User profile updated successfully',
        'user': user_data
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
                    # Continue even if patient info fails, as the main results are already saved
            
            # Just remove the temporary flag, results are already saved to the output folder
            # Remove the temporary result from the session
            del app.config[session_key]
            
            # Update the result_path based on the job type
            result_path = f"/api/results/{job_id}" if is_dicom_volume else f"/api/results/{case_name}"
            
            return jsonify({
                'success': True, 
                'message': 'Results saved successfully',
                'case_name': case_name,
                'job_id': job_id,
                'result_path': result_path
            }), 200
    
    except Exception as e:
        logger.error(f"Error processing results action: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing results action: {str(e)}'}), 500


# Add a new route to get preview results for a specific job ID
@app.route('/api/results/preview/<job_id>', methods=['GET'])
@jwt_required()
def get_preview_result(job_id):
    """Get preview results for a specific job ID."""
    try:
        # Get current user identity and role for permission check
        current_user = get_jwt_identity()
        
        # Check if the job_id exists in the temporary storage
        session_key = f"temp_result_{job_id}"
        if session_key not in app.config:
            logger.error(f"No temporary results found for job ID: {job_id}")
            return jsonify({'error': 'No temporary results found for this job ID'}), 404
        
        # Get the temporary result data
        temp_result = app.config[session_key]
        case_name = temp_result['case_name']
        
        # Get result details from the job-specific subfolder
        job_result_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        
        # Look for result files in the job subfolder
        image_files = list(job_result_dir.glob("*_results.png"))
        if not image_files:
            logger.error(f"Preview results image not found for job ID: {job_id} in {job_result_dir}")
            return jsonify({'error': 'Preview results not found'}), 404
        
        # Use the first image file found
        image_file = image_files[0]
        logger.info(f"Found preview image: {image_file}")
        
        # Get the result details text file
        details_files = list(job_result_dir.glob("*_results.txt"))
        details = ""
        if details_files:
            details_file = details_files[0]
            try:
                with open(details_file, 'r') as f:
                    details = f.read()
                logger.info(f"Loaded details from {details_file}")
            except Exception as e:
                logger.error(f"Error reading details file: {e}")
        
        # Read the processing_info.json file for status
        status = 'completed'
        try:
            with open(job_result_dir / 'processing_info.json', 'r') as f:
                processing_info = json.load(f)
                status = processing_info.get('status', 'completed')
        except Exception as e:
            logger.error(f"Error reading processing info: {e}")
        
        # Create a result object with status and data
        result = {
            'case_name': case_name,
            'details': details,
            'job_id': job_id,
            'image_url': f"/api/results/{job_id}/image",  # Updated path to access image
            'timestamp': temp_result['timestamp'],
            'is_preview': True,
            'status': status,
            'nodule_count': temp_result.get('nodule_count', 0)
        }
        
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
                try:
                    logger.info(f"Starting pipeline processing for job {job_id}")
                    
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
                    
                    logger.info(f"Processing completed for job {job_id}")
                    
                    # Update processing info
                    processing_info['status'] = 'completed'
                    processing_info['completion_time'] = datetime.datetime.now().isoformat()
                    
                    # Extract nodule count
                    nodule_count = len(results.get('nodules', []))
                    processing_info['nodule_count'] = nodule_count
                    
                    # Create a summary
                    if nodule_count > 0:
                        processing_info['details'] = f"Detected {nodule_count} nodules."
                    else:
                        processing_info['details'] = "No nodules detected."
                    
                    # Save updated processing info
                    with open(os.path.join(output_dir, 'processing_info.json'), 'w') as f:
                        json.dump(processing_info, f, indent=2)
                    
                    # Store the result in app.config for the preview endpoint to access
                    case_name = f"{job_id}"
                    temp_result_key = f"temp_result_{job_id}"
                    temp_result = {
                        'case_name': case_name,
                        'job_id': job_id,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'status': 'completed',
                        'nodule_count': nodule_count
                    }
                    app.config[temp_result_key] = temp_result
                    logger.info(f"Stored temporary result in app.config with key: {temp_result_key}")
                        
                    # Clear GPU memory after processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared GPU memory after processing scan")
                    
                except Exception as e:
                    logger.error(f"Error processing scan for job {job_id}: {e}")
                    logger.error(traceback.format_exc())
                    
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
                    
                    logger.info(f"Processing completed for job {job_id}")
                    
                    # Update processing info
                    processing_info['status'] = 'completed'
                    processing_info['completion_time'] = datetime.datetime.now().isoformat()
                    
                    # Extract nodule count
                    nodule_count = len(results.get('nodules', []))
                    processing_info['nodule_count'] = nodule_count
                    
                    # Create a summary
                    if nodule_count > 0:
                        processing_info['details'] = f"Detected {nodule_count} nodules."
                    else:
                        processing_info['details'] = "No nodules detected."
                    
                    # Save updated processing info
                    with open(os.path.join(output_dir, 'processing_info.json'), 'w') as f:
                        json.dump(processing_info, f, indent=2)
                    
                    # Store the result in app.config for the preview endpoint to access
                    case_name = f"{job_id}"
                    temp_result_key = f"temp_result_{job_id}"
                    temp_result = {
                        'case_name': case_name,
                        'job_id': job_id,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'status': 'completed',
                        'nodule_count': nodule_count
                    }
                    app.config[temp_result_key] = temp_result
                    logger.info(f"Stored temporary result in app.config with key: {temp_result_key}")
                        
                    # Clear GPU memory after processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared GPU memory after processing scan")
                    
                except Exception as e:
                    logger.error(f"Error processing scan for job {job_id}: {e}")
                    logger.error(traceback.format_exc())
                    
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
        
    app.run(debug=True, host='0.0.0.0', port=5000) 