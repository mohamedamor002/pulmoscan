from flask import Flask, render_template, request, jsonify
import SimpleITK as sitk
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
app.config.update({
    'UPLOAD_FOLDER': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads'),
    'ALLOWED_EXTENSIONS': {'dcm'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB max
})

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload_page():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/static/js/<path:filename>')
def serve_static_js(filename):
    return send_from_directory('static/js', filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'dicomFile' not in request.files:
        return jsonify({'error': True, 'message': "No file uploaded"}), 400
    
    file = request.files['dicomFile']
    if file.filename == '':
        return jsonify({'error': True, 'message': "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': True, 'message': "Invalid file type (only .dcm allowed)"}), 400
    
    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Read DICOM file
        reader = sitk.ImageFileReader()
        reader.SetFileName(temp_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # Get image data
        image = reader.Execute()
        
        # Basic analysis (replace with your actual analysis)
        array = sitk.GetArrayFromImage(image)
        
        # Generate report
        analysis_report = {
            "findings": "No significant abnormalities detected",
            "confidence": "92%",
            "recommendation": "Routine follow-up recommended",
            "image_quality": "Excellent",
            "dimensions": array.shape,
            "spacing": reader.GetSpacing(),
            "origin": reader.GetOrigin()
        }
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify({
            'error': False,
            'message': "Analysis complete",
            'report': analysis_report
        })
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f"Error processing DICOM: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)