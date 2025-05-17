# Lung Nodule Detection Web Application

A responsive web application for detecting and segmenting lung nodules in CT scans.

## Features

- **User Authentication**: Secure login system with role-based access control
- **Dashboard**: Overview of processed scans and statistics
- **CT Scan Upload**: Upload and process CT scans with customizable parameters
- **Results Visualization**: View detailed results with visualization of detected nodules
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Tech Stack

### Backend
- **Flask**: Web framework for the API
- **JWT**: Token-based authentication
- **SimpleITK**: Medical image processing
- **PyTorch**: Deep learning framework for the neural networks

### Frontend
- **React**: UI library
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js**: Data visualization
- **Axios**: HTTP client for API requests
- **React Router**: Client-side routing

## Installation

### Prerequisites

- Python 3.8+
- Node.js 14+ and npm
- GPU with CUDA support (recommended for faster processing)

### Backend Setup

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Start the Flask development server:
   ```
   python app.py
   ```

### Frontend Setup

1. Navigate to the static directory:
   ```
   cd static
   ```

2. Install the required npm packages:
   ```
   npm install
   ```

3. Build the frontend:
   ```
   npm run build
   ```

## Usage

Once the application is running, access it through your browser at http://localhost:5000

### Default Users

- Admin:
  - Username: `admin`
  - Password: `admin123`

- Doctor:
  - Username: `doctor`
  - Password: `doctor123`

## Development

For development, you can run the frontend development server separately:

```
cd static
npm start
```

The React development server will run on http://localhost:3000 and proxy API requests to the Flask server at http://localhost:5000.

## Deployment

For production deployment, it's recommended to:

1. Use a production WSGI server like Gunicorn
2. Set up a reverse proxy with Nginx
3. Use a proper database instead of the file-based storage
4. Configure HTTPS
5. Use environment variables for sensitive information

Example deployment command:

```
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 