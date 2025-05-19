// Simple toast utility to replace react-hot-toast
const toast = {
  success: (message) => {
    console.log('Success:', message);
    // In a real implementation, this would show a toast notification
    // For now, we'll just use browser alerts
    alert('Success: ' + message);
  },
  error: (message) => {
    console.error('Error:', message);
    // In a real implementation, this would show a toast notification
    alert('Error: ' + message);
  },
  loading: (message) => {
    console.log('Loading:', message);
    return Math.random().toString(); // Return a fake toast ID
  },
  dismiss: (id) => {
    console.log('Dismissed toast:', id);
  }
};

export { toast }; 