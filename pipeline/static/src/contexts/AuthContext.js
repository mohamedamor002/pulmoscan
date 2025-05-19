import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

// Create auth context
const AuthContext = createContext();

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Role hierarchy definition
export const ROLES = {
  SUPERADMIN: 'superadmin',
  ADMIN: 'admin',
  DOCTOR: 'doctor',
  RADIOLOGIST: 'radiologist'  // Adding radiologist role
};

// Role permissions check helpers
export const canManageAdmins = (role) => role === ROLES.SUPERADMIN;
export const canManageDoctors = (role) => [ROLES.SUPERADMIN, ROLES.ADMIN].includes(role);
export const canAccessAdminArea = (role) => [ROLES.SUPERADMIN, ROLES.ADMIN].includes(role);
export const canManageRadiologists = (role) => [ROLES.SUPERADMIN, ROLES.ADMIN].includes(role);

// Role hierarchy check helpers
export const canManageRole = (currentRole, targetRole) => {
  switch (currentRole) {
    case ROLES.SUPERADMIN:
      return true; // Superadmin can manage all roles
    case ROLES.ADMIN:
      return [ROLES.DOCTOR, ROLES.RADIOLOGIST].includes(targetRole); // Admin can only manage doctors and radiologists
    default:
      return false; // Other roles can't manage any roles
  }
};

// Auth provider component
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize auth state from local storage
  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (token && userData) {
      setUser(JSON.parse(userData));
      setIsAuthenticated(true);
      
      // Set the authorization header for all requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }
    
    setIsLoading(false);
  }, []);

  // Login function
  const login = async (username, password) => {
    try {
      const response = await axios.post('/api/auth/login', { username, password });
      const { access_token, username: user, role } = response.data;
      
      // Build user object with all available data
      const userObject = {
        username,
        role,
        // Include any other fields from the response
        firstName: response.data.first_name || '',
        lastName: response.data.last_name || '',
        email: response.data.email || ''
      };
      
      // Store token and user data
      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(userObject));
      
      // Set the authorization header for all requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      // Update state
      setUser(userObject);
      setIsAuthenticated(true);
      
      return { success: true };
    } catch (error) {
      return {
        success: false,
        message: error.response?.data?.msg || 'Login failed. Please check your credentials.'
      };
    }
  };

  // Function to update the user context after profile changes
  const updateUserContext = (updatedFields) => {
    if (!user) return false;
    
    try {
      // Get current user data
      const userData = JSON.parse(localStorage.getItem('user') || '{}');
      
      // Update with new fields
      const updatedUser = {
        ...userData,
        ...updatedFields
      };
      
      // Save to localStorage
      localStorage.setItem('user', JSON.stringify(updatedUser));
      
      // Update state
      setUser(updatedUser);
      
      return true;
    } catch (error) {
      console.error("Error updating user context:", error);
      return false;
    }
  };

  // Logout function
  const logout = () => {
    // Remove token and user data
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    
    // Remove the authorization header
    delete axios.defaults.headers.common['Authorization'];
    
    // Update state
    setUser(null);
    setIsAuthenticated(false);
  };

  // Check if user has specific role
  const hasRole = (requiredRole) => {
    if (!user) return false;
    
    // If required role is an array, check if user has any of the roles
    if (Array.isArray(requiredRole)) {
      return requiredRole.includes(user.role);
    }
    
    // Otherwise check for specific role
    return user.role === requiredRole;
  };

  // Value to be provided to consumers
  const value = {
    user,
    isAuthenticated,
    isLoading,
    login,
    logout,
    updateUserContext,
    hasRole,
    canManageAdmins: user ? canManageAdmins(user.role) : false,
    canManageDoctors: user ? canManageDoctors(user.role) : false,
    canAccessAdminArea: user ? canAccessAdminArea(user.role) : false,
    canManageRadiologists: user ? canManageRadiologists(user.role) : false
  };

  return (
    <AuthContext.Provider value={value}>
      {!isLoading && children}
    </AuthContext.Provider>
  );
}; 