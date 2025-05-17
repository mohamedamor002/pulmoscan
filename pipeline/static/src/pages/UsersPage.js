import React, { useState, useEffect } from 'react';
import { useAuth, ROLES, canManageAdmins, canManageDoctors } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import axios from 'axios';
import { PlusIcon, PencilIcon, TrashIcon, ExclamationCircleIcon, ShieldCheckIcon, UserIcon } from '@heroicons/react/24/outline';

const UsersPage = () => {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateUserModal, setShowCreateUserModal] = useState(false);
  const [showEditUserModal, setShowEditUserModal] = useState(false);
  const [showDeleteUserModal, setShowDeleteUserModal] = useState(false);
  const [userToEdit, setUserToEdit] = useState(null);
  const [userToDelete, setUserToDelete] = useState(null);
  
  const { user: currentUser, canManageAdmins: userCanManageAdmins, canManageDoctors: userCanManageDoctors } = useAuth();
  const { darkMode } = useTheme();
  
  // Form states for creating and editing users
  const [formUsername, setFormUsername] = useState('');
  const [formPassword, setFormPassword] = useState('');
  const [formRole, setFormRole] = useState(ROLES.DOCTOR);
  const [formFirstName, setFormFirstName] = useState('');
  const [formLastName, setFormLastName] = useState('');
  const [formEmail, setFormEmail] = useState('');
  const [formError, setFormError] = useState('');
  
  // Fetch users from the API
  const fetchUsers = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get('/api/users');
      
      // Filter users based on role permissions
      let filteredUsers = response.data;
      
      // If not a superadmin, filter out superadmins
      if (currentUser.role !== ROLES.SUPERADMIN) {
        filteredUsers = filteredUsers.filter(user => user.role !== ROLES.SUPERADMIN);
      }
      
      // If not an admin or superadmin, filter out admins too
      if (!userCanManageDoctors) {
        filteredUsers = filteredUsers.filter(user => user.role === ROLES.DOCTOR);
      }
      
      setUsers(filteredUsers);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching users:', error);
      setError('Failed to load users. Please try again later.');
      setIsLoading(false);
    }
  };
  
  useEffect(() => {
    fetchUsers();
  }, [currentUser.role]);
  
  // Check if current user can access the users page
  if (!userCanManageDoctors) {
    return (
      <div className={`text-center py-12 ${darkMode ? 'text-white' : ''}`}>
        <div className={`${darkMode ? 'bg-red-900' : 'bg-red-50'} p-4 rounded-md inline-block`}>
          <div className="flex">
            <ExclamationCircleIcon className={`h-5 w-5 ${darkMode ? 'text-red-300' : 'text-red-400'}`} />
            <div className="ml-3">
              <h3 className={`text-sm font-medium ${darkMode ? 'text-red-300' : 'text-red-800'}`}>Unauthorized Access</h3>
              <div className={`mt-2 text-sm ${darkMode ? 'text-red-300' : 'text-red-700'}`}>
                <p>You do not have permission to access this page. This page is only available to administrators.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  // Create a new user
  const handleCreateUser = async () => {
    try {
      setFormError('');
      if (!formUsername || !formPassword) {
        setFormError('Username and password are required.');
        return;
      }
      
      // Validate email format if provided
      if (formEmail && !formEmail.includes('@')) {
        setFormError('Please enter a valid email address.');
        return;
      }

      // Check permission for role creation
      if (formRole === ROLES.ADMIN && !userCanManageAdmins) {
        setFormError('You do not have permission to create admin users.');
        return;
      }

      if (formRole === ROLES.SUPERADMIN && currentUser.role !== ROLES.SUPERADMIN) {
        setFormError('Only super admins can create other super admins.');
        return;
      }
      
      const userData = {
        username: formUsername,
        password: formPassword,
        role: formRole,
        first_name: formFirstName,
        last_name: formLastName,
        email: formEmail
      };
      
      const response = await axios.post('/api/users', userData);
      
      // Reset form and close modal
      setFormUsername('');
      setFormPassword('');
      setFormRole(ROLES.DOCTOR);
      setFormFirstName('');
      setFormLastName('');
      setFormEmail('');
      setFormError('');
      setShowCreateUserModal(false);
      
      // Refresh users list
      fetchUsers();
    } catch (error) {
      console.error('Error creating user:', error);
      setFormError(error.response?.data?.error || 'Failed to create user. Please try again.');
    }
  };
  
  // Update an existing user
  const handleUpdateUser = async () => {
    try {
      setFormError('');
      
      const dataToUpdate = {};
      if (formPassword) {
        dataToUpdate.password = formPassword;
      }
      
      // Check if role is being changed
      if (formRole && formRole !== userToEdit.role) {
        // Verify permission for role change
        if (formRole === ROLES.ADMIN && !userCanManageAdmins) {
          setFormError('You do not have permission to promote users to admin.');
          return;
        }
        
        if (formRole === ROLES.SUPERADMIN && currentUser.role !== ROLES.SUPERADMIN) {
          setFormError('Only super admins can promote users to super admin.');
          return;
        }
        
        // Prevent downgrading superadmins if not a superadmin
        if (userToEdit.role === ROLES.SUPERADMIN && currentUser.role !== ROLES.SUPERADMIN) {
          setFormError('You cannot modify a super admin account.');
          return;
        }
        
        dataToUpdate.role = formRole;
      }
      
      if (formFirstName !== undefined) {
        dataToUpdate.first_name = formFirstName;
      }
      if (formLastName !== undefined) {
        dataToUpdate.last_name = formLastName;
      }
      if (formEmail !== undefined) {
        // Validate email format if provided
        if (formEmail && !formEmail.includes('@')) {
          setFormError('Please enter a valid email address.');
          return;
        }
        dataToUpdate.email = formEmail;
      }
      
      if (Object.keys(dataToUpdate).length === 0) {
        setFormError('Please make at least one change.');
        return;
      }
      
      // Prevent modifying superadmin users if not a superadmin
      if (userToEdit.role === ROLES.SUPERADMIN && currentUser.role !== ROLES.SUPERADMIN) {
        setFormError('You do not have permission to modify super admin accounts.');
        return;
      }
      
      // Prevent admins from modifying other admins (only superadmins can)
      if (userToEdit.role === ROLES.ADMIN && currentUser.role === ROLES.ADMIN) {
        setFormError('Admins cannot modify other admin accounts.');
        return;
      }
      
      const response = await axios.put(`/api/users/${userToEdit.username}`, dataToUpdate);
      
      // Reset form and close modal
      setFormPassword('');
      setFormRole(userToEdit.role);
      setFormFirstName('');
      setFormLastName('');
      setFormEmail('');
      setUserToEdit(null);
      setShowEditUserModal(false);
      
      // Refresh users list
      fetchUsers();
    } catch (error) {
      console.error('Error updating user:', error);
      setFormError(error.response?.data?.error || 'Failed to update user. Please try again.');
    }
  };
  
  // Delete a user
  const handleDeleteUser = async () => {
    try {
      // Check permissions for role deletion
      if (userToDelete.role === ROLES.SUPERADMIN && currentUser.role !== ROLES.SUPERADMIN) {
        setError('You do not have permission to delete super admin accounts.');
        setShowDeleteUserModal(false);
        return;
      }
      
      // Prevent admins from deleting other admins (only superadmins can)
      if (userToDelete.role === ROLES.ADMIN && currentUser.role === ROLES.ADMIN) {
        setError('Admins cannot delete other admin accounts.');
        setShowDeleteUserModal(false);
        return;
      }
      
      // Prevent deleting your own account
      if (userToDelete.username === currentUser.username) {
        setError('You cannot delete your own account.');
        setShowDeleteUserModal(false);
        return;
      }
      
      const response = await axios.delete(`/api/users/${userToDelete.username}`);
      
      // Reset and close modal
      setUserToDelete(null);
      setShowDeleteUserModal(false);
      
      // Refresh users list
      fetchUsers();
    } catch (error) {
      console.error('Error deleting user:', error);
      setError(error.response?.data?.error || 'Failed to delete user. Please try again.');
    }
  };
  
  // Open the edit user modal
  const openEditUserModal = (user) => {
    setUserToEdit(user);
    setFormUsername(user.username);
    setFormPassword(''); // Don't prefill password for security
    setFormRole(user.role);
    setFormFirstName(user.first_name || '');
    setFormLastName(user.last_name || '');
    setFormEmail(user.email || '');
    setFormError('');
    setShowEditUserModal(true);
  };
  
  // Open the delete user modal
  const openDeleteUserModal = (user) => {
    setUserToDelete(user);
    setShowDeleteUserModal(true);
  };

  return (
    <div className={`space-y-6 ${darkMode ? 'text-white' : ''}`}>
      <div className="flex justify-between items-center">
        <h1 className={`text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>User Management</h1>
        <button 
          onClick={() => {
            setFormUsername('');
            setFormPassword('');
            // Set default role based on user's permissions
            setFormRole(userCanManageAdmins ? ROLES.ADMIN : ROLES.DOCTOR);
            setFormFirstName('');
            setFormLastName('');
            setFormEmail('');
            setFormError('');
            setShowCreateUserModal(true);
          }}
          className={`flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
        >
          <PlusIcon className="h-5 w-5 mr-1" />
          Add User
        </button>
      </div>

      {error && (
        <div className={`${darkMode ? 'bg-red-900' : 'bg-red-50'} p-4 rounded-md`}>
          <div className="flex">
            <ExclamationCircleIcon className={`h-5 w-5 ${darkMode ? 'text-red-300' : 'text-red-400'}`} />
            <div className="ml-3">
              <h3 className={`text-sm font-medium ${darkMode ? 'text-red-300' : 'text-red-800'}`}>{error}</h3>
            </div>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="text-center py-12">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-indigo-600 border-r-transparent"></div>
          <p className={`mt-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading users...</p>
        </div>
      ) : (
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow overflow-hidden sm:rounded-md`}>
          <ul className={`divide-y ${darkMode ? 'divide-gray-700' : 'divide-gray-200'}`}>
            {users.length === 0 ? (
              <li className={`px-6 py-4 text-center ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                No users found.
              </li>
            ) : (
              users.map((user) => (
                <li key={user.username}>
                  <div className="px-4 py-4 flex items-center sm:px-6">
                    <div className="min-w-0 flex-1 sm:flex sm:items-center sm:justify-between">
                      <div>
                        <div className="flex text-sm">
                          <p className={`font-medium ${darkMode ? 'text-indigo-300' : 'text-indigo-600'} truncate`}>
                            {user.username}
                          </p>
                          <div className="ml-2 flex items-center">
                            {user.role === ROLES.SUPERADMIN && (
                              <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${darkMode ? 'bg-purple-900 text-purple-100' : 'bg-purple-100 text-purple-800'}`}>
                                <ShieldCheckIcon className="mr-1 h-3 w-3" />
                                Super Admin
                              </span>
                            )}
                            {user.role === ROLES.ADMIN && (
                              <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${darkMode ? 'bg-blue-900 text-blue-100' : 'bg-blue-100 text-blue-800'}`}>
                                <ShieldCheckIcon className="mr-1 h-3 w-3" />
                                Admin
                              </span>
                            )}
                            {user.role === ROLES.DOCTOR && (
                              <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${darkMode ? 'bg-green-900 text-green-100' : 'bg-green-100 text-green-800'}`}>
                                <UserIcon className="mr-1 h-3 w-3" />
                                Doctor
                              </span>
                            )}
                          </div>
                          {user.username === currentUser.username && (
                            <span className={`ml-2 px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${darkMode ? 'bg-yellow-900 text-yellow-100' : 'bg-yellow-100 text-yellow-800'}`}>
                              Current User
                            </span>
                          )}
                        </div>
                        <div className={`mt-1 ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                          {user.first_name && user.last_name 
                            ? `${user.first_name} ${user.last_name}`
                            : ''}
                          {user.email && (
                            <span className="ml-2">
                              {user.email}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="mt-4 flex-shrink-0 sm:mt-0 sm:ml-5">
                        <div className="flex -space-x-1 overflow-hidden">
                          {/* Only show edit button if user has permission */}
                          {((user.role === ROLES.ADMIN && userCanManageAdmins) || 
                             (user.role === ROLES.DOCTOR && userCanManageDoctors) ||
                             (user.role === ROLES.SUPERADMIN && currentUser.role === ROLES.SUPERADMIN)) && (
                            <button
                              type="button"
                              onClick={() => openEditUserModal(user)}
                              className={`mr-2 inline-flex items-center px-2.5 py-1.5 border border-gray-300 shadow-sm text-xs font-medium rounded ${
                                darkMode 
                                  ? 'text-gray-300 bg-gray-700 hover:bg-gray-600 border-gray-600' 
                                  : 'text-gray-700 bg-white hover:bg-gray-50'
                              }`}
                            >
                              <PencilIcon className="-ml-0.5 mr-1 h-4 w-4" />
                              Edit
                            </button>
                          )}
                          
                          {/* Only show delete button if user has permission */}
                          {((user.role === ROLES.ADMIN && userCanManageAdmins) || 
                             (user.role === ROLES.DOCTOR && userCanManageDoctors) ||
                             (user.role === ROLES.SUPERADMIN && currentUser.role === ROLES.SUPERADMIN)) && 
                            user.username !== currentUser.username && (
                            <button
                              type="button"
                              onClick={() => openDeleteUserModal(user)}
                              className={`inline-flex items-center px-2.5 py-1.5 border border-transparent text-xs font-medium rounded ${
                                darkMode 
                                  ? 'text-red-300 bg-red-900 hover:bg-red-800' 
                                  : 'text-white bg-red-600 hover:bg-red-700'
                              }`}
                            >
                              <TrashIcon className="-ml-0.5 mr-1 h-4 w-4" />
                              Delete
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
              ))
            )}
          </ul>
        </div>
      )}

      {/* Create User Modal - Update this to allow proper role selection based on permissions */}
      {showCreateUserModal && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className={`absolute inset-0 ${darkMode ? 'bg-gray-900' : 'bg-gray-500'} opacity-75`}></div>
            </div>
            
            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
            
            <div className={`inline-block align-bottom ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full`}>
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} px-4 pt-5 pb-4 sm:p-6 sm:pb-4`}>
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                    <h3 className={`text-lg leading-6 font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      Create New User
                    </h3>
                    
                    {formError && (
                      <div className="mt-2">
                        <p className="text-sm text-red-500">{formError}</p>
                      </div>
                    )}
                    
                    <div className="mt-4 space-y-4">
                      <div>
                        <label htmlFor="username" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                          Username *
                        </label>
                        <input
                          type="text"
                          name="username"
                          id="username"
                          value={formUsername}
                          onChange={(e) => setFormUsername(e.target.value)}
                          required
                          className={`mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md ${
                            darkMode ? 'bg-gray-700 text-white border-gray-600' : ''
                          }`}
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="password" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                          Password *
                        </label>
                        <input
                          type="password"
                          name="password"
                          id="password"
                          value={formPassword}
                          onChange={(e) => setFormPassword(e.target.value)}
                          required
                          className={`mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md ${
                            darkMode ? 'bg-gray-700 text-white border-gray-600' : ''
                          }`}
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="role" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                          Role
                        </label>
                        <select
                          id="role"
                          name="role"
                          value={formRole}
                          onChange={(e) => setFormRole(e.target.value)}
                          className={`mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm ${
                            darkMode ? 'bg-gray-700 text-white border-gray-600' : ''
                          }`}
                        >
                          <option value={ROLES.DOCTOR}>Doctor</option>
                          {userCanManageAdmins && <option value={ROLES.ADMIN}>Admin</option>}
                          {currentUser.role === ROLES.SUPERADMIN && <option value={ROLES.SUPERADMIN}>Super Admin</option>}
                        </select>
                      </div>
                      
                      {/* Add the rest of the form fields */}
                      
                    </div>
                  </div>
                </div>
              </div>
              
              <div className={`${darkMode ? 'bg-gray-800 border-t border-gray-700' : 'bg-gray-50 border-t border-gray-200'} px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse`}>
                <button
                  type="button"
                  onClick={handleCreateUser}
                  className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-indigo-600 text-base font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:ml-3 sm:w-auto sm:text-sm"
                >
                  Create
                </button>
                <button
                  type="button"
                  onClick={() => setShowCreateUserModal(false)}
                  className={`mt-3 w-full inline-flex justify-center rounded-md border shadow-sm px-4 py-2 text-base font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm ${
                    darkMode 
                      ? 'border-gray-600 bg-gray-700 text-gray-300 hover:bg-gray-600' 
                      : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Edit User Modal - The modal should support dark mode and respect role permissions */}
      {showEditUserModal && userToEdit && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h2 className="text-xl font-semibold mb-4">Edit User: {userToEdit.username}</h2>
            
            {formError && (
              <div className="mb-4 bg-red-50 p-3 rounded-md">
                <p className="text-sm text-red-700">{formError}</p>
              </div>
            )}
            
            <div className="space-y-4">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label htmlFor="edit-first-name" className="block text-sm font-medium text-gray-700">
                    First Name
                  </label>
                  <input
                    type="text"
                    id="edit-first-name"
                    name="first_name"
                    value={formFirstName}
                    onChange={(e) => setFormFirstName(e.target.value)}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                  />
                </div>
                
                <div>
                  <label htmlFor="edit-last-name" className="block text-sm font-medium text-gray-700">
                    Last Name
                  </label>
                  <input
                    type="text"
                    id="edit-last-name"
                    name="last_name"
                    value={formLastName}
                    onChange={(e) => setFormLastName(e.target.value)}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                  />
                </div>
              </div>
              
              <div>
                <label htmlFor="edit-email" className="block text-sm font-medium text-gray-700">
                  Email
                </label>
                <input
                  type="email"
                  id="edit-email"
                  name="email"
                  value={formEmail}
                  onChange={(e) => setFormEmail(e.target.value)}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                />
              </div>
              
              <div>
                <label htmlFor="edit-password" className="block text-sm font-medium text-gray-700">
                  New Password (leave blank to keep current)
                </label>
                <input
                  type="password"
                  id="edit-password"
                  name="password"
                  value={formPassword}
                  onChange={(e) => setFormPassword(e.target.value)}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                />
              </div>
              
              <div>
                <label htmlFor="edit-role" className="block text-sm font-medium text-gray-700">
                  Role {userToEdit.username === currentUser.username && 
                    <span className="text-xs text-gray-500">(cannot change your own role)</span>}
                </label>
                <select
                  id="edit-role"
                  name="role"
                  value={formRole}
                  onChange={(e) => setFormRole(e.target.value)}
                  disabled={userToEdit.username === currentUser.username}
                  className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm ${
                    userToEdit.username === currentUser.username
                      ? 'bg-gray-100 text-gray-500 border-gray-300 cursor-not-allowed'
                      : 'border-gray-300 focus:border-indigo-500 focus:ring-indigo-500'
                  }`}
                >
                  <option value="doctor">Doctor</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
            </div>
            
            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setUserToEdit(null);
                  setShowEditUserModal(false);
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={handleUpdateUser}
              >
                Update User
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete User Confirmation Modal */}
      {showDeleteUserModal && userToDelete && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h2 className="text-xl font-semibold mb-4">Delete User</h2>
            
            <p className="text-gray-700 mb-4">
              Are you sure you want to delete the user "{userToDelete.username}"? 
              This action cannot be undone.
            </p>
            
            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setUserToDelete(null);
                  setShowDeleteUserModal(false);
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-danger"
                onClick={handleDeleteUser}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UsersPage; 