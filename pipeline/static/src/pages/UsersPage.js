import React, { useState, useEffect } from 'react';
import { useAuth, ROLES, canManageAdmins, canManageDoctors, canManageRole } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import axios from 'axios';
import { PlusIcon, PencilIcon, TrashIcon, ExclamationCircleIcon, ShieldCheckIcon, UserIcon } from '@heroicons/react/24/outline';
import { toast } from '../utils/toast';
import Modal from '../components/Modal';

const UsersPage = () => {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateUserModal, setShowCreateUserModal] = useState(false);
  const [showEditUserModal, setShowEditUserModal] = useState(false);
  const [showDeleteUserModal, setShowDeleteUserModal] = useState(false);
  const [userToEdit, setUserToEdit] = useState({
    username: '',
    role: '',
    email: '',
    description: '',
    plan: 'usage_based',
    password: ''
  });
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
  
  const PLANS = {
    USAGE_BASED: 'usage_based',
    ONE_TIME: 'one_time',
    SUBSCRIPTION: 'subscription'
  };

  const PLAN_LABELS = {
    usage_based: 'Usage Based',
    one_time: 'One Time Payment',
    subscription: 'Subscription'
  };

  const [newUser, setNewUser] = useState({
    username: '',
    password: '',
    email: '',
    role: 'doctor',
    description: '',
    plan: PLANS.USAGE_BASED
  });
  
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
  if (!currentUser || !userCanManageDoctors) {
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
  
  // Get available roles based on current user's role
  const getAvailableRoles = () => {
    const roles = [];
    
    // If current user is not defined yet, return an empty array
    if (!currentUser) {
      return roles;
    }
    
    // Add roles based on permissions
    if (canManageRole(currentUser.role, ROLES.DOCTOR)) {
      roles.push({ value: ROLES.DOCTOR, label: 'Doctor' });
    }
    if (canManageRole(currentUser.role, ROLES.RADIOLOGIST)) {
      roles.push({ value: ROLES.RADIOLOGIST, label: 'Radiologist' });
    }
    if (canManageRole(currentUser.role, ROLES.ADMIN)) {
      roles.push({ value: ROLES.ADMIN, label: 'Admin' });
    }
    if (canManageRole(currentUser.role, ROLES.SUPERADMIN)) {
      roles.push({ value: ROLES.SUPERADMIN, label: 'Super Admin' });
    }
    
    return roles;
  };
  
  // Create a new user
  const handleCreateUser = async (e) => {
    e.preventDefault();
    if (!canManageRole(newUser.role)) {
      toast.error('You do not have permission to create users with this role');
      return;
    }

    try {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newUser),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to create user');
      }

      const data = await response.json();
      setUsers([...users, data]);
      setShowCreateUserModal(false);
      setNewUser({
        username: '',
        password: '',
        email: '',
        role: 'doctor',
        description: '',
        plan: PLANS.USAGE_BASED
      });
      toast.success('User created successfully');
    } catch (error) {
      toast.error(error.message);
    }
  };
  
  // Update an existing user
  const handleUpdateUser = async (e) => {
    e.preventDefault();
    
    // Ensure currentUser exists
    if (!currentUser) {
      toast.error('You must be logged in to perform this action');
      return;
    }
    
    try {
      // Check if user has permission for the selected role
      if (!canManageRole(userToEdit.role)) {
        toast.error('You do not have permission to assign this role');
        return;
      }
      
      // Prevent downgrading superadmins if not a superadmin
      if (userToEdit.role === ROLES.SUPERADMIN && currentUser.role !== ROLES.SUPERADMIN) {
        toast.error('You cannot modify a super admin account');
        return;
      }

      const response = await fetch(`/api/users/${userToEdit.username}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          role: userToEdit.role,
          email: userToEdit.email,
          description: userToEdit.description,
          plan: userToEdit.plan,
          ...(userToEdit.password ? { password: userToEdit.password } : {})
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to update user');
      }

      // Refresh users list
      fetchUsers();
      setShowEditUserModal(false);
      toast.success('User updated successfully');
    } catch (error) {
      toast.error(error.message || 'Failed to update user');
    }
  };
  
  // Delete a user
  const handleDeleteUser = async () => {
    if (!userToDelete) return;
    
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
    setUserToEdit({
      ...user,
      password: '', // Don't prefill password for security
    });
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
        <div className="mt-8 flex flex-col">
          <div className="-my-2 -mx-4 overflow-x-auto sm:-mx-6 lg:-mx-8">
            <div className="inline-block min-w-full py-2 align-middle md:px-6 lg:px-8">
              <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300 dark:divide-gray-700">
                  <thead className={`${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
                    <tr>
                      <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 dark:text-gray-100 sm:pl-6">
                        Username
                      </th>
                      <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        Role
                      </th>
                      <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        Email
                      </th>
                      <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        Plan
                      </th>
                      <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        Description
                      </th>
                      <th scope="col" className="relative py-3.5 pl-3 pr-4 sm:pr-6">
                        <span className="sr-only">Actions</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y divide-gray-200 dark:divide-gray-700 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
                    {users.map((user) => (
                      <tr key={user.username}>
                        <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 dark:text-gray-100 sm:pl-6">
                          {user.username}
                        </td>
                        <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500 dark:text-gray-400">
                          {user.role}
                        </td>
                        <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500 dark:text-gray-400">
                          {user.email}
                        </td>
                        <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500 dark:text-gray-400">
                          {PLAN_LABELS[user.plan] || user.plan}
                        </td>
                        <td className="px-3 py-4 text-sm text-gray-500 dark:text-gray-400">
                          {user.description || '-'}
                        </td>
                        <td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">
                          <button
                            onClick={() => openEditUserModal(user)}
                            className="text-indigo-600 hover:text-indigo-900 dark:text-indigo-400 dark:hover:text-indigo-300 mr-4"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => openDeleteUserModal(user)}
                            className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Create User Modal */}
      <Modal
        isOpen={showCreateUserModal}
        onClose={() => setShowCreateUserModal(false)}
        title="Create New User"
      >
        <form onSubmit={handleCreateUser} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Username
            </label>
            <input
              type="text"
              value={newUser.username}
              onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Password
            </label>
            <input
              type="password"
              value={newUser.password}
              onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Email
            </label>
            <input
              type="email"
              value={newUser.email}
              onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Role
            </label>
            <select
              value={newUser.role}
              onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            >
              {getAvailableRoles().map(role => (
                <option key={role.value} value={role.value}>
                  {role.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Description
            </label>
            <textarea
              value={newUser.description}
              onChange={(e) => setNewUser({ ...newUser, description: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
              rows="3"
              placeholder="Enter user description or notes"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Plan
            </label>
            <select
              value={newUser.plan}
              onChange={(e) => setNewUser({ ...newUser, plan: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            >
              {Object.entries(PLAN_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={() => setShowCreateUserModal(false)}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-600"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Create User
            </button>
          </div>
        </form>
      </Modal>

      {/* Edit User Modal */}
      <Modal
        isOpen={showEditUserModal}
        onClose={() => {
          setShowEditUserModal(false);
          // Reset form state
          setUserToEdit({
            username: '',
            role: '',
            email: '',
            description: '',
            plan: 'usage_based',
            password: ''
          });
        }}
        title="Edit User"
      >
        <form onSubmit={handleUpdateUser} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Username
            </label>
            <input
              type="text"
              value={userToEdit.username || ''}
              disabled
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm bg-gray-100 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300 sm:text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Role
            </label>
            <select
              value={userToEdit.role || ''}
              onChange={(e) => setUserToEdit({ ...userToEdit, role: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            >
              {getAvailableRoles().map(role => (
                <option key={role.value} value={role.value}>
                  {role.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Email
            </label>
            <input
              type="email"
              value={userToEdit.email || ''}
              onChange={(e) => setUserToEdit({ ...userToEdit, email: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Description
            </label>
            <textarea
              value={userToEdit.description || ''}
              onChange={(e) => setUserToEdit({ ...userToEdit, description: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
              rows="3"
              placeholder="Enter user description or notes"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Plan
            </label>
            <select
              value={userToEdit.plan || 'usage_based'}
              onChange={(e) => setUserToEdit({ ...userToEdit, plan: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            >
              {Object.entries(PLAN_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              New Password (leave blank to keep current)
            </label>
            <input
              type="password"
              value={userToEdit.password || ''}
              onChange={(e) => setUserToEdit({ ...userToEdit, password: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
            />
          </div>
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={() => {
                setShowEditUserModal(false);
                // Reset form state
                setUserToEdit({
                  username: '',
                  role: '',
                  email: '',
                  description: '',
                  plan: 'usage_based',
                  password: ''
                });
              }}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-600"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Save Changes
            </button>
          </div>
        </form>
      </Modal>

      {/* Delete User Confirmation Modal */}
      <Modal
        isOpen={showDeleteUserModal && userToDelete !== null}
        onClose={() => {
          setShowDeleteUserModal(false);
          setUserToDelete(null);
        }}
        title="Delete User"
      >
        {userToDelete && (
          <div>
            <p className={`text-gray-700 dark:text-gray-300 mb-4`}>
              Are you sure you want to delete the user "{userToDelete.username}"? 
              This action cannot be undone.
            </p>
            
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => {
                  setShowDeleteUserModal(false);
                  setUserToDelete(null);
                }}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-600"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteUser}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-md shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
              >
                Delete
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default UsersPage; 