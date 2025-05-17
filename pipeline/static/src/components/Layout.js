import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth, ROLES, canAccessAdminArea } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import {
  HomeIcon,
  CloudArrowUpIcon,
  DocumentTextIcon,
  Bars3Icon,
  XMarkIcon,
  ArrowRightOnRectangleIcon,
  UserCircleIcon,
  UserGroupIcon,
  BeakerIcon,
  Cog6ToothIcon,
  ShieldCheckIcon,
  SunIcon,
  MoonIcon,
  MagnifyingGlassIcon,
  BellIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

const Layout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notifications, setNotifications] = useState(3); // Example notification count
  const { user, logout } = useAuth();
  const { darkMode, toggleDarkMode } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);

  // Handle scroll effect for header
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Base navigation for all users
  const baseNavigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Upload', href: '/upload', icon: CloudArrowUpIcon },
    { name: 'Results', href: '/results', icon: DocumentTextIcon },
    { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
    { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
  ];

  // Admin-only navigation items
  const adminNavigation = [
    { name: 'Users', href: '/users', icon: UserGroupIcon }
  ];

  // Determine which navigation items to show based on user role
  const navigation = canAccessAdminArea(user?.role)
    ? [...baseNavigation, ...adminNavigation]
    : baseNavigation;

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  // Function to get role badge component
  const getRoleBadge = (role) => {
    switch(role) {
      case ROLES.SUPERADMIN:
        return (
          <span className={`ml-1 px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${darkMode ? 'bg-purple-900 text-purple-100' : 'bg-purple-100 text-purple-800'}`}>
            <ShieldCheckIcon className="mr-1 h-3 w-3" />
            Super Admin
          </span>
        );
      case ROLES.ADMIN:
        return (
          <span className={`ml-1 px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${darkMode ? 'bg-blue-900 text-blue-100' : 'bg-blue-100 text-blue-800'}`}>
            <ShieldCheckIcon className="mr-1 h-3 w-3" />
            Admin
          </span>
        );
      default:
        return null;
    }
  };

  // Get user display name
  const getUserDisplayName = () => {
    if (user?.firstName && user?.lastName) {
      return `${user.firstName} ${user.lastName}`;
    } else if (user?.firstName) {
      return user.firstName;
    } else {
      return user?.username || 'User';
    }
  };

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-100'} flex`}>
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-gray-600 bg-opacity-75 md:hidden"
          onClick={() => setSidebarOpen(false)} 
        />
      )}

      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-72 ${darkMode ? 'bg-gray-800' : 'bg-white'} transform transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} shadow-xl`}>
        <div className="h-full flex flex-col">
          {/* Sidebar header with logo and close button */}
          <div className={`flex items-center justify-between h-16 px-6 ${darkMode ? 'bg-gray-900 text-white' : 'bg-indigo-600 text-white'}`}>
            <Link to="/" className="flex items-center">
              <div className="h-8 w-8 rounded-full bg-white flex items-center justify-center text-indigo-600">
                <BeakerIcon className="h-5 w-5" />
              </div>
              <span className="ml-2 text-xl font-semibold text-white">PulmoScan</span>
            </Link>
            <button
              type="button"
              className="md:hidden text-white hover:text-gray-200"
              onClick={() => setSidebarOpen(false)}
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>

          {/* User profile section */}
          <div className={`px-6 py-4 ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <div className="flex items-center space-x-3">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center ${darkMode ? 'bg-indigo-900 text-indigo-200' : 'bg-indigo-100 text-indigo-700'}`}>
                <UserCircleIcon className="h-6 w-6" />
              </div>
              <div>
                <div className="flex items-center">
                  <p className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-800'}`}>{getUserDisplayName()}</p>
                  {user?.role && getRoleBadge(user.role)}
                </div>
                <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{user?.email || user?.username}</p>
              </div>
            </div>
          </div>

          {/* Navigation links */}
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            <div className={`mb-3 px-3 py-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'} text-xs font-semibold uppercase tracking-wider`}>
              Menu
            </div>
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 ${
                    isActive
                      ? darkMode 
                        ? 'bg-indigo-600 text-white' 
                        : 'bg-indigo-50 text-indigo-700'
                      : darkMode 
                        ? 'text-gray-300 hover:bg-gray-700 hover:text-white' 
                        : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <item.icon className={`mr-3 h-5 w-5 transition-colors ${
                    isActive 
                      ? darkMode 
                        ? 'text-white' 
                        : 'text-indigo-600' 
                      : darkMode 
                        ? 'text-gray-400 group-hover:text-gray-300' 
                        : 'text-gray-400 group-hover:text-gray-500'
                  }`} />
                  {item.name}
                  {item.name === 'Results' && (
                    <span className={`ml-auto ${isActive ? 'bg-indigo-300 text-indigo-800' : 'bg-indigo-100 text-indigo-600'} px-2 py-0.5 rounded-full text-xs font-medium`}>
                      New
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Theme toggle and logout */}
          <div className={`p-4 ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="flex items-center justify-between mb-4">
              <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Appearance
              </span>
              <button
                onClick={toggleDarkMode}
                className={`rounded-full p-2 transition-colors ${
                  darkMode
                    ? 'bg-gray-700 text-yellow-300 hover:bg-gray-600'
                    : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                }`}
              >
                {darkMode ? (
                  <SunIcon className="h-5 w-5" />
                ) : (
                  <MoonIcon className="h-5 w-5" />
                )}
              </button>
            </div>
            <button
              onClick={handleLogout}
              className={`w-full flex items-center justify-center px-4 py-2.5 border rounded-lg shadow-sm text-sm font-medium transition-colors ${
                darkMode 
                  ? 'border-gray-600 text-gray-200 bg-gray-700 hover:bg-gray-600' 
                  : 'border-gray-300 text-gray-700 bg-white hover:bg-gray-50'
              }`}
            >
              <ArrowRightOnRectangleIcon className={`h-4 w-4 mr-2 ${darkMode ? 'text-gray-300' : 'text-gray-400'}`} />
              Sign out
            </button>
          </div>
        </div>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col">
        {/* Mobile header */}
        <div className={`md:hidden sticky top-0 z-10 flex items-center justify-between h-16 ${
          darkMode 
            ? 'bg-gray-800 shadow-md' 
            : 'bg-white shadow-sm'
        } px-4`}>
          <button
            type="button"
            className={`${darkMode ? 'text-gray-400 hover:text-gray-300' : 'text-gray-500 hover:text-gray-600'} focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500`}
            onClick={() => setSidebarOpen(true)}
          >
            <Bars3Icon className="h-5 w-5" />
          </button>
          <Link to="/" className="flex items-center">
            <div className="h-8 w-8 rounded-full bg-indigo-600 flex items-center justify-center text-white">
              <BeakerIcon className="h-5 w-5" />
            </div>
            <span className={`ml-2 text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>PulmoScan</span>
          </Link>
          <div className="flex items-center">
            <button className={`p-1.5 rounded-full ${darkMode ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-500 hover:bg-gray-100'} relative`}>
              <BellIcon className="h-5 w-5" />
              {notifications > 0 && (
                <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-500 ring-2 ring-white"></span>
              )}
            </button>
          </div>
        </div>

        {/* Desktop header (only visible on md breakpoint and up) */}
        <div className={`hidden md:block sticky top-0 z-10 ${
          darkMode 
            ? 'bg-gray-800' 
            : 'bg-white'
        } ${scrolled ? 'shadow-md' : ''} transition-shadow duration-200`}>
          <div className="h-16 px-6 flex items-center justify-between">
            {/* Search bar */}
            <div className={`max-w-md w-96 relative rounded-lg ${
              darkMode ? 'bg-gray-700' : 'bg-gray-100'
            }`}>
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <MagnifyingGlassIcon className={`h-5 w-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              </div>
              <input
                type="text"
                placeholder="Search..."
                className={`block w-full pl-10 pr-3 py-2 rounded-lg border-0 ${
                  darkMode 
                    ? 'bg-gray-700 placeholder-gray-400 text-white focus:ring-indigo-500' 
                    : 'bg-gray-100 placeholder-gray-500 text-gray-900 focus:ring-indigo-600'
                } focus:outline-none focus:ring-2 text-sm`}
              />
            </div>

            {/* Right side controls */}
            <div className="flex items-center space-x-4">
              {/* Theme toggle */}
              <button
                onClick={toggleDarkMode}
                className={`p-1.5 rounded-lg ${
                  darkMode
                    ? 'text-gray-300 hover:bg-gray-700'
                    : 'text-gray-500 hover:bg-gray-100'
                }`}
                title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {darkMode ? (
                  <SunIcon className="h-5 w-5" />
                ) : (
                  <MoonIcon className="h-5 w-5" />
                )}
              </button>

              {/* Notifications */}
              <button className={`p-1.5 rounded-lg ${darkMode ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-500 hover:bg-gray-100'} relative`}>
                <BellIcon className="h-5 w-5" />
                {notifications > 0 && (
                  <span className="absolute top-0 right-0 block h-5 w-5 rounded-full bg-red-500 border-2 border-white dark:border-gray-800 flex items-center justify-center text-white text-xs font-bold">
                    {notifications}
                  </span>
                )}
              </button>

              {/* User menu */}
              <div className={`flex items-center px-3 py-1.5 rounded-lg ${
                darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'
              } cursor-pointer transition-colors`}>
                <div className={`h-8 w-8 rounded-full ${
                  darkMode ? 'bg-indigo-800 text-indigo-200' : 'bg-indigo-100 text-indigo-700'
                } flex items-center justify-center mr-2`}>
                  <UserCircleIcon className="h-5 w-5" />
                </div>
                <div>
                  <div className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                    {getUserDisplayName()}
                  </div>
                  <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {user?.role || 'User'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 p-6">
          <Outlet />
        </main>

        {/* Footer */}
        <footer className={`p-4 ${darkMode ? 'bg-gray-800 text-gray-400' : 'bg-white text-gray-500'} border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
          <div className="container mx-auto">
            <div className="flex flex-col md:flex-row justify-between items-center">
              <div className="text-sm">
                &copy; {new Date().getFullYear()} PulmoScan. All rights reserved.
              </div>
              <div className="text-sm mt-2 md:mt-0">
                <a href="#" className="hover:underline">Privacy Policy</a> · <a href="#" className="hover:underline">Terms of Service</a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Layout; 