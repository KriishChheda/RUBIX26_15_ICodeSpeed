import React, { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../services/api'; // Import your API service

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for stored user data on mount
    const storedUser = localStorage.getItem('user');
    console.log('Stored user from localStorage:', storedUser);
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        console.log('Parsed user:', parsedUser);
        setUser(parsedUser);
      } catch (error) {
        console.error('Error parsing stored user:', error);
        localStorage.removeItem('user');
      }
    }
    setLoading(false);
  }, []);

  // const login = async (email, password) => {
  //   try {
  //     // TODO: Replace with actual API call
  //     // const response = await fetch('/api/auth/login', {
  //     //   method: 'POST',
  //     //   headers: { 'Content-Type': 'application/json' },
  //     //   body: JSON.stringify({ email, password })
  //     // });
  //     // const data = await response.json();
      
  //     // Mock authentication for now
  //     const mockUser = {
  //       id: '1',
  //       email: email,
  //       name: email.split('@')[0],
  //       role: email.includes('admin') ? 'admin' : 'student',
  //       token: 'mock-jwt-token'
  //     };

  //     setUser(mockUser);
  //     localStorage.setItem('user', JSON.stringify(mockUser));
  //     return { success: true, user: mockUser };
  //   } catch (error) {
  //     console.error('Login error:', error);
  //     return { success: false, error: error.message };
  //   }
  // };

  // const signup = async (name, email, password, role = 'student') => {
  //   try {
  //     // TODO: Replace with actual API call
  //     // const response = await fetch('/api/auth/signup', {
  //     //   method: 'POST',
  //     //   headers: { 'Content-Type': 'application/json' },
  //     //   body: JSON.stringify({ name, email, password, role })
  //     // });
  //     // const data = await response.json();
      
  //     // Mock signup for now
  //     const mockUser = {
  //       id: Date.now().toString(),
  //       name,
  //       email,
  //       role,
  //       token: 'mock-jwt-token'
  //     };

  //     setUser(mockUser);
  //     localStorage.setItem('user', JSON.stringify(mockUser));
  //     return { success: true, user: mockUser };
  //   } catch (error) {
  //     console.error('Signup error:', error);
  //     return { success: false, error: error.message };
  //   }
  // };

  const login = async (email, password) => {
  try {
    const response = await authAPI.login(email, password);
    const { user, token } = response.data; // Ensure this matches Django's response structure
    
    const userData = { ...user, token };
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
    return { success: true, user: userData };
  } catch (error) {
    return { success: false, error: error.response?.data?.message || 'Login failed' };
  }
};

const signup = async (name, email, password, role = 'student') => {
  try {
    const response = await authAPI.signup(name, email, password, role);
    console.log('Signup response:', response.data);
    const { user, token } = response.data;
    
    const userData = { ...user, token };
    console.log('Storing user data:', userData);
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
    return { success: true, user: userData };
  } catch (error) {
    console.error('Signup error:', error);
    return { success: false, error: error.response?.data?.error || 'Signup failed' };
  }
};

  // const logout = () => {
  //   setUser(null);
  //   localStorage.removeItem('user');
  // };

  const logout = async () => {
  try {
    await authAPI.logout();
  } catch (error) {
    console.error('Logout API error:', error);
  } finally {
    // Always clear local state regardless of API success/failure
    setUser(null);
    localStorage.removeItem('user');
  }
};

  const value = {
    user,
    loading,
    login,
    signup,
    logout,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
    isStudent: user?.role === 'student'
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};