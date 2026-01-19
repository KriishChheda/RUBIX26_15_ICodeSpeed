import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    if (user.token) {
      config.headers.Authorization = `Bearer ${user.token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: (email, password) => 
    apiClient.post('/auth/login', { email, password }),
  
  signup: (name, email, password, role) => 
    apiClient.post('/auth/signup', { name, email, password, role }),
  
  logout: () => 
    apiClient.post('/auth/logout'),
};

// Tests API
export const testsAPI = {
  getAllTests: () => 
    apiClient.get('/tests'),
  
  getTestById: (testId) => 
    apiClient.get(`/tests/${testId}`),
  
  createTest: (testData) => 
    apiClient.post('/tests', testData),
  
  updateTest: (testId, testData) => 
    apiClient.put(`/tests/${testId}`, testData),
  
  deleteTest: (testId) => 
    apiClient.delete(`/tests/${testId}`),
  
  publishTest: (testId) => 
    apiClient.post(`/tests/${testId}/publish`),
};

// Student Tests API
export const studentTestsAPI = {
  getAvailableTests: () => 
    apiClient.get('/student/tests'),
  
  startTest: (testId) => 
    apiClient.post(`/student/tests/${testId}/start`),
  
  submitTest: (testId, answers) => 
    apiClient.post(`/student/tests/${testId}/submit`, { answers }),
  
  getTestResults: (testId) => 
    apiClient.get(`/student/tests/${testId}/results`),
};

// Proctoring API
export const proctoringAPI = {
  sendProctoringData: (sessionId, data) => 
    apiClient.post(`/proctoring/${sessionId}/data`, data),
  
  reportViolation: (sessionId, violation) => 
    apiClient.post(`/proctoring/${sessionId}/violation`, violation),
  
  getSessionLogs: (sessionId) => 
    apiClient.get(`/proctoring/${sessionId}/logs`),
};

// Results API
export const resultsAPI = {
  getStudentResults: (studentId) => 
    apiClient.get(`/results/student/${studentId}`),
  
  getTestResults: (testId) => 
    apiClient.get(`/results/test/${testId}`),
  
  getAllResults: () => 
    apiClient.get('/results'),
};

export default apiClient;