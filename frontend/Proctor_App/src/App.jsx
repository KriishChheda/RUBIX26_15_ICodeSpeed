// import React from 'react';
// import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
// import { useAuth } from './contexts/AuthContext';

// // Pages
// import Login from './pages/Login';
// import Signup from './pages/Signup';

// // Admin
// import AdminLayout from './components/AdminLayout';
// import AdminDashboard from './pages/AdminDashboard';
// import AdminTests from './pages/AdminTests';
// import CreateTest from './pages/CreateTest';
// import AdminResults from './pages/AdminResults';

// // Student
// import StudentLayout from './components/StudentLayout';
// import StudentDashboard from './pages/StudentDashboard';
// import StudentTests from './pages/StudentTests';
// import TakeTest from './pages/TakeTest';
// import StudentResults from './pages/StudentResults';

// // Protected Route Component
// const ProtectedRoute = ({ children, requiredRole }) => {
//   const { isAuthenticated, user, loading } = useAuth();

//   if (loading) {
//     return (
//       <div className="min-h-screen flex items-center justify-center">
//         <div className="animate-spin rounded-full h-12 w-12 border-4 border-dark-900 border-t-transparent"></div>
//       </div>
//     );
//   }

//   if (!isAuthenticated) {
//     return <Navigate to="/login" replace />;
//   }

//   if (requiredRole && user?.role !== requiredRole) {
//     // Redirect to appropriate dashboard based on role
//     if (user?.role === 'admin') {
//       return <Navigate to="/admin/dashboard" replace />;
//     } else {
//       return <Navigate to="/student/dashboard" replace />;
//     }
//   }

//   return children;
// };

// // Public Route Component (redirect if already authenticated)
// const PublicRoute = ({ children }) => {
//   const { isAuthenticated, user, loading } = useAuth();

//   if (loading) {
//     return (
//       <div className="min-h-screen flex items-center justify-center">
//         <div className="animate-spin rounded-full h-12 w-12 border-4 border-dark-900 border-t-transparent"></div>
//       </div>
//     );
//   }

//   if (isAuthenticated) {
//     // Redirect to appropriate dashboard based on role
//     if (user?.role === 'admin') {
//       return <Navigate to="/admin/dashboard" replace />;
//     } else {
//       return <Navigate to="/student/dashboard" replace />;
//     }
//   }

//   return children;
// };

// function App() {
//   return (
//     <AuthProvider>
//       <BrowserRouter>
//         <Routes>
//           {/* Public Routes */}
//           <Route path="/" element={<Navigate to="/login" replace />} />
//           <Route
//             path="/login"
//             element={
//               <PublicRoute>
//                 <Login />
//               </PublicRoute>
//             }
//           />
//           <Route
//             path="/signup"
//             element={
//               <PublicRoute>
//                 <Signup />
//               </PublicRoute>
//             }
//           />

//           {/* Admin Routes */}
//           <Route
//             path="/admin"
//             element={
//               <ProtectedRoute requiredRole="admin">
//                 <AdminLayout />
//               </ProtectedRoute>
//             }
//           >
//             <Route index element={<Navigate to="/admin/dashboard" replace />} />
//             <Route path="dashboard" element={<AdminDashboard />} />
//             <Route path="tests" element={<AdminTests />} />
//             <Route path="tests/create" element={<CreateTest />} />
//             <Route path="tests/:testId/edit" element={<CreateTest />} />
//             <Route path="results" element={<AdminResults />} />
//           </Route>

//           {/* Student Routes */}
//           <Route
//             path="/student"
//             element={
//               <ProtectedRoute requiredRole="student">
//                 <StudentLayout />
//               </ProtectedRoute>
//             }
//           >
//             <Route index element={<Navigate to="/student/dashboard" replace />} />
//             <Route path="dashboard" element={<StudentDashboard />} />
//             <Route path="tests" element={<StudentTests />} />
//             <Route path="tests/:testId" element={<TakeTest />} />
//             <Route path="results" element={<StudentResults />} />
//           </Route>

//           {/* 404 */}
//           <Route path="*" element={<Navigate to="/login" replace />} />
//         </Routes>
//       </BrowserRouter>
//     </AuthProvider>
//   );
// }

// export default App;



import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Pages
import Login from './pages/Login';
import Signup from './pages/Signup';

// Admin
import AdminLayout from './components/AdminLayout';
import AdminDashboard from './pages/AdminDashboard';
import AdminTests from './pages/AdminTests';
import CreateTest from './pages/CreateTest';
import AdminResults from './pages/AdminResults';

// Student
import StudentLayout from './components/StudentLayout';
import StudentDashboard from './pages/StudentDashboard';
import StudentTests from './pages/StudentTests';
import TakeTest from './pages/TakeTest';
import StudentResults from './pages/StudentResults';

// Protected Route Component
const ProtectedRoute = ({ children, requiredRole }) => {
  const { isAuthenticated, user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-dark-900 border-t-transparent"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (requiredRole && user?.role !== requiredRole) {
    if (user?.role === 'admin') {
      return <Navigate to="/admin/dashboard" replace />;
    } else {
      return <Navigate to="/student/dashboard" replace />;
    }
  }

  return children;
};

// Public Route Component
const PublicRoute = ({ children }) => {
  const { isAuthenticated, user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-dark-900 border-t-transparent"></div>
      </div>
    );
  }

  if (isAuthenticated) {
    if (user?.role === 'admin') {
      return <Navigate to="/admin/dashboard" replace />;
    } else {
      return <Navigate to="/student/dashboard" replace />;
    }
  }

  return children;
};

console.log('App.jsx loaded');

function App() {
  console.log('App rendering');
  const { user } = useAuth();
  console.log('Current user:', user);
  return (
    <NotificationProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route
            path="/login"
            element={
              <PublicRoute>
                <Login />
              </PublicRoute>
            }
          />
          <Route
            path="/signup"
            element={
              <PublicRoute>
                <Signup />
              </PublicRoute>
            }
          />

          {/* Admin Routes */}
          <Route
            path="/admin"
            element={
              <ProtectedRoute requiredRole="admin">
                <AdminLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Navigate to="/admin/dashboard" replace />} />
            <Route path="dashboard" element={<AdminDashboard />} />
            <Route path="tests" element={<AdminTests />} />
            <Route path="tests/create" element={<CreateTest />} />
            <Route path="tests/:testId/edit" element={<CreateTest />} />
            <Route path="results" element={<AdminResults />} />
          </Route>

          {/* Student Routes */}
          <Route
            path="/student"
            element={
              <ProtectedRoute requiredRole="student">
                <StudentLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Navigate to="/student/dashboard" replace />} />
            <Route path="dashboard" element={<StudentDashboard />} />
            <Route path="tests" element={<StudentTests />} />
            <Route path="tests/:testId" element={<TakeTest />} />
            <Route path="results" element={<StudentResults />} />
          </Route>

          {/* 404 */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </BrowserRouter>
    </NotificationProvider>
  );
}

export default App;