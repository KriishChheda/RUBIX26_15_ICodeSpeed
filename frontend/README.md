# AI Proctor - Smart Examination Proctoring System

A modern, full-featured examination proctoring system built with React, featuring AI-powered monitoring to detect suspicious activities during online tests.

## 🚀 Features

### Authentication

- Login/Signup with role-based access (Admin/Student)
- Protected routes based on user roles
- Mock authentication (ready for backend integration)

### Admin Dashboard

- **Overview Dashboard**: View statistics, recent tests, and submissions
- **Test Management**: Create, edit, publish, and delete tests
- **Question Builder**: Support for multiple-choice, text, and true/false questions
- **Results Analytics**: View student performance with charts and detailed analytics
- **Export Functionality**: Export results to CSV

### Student Portal

- **Dashboard**: View available tests and performance statistics
- **Test Taking**: Clean, distraction-free test interface
- **Results**: View scores, performance trends, and detailed analytics

### AI Proctoring Features

- **Face Detection**: Real-time monitoring of student presence
- **Multiple Face Detection**: Alerts when multiple people are detected
- **Tab Switching Detection**: Tracks when students leave the test window
- **Risk Score**: Cumulative scoring based on detected violations
- **Violation Logging**: Timestamped records of all suspicious activities
- **Live Camera Feed**: Continuous monitoring during tests

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/         # Reusable components
│   │   ├── AdminLayout.jsx
│   │   └── StudentLayout.jsx
│   ├── contexts/           # React contexts
│   │   └── AuthContext.jsx
│   ├── pages/             # Page components
│   │   ├── Login.jsx
│   │   ├── Signup.jsx
│   │   ├── AdminDashboard.jsx
│   │   ├── AdminTests.jsx
│   │   ├── CreateTest.jsx
│   │   ├── AdminResults.jsx
│   │   ├── StudentDashboard.jsx
│   │   ├── StudentTests.jsx
│   │   ├── TakeTest.jsx
│   │   └── StudentResults.jsx
│   ├── services/          # API and data services
│   │   ├── api.js         # API client (ready for backend)
│   │   └── mockData.js    # Mock data for development
│   ├── App.jsx            # Main app with routing
│   ├── main.jsx          # Entry point
│   └── index.css         # Global styles with Tailwind
├── vite.config.js        # Vite configuration (only config needed!)
├── tailwind.config.js    # Minimal Tailwind config
├── postcss.config.js     # PostCSS config
└── package.json
```

## 🛠️ Installation

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Setup

1. **Navigate to the frontend directory**

   ```bash
   cd frontend
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Start the development server**

   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:3000`

## 🔑 Demo Credentials

### Admin Account

- Email: `admin@test.com`
- Password: Any password (mock auth)

### Student Account

- Email: `student@test.com`
- Password: Any password (mock auth)

Or sign up with any email - use "admin" in the email for admin role, otherwise defaults to student.

## 🎨 Design Features

- **Modern UI**: Clean, professional interface with Tailwind CSS
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Standard Tailwind**: Uses default Tailwind classes (slate colors, etc.)
- **Smooth Animations**: Subtle transitions and loading states
- **Accessible**: Keyboard navigation and screen reader support
- **Simple Config**: Only needs vite.config.js - no complex Tailwind customization

## ⚙️ Configuration

The project uses a **minimal configuration approach**:

- **vite.config.js**: Main configuration file (server, proxy, etc.)
- **tailwind.config.js**: Minimal config (just content paths)
- **postcss.config.js**: Simple PostCSS setup
- **No custom theme**: Uses default Tailwind colors and utilities

All styling is done with standard Tailwind utility classes!

## 🔌 Backend Integration

The frontend is designed to work with mock data but is fully ready for backend integration:

### API Service (`src/services/api.js`)

All API calls are centralized and ready to connect:

```javascript
// Example: Connect to your backend
const API_BASE_URL = "http://localhost:5000/api";

// All endpoints are pre-configured:
// - authAPI.login(email, password)
// - testsAPI.getAllTests()
// - proctoringAPI.sendProctoringData(sessionId, data)
// ... and more
```

### Integration Steps:

1. **Update API Base URL**
   - Set `VITE_API_URL` in `.env`
   - Or update `API_BASE_URL` in `src/services/api.js`

2. **Replace Mock Auth**
   - Update `login()` and `signup()` in `AuthContext.jsx`
   - Remove mock data returns, use actual API responses

3. **Connect AI Proctoring**
   - In `TakeTest.jsx`, replace simulated checks with actual AI API calls
   - Send webcam frames to your backend AI service
   - Process real-time detections

4. **Update Data Fetching**
   - Replace mock data imports with actual API calls
   - Update `useEffect` hooks to use `api.js` functions

## 📦 Build for Production

```bash
npm run build
```

This creates an optimized production build in the `dist` folder.

## 🧪 Technology Stack

- **React 18**: Modern React with hooks
- **React Router**: Client-side routing
- **Tailwind CSS**: Utility-first styling (default config)
- **Vite**: Fast build tool and dev server
- **Recharts**: Beautiful charts and analytics
- **React Webcam**: Camera integration for proctoring
- **Lucide React**: Modern icon library
- **Axios**: HTTP client for API calls

## 📱 Features by Role

### Admin Features

- ✅ Create and manage tests
- ✅ Add multiple question types
- ✅ Set test duration and scheduling
- ✅ Publish/unpublish tests
- ✅ View all student results
- ✅ Analyze performance with charts
- ✅ Export results to CSV
- ✅ Review proctoring violations

### Student Features

- ✅ View available tests
- ✅ Take tests with proctoring
- ✅ Real-time violation warnings
- ✅ View results and performance
- ✅ Track progress over time
- ✅ Performance analytics with charts

## 🔐 Security Features

- Protected routes based on authentication
- Role-based access control
- Token-based authentication (ready for JWT)
- Secure API client with interceptors
- Camera permission handling
- Data validation on forms

## 🎯 Proctoring Detection (Ready for AI Integration)

The system includes hooks for these AI features:

1. **Face Detection**: Verify student presence
2. **Multiple Face Detection**: Detect additional people
3. **Head Pose & Gaze Tracking**: Monitor attention
4. **Tab Switching**: Track window focus
5. **Session Risk Scoring**: Cumulative violation scoring
6. **Audit Logs**: Timestamped violation records

## 🚧 TODO: Backend Requirements

To complete the system, implement:

1. **Authentication API**
   - POST `/api/auth/login`
   - POST `/api/auth/signup`
   - POST `/api/auth/logout`

2. **Tests API**
   - GET/POST `/api/tests`
   - GET/PUT/DELETE `/api/tests/:id`
   - POST `/api/tests/:id/publish`

3. **Student Tests API**
   - GET `/api/student/tests`
   - POST `/api/student/tests/:id/start`
   - POST `/api/student/tests/:id/submit`

4. **Proctoring API**
   - POST `/api/proctoring/:sessionId/data`
   - POST `/api/proctoring/:sessionId/violation`
   - GET `/api/proctoring/:sessionId/logs`

5. **AI Detection Service**
   - Face detection model
   - Head pose estimation
   - Multiple face detection
   - Gaze tracking

## 📄 License

This project is built for educational/demonstration purposes.

## 🤝 Contributing

This is a demonstration project. Feel free to use it as a starting point for your own proctoring system!

---

Built with ❤️ using React, Vite, and Tailwind CSS (default configuration)
