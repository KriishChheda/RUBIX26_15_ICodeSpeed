import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Eye, EyeOff, ShieldCheck, Lock, Mail } from 'lucide-react';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const result = await login(email, password);
      if (result.success) {
        if (result.user.role === 'admin') {
          navigate('/admin/dashboard');
        } else {
          navigate('/student/dashboard');
        }
      } else {
        setError(result.error || 'Login failed');
      }
    } catch (err) {
      setError('An error occurred during login');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white flex">
      {/* Left Side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-dark-900 p-12 flex-col justify-between">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <ShieldCheck className="w-10 h-10 text-white" />
            <span className="text-2xl font-display font-bold text-white">AI Proctor</span>
          </div>
          
          <div className="max-w-md">
            <h1 className="text-5xl font-display font-bold text-white mb-6 leading-tight">
              Smart Examination
              <br />
              Proctoring System
            </h1>
            <p className="text-lg text-dark-300 leading-relaxed">
              Ensure integrity in online examinations with AI-powered monitoring 
              that detects suspicious activities in real-time.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6 max-w-md">
          <div className="bg-dark-800 p-4 rounded-xl">
            <div className="text-3xl font-bold text-white mb-1">99.8%</div>
            <div className="text-sm text-dark-400">Accuracy Rate</div>
          </div>
          <div className="bg-dark-800 p-4 rounded-xl">
            <div className="text-3xl font-bold text-white mb-1">24/7</div>
            <div className="text-sm text-dark-400">Monitoring</div>
          </div>
        </div>
      </div>

      {/* Right Side - Login Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center gap-3 mb-8">
            <ShieldCheck className="w-8 h-8 text-dark-900" />
            <span className="text-xl font-display font-bold text-dark-900">AI Proctor</span>
          </div>

          <div className="mb-8">
            <h2 className="text-3xl font-display font-bold text-dark-900 mb-2">
              Welcome back
            </h2>
            <p className="text-dark-600">
              Enter your credentials to access your account
            </p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg">
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-dark-700 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="input-field pl-12"
                  placeholder="you@example.com"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-700 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="input-field pl-12 pr-12"
                  placeholder="Enter your password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-dark-400 hover:text-dark-600"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between text-sm">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="w-4 h-4 rounded border-dark-300" />
                <span className="text-dark-700">Remember me</span>
              </label>
              <Link to="/forgot-password" className="text-dark-900 font-medium hover:underline">
                Forgot password?
              </Link>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-dark-600">
              Don't have an account?{' '}
              <Link to="/signup" className="text-dark-900 font-medium hover:underline">
                Sign up
              </Link>
            </p>
          </div>

          {/* Quick Login Hints */}
          <div className="mt-8 p-4 bg-dark-50 rounded-lg">
            <p className="text-xs text-dark-600 font-medium mb-2">Demo Credentials:</p>
            <p className="text-xs text-dark-500">Admin: admin@test.com / any password</p>
            <p className="text-xs text-dark-500">Student: student@test.com / any password</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;