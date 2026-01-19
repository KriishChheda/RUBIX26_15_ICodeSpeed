import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Search,
  Clock,
  FileText,
  PlayCircle,
  CheckCircle,
  AlertCircle,
  Calendar
} from 'lucide-react';
import { mockStudentTests } from '../services/mockData';

const StudentTests = () => {
  const [tests, setTests] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setTests(mockStudentTests);
      setLoading(false);
    }, 500);
  }, []);

  const filteredTests = tests.filter(test => {
    const matchesSearch = test.title.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === 'all' || test.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  const getStatusBadge = (test) => {
    if (test.status === 'completed') {
      return <span className="badge badge-success">Completed</span>;
    } else if (test.attempts >= test.maxAttempts && test.maxAttempts > 0) {
      return <span className="badge badge-danger">No attempts left</span>;
    } else {
      return <span className="badge badge-info">Available</span>;
    }
  };

  const canStartTest = (test) => {
    return test.status === 'available' && test.attempts < test.maxAttempts;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-dark-900 border-t-transparent"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-display font-bold text-dark-900">
          My Tests
        </h2>
        <p className="text-dark-600 mt-1">
          View and take your available examinations
        </p>
      </div>

      {/* Filters */}
      <div className="card p-6">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
            <input
              type="text"
              placeholder="Search tests..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input-field pl-12 w-full"
            />
          </div>
          <div className="sm:w-48">
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="input-field w-full"
            >
              <option value="all">All Tests</option>
              <option value="available">Available</option>
              <option value="completed">Completed</option>
            </select>
          </div>
        </div>
      </div>

      {/* Tests List */}
      <div className="space-y-4">
        {filteredTests.length === 0 ? (
          <div className="card p-12 text-center">
            <FileText className="w-16 h-16 text-dark-300 mx-auto mb-4" />
            <h3 className="text-xl font-display font-bold text-dark-900 mb-2">
              No tests found
            </h3>
            <p className="text-dark-600">
              {searchTerm || filterStatus !== 'all'
                ? 'Try adjusting your filters'
                : 'No tests are currently available'}
            </p>
          </div>
        ) : (
          filteredTests.map((test) => (
            <div key={test.id} className="card p-6 hover:shadow-lg transition-all">
              <div className="flex flex-col lg:flex-row lg:items-center gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-xl font-display font-bold text-dark-900">
                      {test.title}
                    </h3>
                    {getStatusBadge(test)}
                  </div>
                  
                  <div className="flex flex-wrap gap-4 text-sm text-dark-600 mb-3">
                    <span className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {test.duration} minutes
                    </span>
                    <span className="flex items-center gap-1">
                      <FileText className="w-4 h-4" />
                      {test.totalQuestions} questions
                    </span>
                    <span>
                      Total Marks: {test.totalMarks}
                    </span>
                  </div>

                  {test.startDate && test.endDate && (
                    <div className="flex items-center gap-2 text-sm text-dark-600 mb-2">
                      <Calendar className="w-4 h-4" />
                      <span>
                        Available: {new Date(test.startDate).toLocaleDateString()} - {new Date(test.endDate).toLocaleDateString()}
                      </span>
                    </div>
                  )}

                  {test.maxAttempts > 0 && (
                    <div className="text-sm text-dark-600">
                      Attempts: {test.attempts}/{test.maxAttempts}
                    </div>
                  )}

                  {test.status === 'completed' && test.score && (
                    <div className="mt-3 p-3 bg-green-50 rounded-lg inline-block">
                      <div className="text-sm text-green-800 font-medium">
                        Your Score: {test.score}/{test.totalMarks} ({((test.score / test.totalMarks) * 100).toFixed(1)}%)
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex lg:flex-col gap-2">
                  {canStartTest(test) ? (
                    <Link
                      to={`/student/tests/${test.id}`}
                      className="btn-primary flex items-center gap-2 justify-center"
                    >
                      <PlayCircle className="w-4 h-4" />
                      Start Test
                    </Link>
                  ) : test.status === 'completed' ? (
                    <Link
                      to={`/student/results`}
                      className="btn-secondary flex items-center gap-2 justify-center"
                    >
                      <CheckCircle className="w-4 h-4" />
                      View Results
                    </Link>
                  ) : (
                    <button
                      disabled
                      className="px-6 py-3 bg-dark-100 text-dark-400 rounded-lg font-medium cursor-not-allowed"
                    >
                      Unavailable
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Important Instructions */}
      {filteredTests.some(t => canStartTest(t)) && (
        <div className="card p-6 border-l-4 border-yellow-500">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 bg-yellow-100 rounded-lg flex items-center justify-center flex-shrink-0">
              <AlertCircle className="w-5 h-5 text-yellow-600" />
            </div>
            <div className="flex-1">
              <h4 className="font-semibold text-dark-900 mb-2">
                Before Starting Your Test
              </h4>
              <ul className="text-sm text-dark-600 space-y-1">
                <li>• Ensure stable internet connection</li>
                <li>• Allow camera and microphone permissions</li>
                <li>• Close all other applications and browser tabs</li>
                <li>• Find a quiet, well-lit environment</li>
                <li>• Keep your face visible in the camera at all times</li>
                <li>• Do not leave the test window during the examination</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StudentTests;