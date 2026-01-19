import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Calendar,
  Clock,
  TrendingUp,
  AlertCircle,
  PlayCircle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { mockStudentTests } from '../services/mockData';

const StudentDashboard = () => {
  const [stats, setStats] = useState({
    availableTests: 0,
    completedTests: 0,
    averageScore: 0,
    upcomingTests: [],
    recentTests: []
  });

  useEffect(() => {
    const available = mockStudentTests.filter(t => t.status === 'available').length;
    const completed = mockStudentTests.filter(t => t.status === 'completed').length;
    const completedTestsData = mockStudentTests.filter(t => t.status === 'completed');
    const avgScore = completedTestsData.length > 0
      ? completedTestsData.reduce((sum, t) => sum + (t.score / t.totalMarks * 100), 0) / completedTestsData.length
      : 0;

    setStats({
      availableTests: available,
      completedTests: completed,
      averageScore: avgScore.toFixed(1),
      upcomingTests: mockStudentTests.filter(t => t.status === 'available').slice(0, 3),
      recentTests: mockStudentTests.slice(0, 3)
    });
  }, []);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Welcome Section */}
      <div className="bg-dark-900 text-white rounded-2xl p-8">
        <h2 className="text-3xl font-display font-bold mb-2">
          Ready to take your test?
        </h2>
        <p className="text-dark-300 mb-6">
          Here's an overview of your examinations and performance.
        </p>
        <div className="flex flex-wrap gap-4">
          <Link
            to="/student/tests"
            className="px-6 py-3 bg-white text-dark-900 rounded-lg font-medium 
                     hover:bg-dark-100 transition-all"
          >
            View All Tests
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
              <Calendar className="w-6 h-6 text-blue-600" />
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-dark-900 mb-1">
            {stats.availableTests}
          </div>
          <div className="text-sm text-dark-600">Available Tests</div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
              <CheckCircle className="w-6 h-6 text-green-600" />
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-dark-900 mb-1">
            {stats.completedTests}
          </div>
          <div className="text-sm text-dark-600">Completed Tests</div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-purple-600" />
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-dark-900 mb-1">
            {stats.averageScore}%
          </div>
          <div className="text-sm text-dark-600">Average Score</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upcoming Tests */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-display font-bold text-dark-900">
              Upcoming Tests
            </h3>
            <Link
              to="/student/tests"
              className="text-sm text-dark-900 font-medium hover:underline"
            >
              View all
            </Link>
          </div>

          {stats.upcomingTests.length === 0 ? (
            <div className="text-center py-8">
              <AlertCircle className="w-12 h-12 text-dark-300 mx-auto mb-3" />
              <p className="text-dark-600">No upcoming tests</p>
            </div>
          ) : (
            <div className="space-y-4">
              {stats.upcomingTests.map((test) => (
                <div
                  key={test.id}
                  className="p-4 bg-dark-50 rounded-lg hover:bg-dark-100 transition-colors"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h4 className="font-semibold text-dark-900 mb-1">
                        {test.title}
                      </h4>
                      <div className="flex items-center gap-3 text-sm text-dark-600">
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {test.duration} min
                        </span>
                        <span>{test.totalQuestions} questions</span>
                      </div>
                    </div>
                    <span className="badge badge-info">Available</span>
                  </div>
                  <Link
                    to={`/student/tests/${test.id}`}
                    className="btn-primary w-full flex items-center justify-center gap-2"
                  >
                    <PlayCircle className="w-4 h-4" />
                    Start Test
                  </Link>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recent Activity */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-display font-bold text-dark-900">
              Recent Activity
            </h3>
            <Link
              to="/student/results"
              className="text-sm text-dark-900 font-medium hover:underline"
            >
              View all
            </Link>
          </div>

          <div className="space-y-4">
            {stats.recentTests.map((test) => (
              <div
                key={test.id}
                className="flex items-center justify-between p-4 bg-dark-50 rounded-lg"
              >
                <div className="flex-1">
                  <h4 className="font-semibold text-dark-900 mb-1">
                    {test.title}
                  </h4>
                  <div className="text-sm text-dark-600">
                    {test.status === 'completed' ? (
                      <span className="flex items-center gap-1">
                        <CheckCircle className="w-4 h-4 text-green-600" />
                        Completed
                      </span>
                    ) : (
                      <span className="flex items-center gap-1">
                        <AlertCircle className="w-4 h-4 text-blue-600" />
                        Available
                      </span>
                    )}
                  </div>
                </div>
                {test.status === 'completed' && test.score && (
                  <div className="text-right">
                    <div className="text-lg font-bold text-dark-900">
                      {((test.score / test.totalMarks) * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-dark-600">
                      {test.score}/{test.totalMarks}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Important Notice */}
      <div className="card p-6 border-l-4 border-blue-600">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
            <AlertCircle className="w-5 h-5 text-blue-600" />
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-dark-900 mb-1">
              Proctoring Information
            </h4>
            <p className="text-sm text-dark-600 mb-3">
              All tests are monitored using AI-powered proctoring. Please ensure:
            </p>
            <ul className="text-sm text-dark-600 space-y-1">
              <li>• Your camera and microphone are working</li>
              <li>• You are in a well-lit, quiet environment</li>
              <li>• No other person is visible in the camera frame</li>
              <li>• You don't switch tabs or windows during the test</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentDashboard;