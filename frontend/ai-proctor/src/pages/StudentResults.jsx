import React, { useState, useEffect } from 'react';
import { TrendingUp, Award, AlertTriangle, Calendar, Eye } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { mockStudentTests } from '../services/mockData';

const StudentResults = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedResult, setSelectedResult] = useState(null);

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      const completedTests = mockStudentTests.filter(t => t.status === 'completed');
      setResults(completedTests);
      setLoading(false);
    }, 500);
  }, []);

  const calculateStats = () => {
    if (results.length === 0) return { average: 0, highest: 0, lowest: 0 };
    
    const scores = results.map(r => (r.score / r.totalMarks) * 100);
    return {
      average: (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1),
      highest: Math.max(...scores).toFixed(1),
      lowest: Math.min(...scores).toFixed(1)
    };
  };

  const stats = calculateStats();

  const chartData = results.map((result, index) => ({
    name: `Test ${index + 1}`,
    score: ((result.score / result.totalMarks) * 100).toFixed(1)
  }));

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
          My Results
        </h2>
        <p className="text-dark-600 mt-1">
          View your test scores and performance analytics
        </p>
      </div>

      {results.length === 0 ? (
        <div className="card p-12 text-center">
          <Award className="w-16 h-16 text-dark-300 mx-auto mb-4" />
          <h3 className="text-xl font-display font-bold text-dark-900 mb-2">
            No Results Yet
          </h3>
          <p className="text-dark-600">
            Complete a test to see your results here
          </p>
        </div>
      ) : (
        <>
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-blue-600" />
                </div>
              </div>
              <div className="text-3xl font-display font-bold text-dark-900 mb-1">
                {stats.average}%
              </div>
              <div className="text-sm text-dark-600">Average Score</div>
            </div>

            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                  <Award className="w-6 h-6 text-green-600" />
                </div>
              </div>
              <div className="text-3xl font-display font-bold text-dark-900 mb-1">
                {stats.highest}%
              </div>
              <div className="text-sm text-dark-600">Highest Score</div>
            </div>

            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 bg-orange-100 rounded-xl flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-orange-600" />
                </div>
              </div>
              <div className="text-3xl font-display font-bold text-dark-900 mb-1">
                {stats.lowest}%
              </div>
              <div className="text-sm text-dark-600">Lowest Score</div>
            </div>
          </div>

          {/* Performance Chart */}
          <div className="card p-6">
            <h3 className="text-xl font-display font-bold text-dark-900 mb-6">
              Performance Trend
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" stroke="#64748b" />
                  <YAxis stroke="#64748b" domain={[0, 100]} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '2px solid #e2e8f0',
                      borderRadius: '8px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="score" 
                    stroke="#0f172a" 
                    strokeWidth={2}
                    dot={{ fill: '#0f172a', r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Results List */}
          <div className="card p-6">
            <h3 className="text-xl font-display font-bold text-dark-900 mb-6">
              Test History
            </h3>
            <div className="space-y-4">
              {results.map((result) => {
                const percentage = ((result.score / result.totalMarks) * 100).toFixed(1);
                const getGrade = (percentage) => {
                  if (percentage >= 90) return { grade: 'A+', color: 'text-green-600' };
                  if (percentage >= 80) return { grade: 'A', color: 'text-green-600' };
                  if (percentage >= 70) return { grade: 'B', color: 'text-blue-600' };
                  if (percentage >= 60) return { grade: 'C', color: 'text-yellow-600' };
                  return { grade: 'D', color: 'text-red-600' };
                };
                const { grade, color } = getGrade(percentage);

                return (
                  <div
                    key={result.id}
                    className="p-4 bg-dark-50 rounded-lg hover:bg-dark-100 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-dark-900">{result.title}</h4>
                      <div className="flex items-center gap-3">
                        <span className={`text-2xl font-bold ${color}`}>
                          {grade}
                        </span>
                        <button
                          onClick={() => setSelectedResult(result)}
                          className="text-dark-600 hover:text-dark-900"
                        >
                          <Eye className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-dark-600">Score:</span>
                        <span className="ml-2 font-semibold text-dark-900">
                          {result.score}/{result.totalMarks}
                        </span>
                      </div>
                      <div>
                        <span className="text-dark-600">Percentage:</span>
                        <span className="ml-2 font-semibold text-dark-900">
                          {percentage}%
                        </span>
                      </div>
                      <div>
                        <span className="text-dark-600">Duration:</span>
                        <span className="ml-2 font-semibold text-dark-900">
                          {result.duration} min
                        </span>
                      </div>
                      <div>
                        <span className="text-dark-600">Date:</span>
                        <span className="ml-2 font-semibold text-dark-900">
                          {new Date(result.completedAt).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}

      {/* Detailed Result Modal */}
      {selectedResult && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b-2 border-dark-100">
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-display font-bold text-dark-900">
                  Test Details
                </h3>
                <button
                  onClick={() => setSelectedResult(null)}
                  className="text-dark-600 hover:text-dark-900"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="p-6 space-y-6">
              <div>
                <h4 className="font-semibold text-dark-900 mb-2">{selectedResult.title}</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-dark-600">Score:</span>
                    <span className="ml-2 font-semibold">
                      {selectedResult.score}/{selectedResult.totalMarks}
                    </span>
                  </div>
                  <div>
                    <span className="text-dark-600">Percentage:</span>
                    <span className="ml-2 font-semibold">
                      {((selectedResult.score / selectedResult.totalMarks) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-dark-600">Duration:</span>
                    <span className="ml-2 font-semibold">{selectedResult.duration} min</span>
                  </div>
                  <div>
                    <span className="text-dark-600">Completed:</span>
                    <span className="ml-2 font-semibold">
                      {new Date(selectedResult.completedAt).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-dark-50 rounded-lg">
                <p className="text-sm text-dark-600">
                  Detailed question-wise analysis will be available once the admin reviews your test.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StudentResults;