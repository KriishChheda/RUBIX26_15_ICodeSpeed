import React, { useState, useEffect } from 'react';
import { Search, Download, Eye, AlertTriangle, TrendingUp, Users, Award } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { mockResults } from '../services/mockData';

const AdminResults = () => {
  const [results, setResults] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedResult, setSelectedResult] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setResults(mockResults);
      setLoading(false);
    }, 500);
  }, []);

  const filteredResults = results.filter(result =>
    result.studentName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    result.studentEmail.toLowerCase().includes(searchTerm.toLowerCase()) ||
    result.testTitle.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const calculateStats = () => {
    if (results.length === 0) return { average: 0, passRate: 0, totalViolations: 0 };
    
    const average = results.reduce((sum, r) => sum + r.percentage, 0) / results.length;
    const passed = results.filter(r => r.percentage >= 60).length;
    const passRate = (passed / results.length) * 100;
    const totalViolations = results.reduce((sum, r) => sum + r.violations, 0);

    return {
      average: average.toFixed(1),
      passRate: passRate.toFixed(1),
      totalViolations
    };
  };

  const stats = calculateStats();

  // Score distribution data for chart
  const scoreDistribution = [
    { range: '0-20', count: results.filter(r => r.percentage < 20).length },
    { range: '20-40', count: results.filter(r => r.percentage >= 20 && r.percentage < 40).length },
    { range: '40-60', count: results.filter(r => r.percentage >= 40 && r.percentage < 60).length },
    { range: '60-80', count: results.filter(r => r.percentage >= 60 && r.percentage < 80).length },
    { range: '80-100', count: results.filter(r => r.percentage >= 80).length },
  ];

  const handleExport = () => {
    // TODO: Implement CSV export
    const csv = [
      ['Student Name', 'Email', 'Test', 'Score', 'Percentage', 'Violations', 'Risk Score', 'Date'].join(','),
      ...filteredResults.map(r => [
        r.studentName,
        r.studentEmail,
        r.testTitle,
        `${r.score}/${r.totalMarks}`,
        `${r.percentage.toFixed(1)}%`,
        r.violations,
        r.riskScore,
        new Date(r.submittedAt).toLocaleString()
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `results-${new Date().toISOString()}.csv`;
    a.click();
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
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-display font-bold text-dark-900">
            Student Results
          </h2>
          <p className="text-dark-600 mt-1">
            View and analyze test submissions
          </p>
        </div>
        <button
          onClick={handleExport}
          className="btn-primary flex items-center gap-2"
        >
          <Download className="w-4 h-4" />
          Export Results
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
              <Users className="w-6 h-6 text-blue-600" />
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-dark-900 mb-1">
            {results.length}
          </div>
          <div className="text-sm text-dark-600">Total Submissions</div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-dark-900 mb-1">
            {stats.average}%
          </div>
          <div className="text-sm text-dark-600">Average Score</div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
              <Award className="w-6 h-6 text-purple-600" />
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-dark-900 mb-1">
            {stats.passRate}%
          </div>
          <div className="text-sm text-dark-600">Pass Rate</div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
              <AlertTriangle className="w-6 h-6 text-red-600" />
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-dark-900 mb-1">
            {stats.totalViolations}
          </div>
          <div className="text-sm text-dark-600">Total Violations</div>
        </div>
      </div>

      {/* Score Distribution Chart */}
      <div className="card p-6">
        <h3 className="text-xl font-display font-bold text-dark-900 mb-6">
          Score Distribution
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={scoreDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="range" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '2px solid #e2e8f0',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="count" fill="#0f172a" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Search */}
      <div className="card p-6">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
          <input
            type="text"
            placeholder="Search by student name, email, or test..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input-field pl-12 w-full"
          />
        </div>
      </div>

      {/* Results Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-dark-50 border-b-2 border-dark-100">
              <tr>
                <th className="px-6 py-4 text-left text-sm font-semibold text-dark-900">Student</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-dark-900">Test</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-dark-900">Score</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-dark-900">Violations</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-dark-900">Risk Score</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-dark-900">Date</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-dark-900">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-100">
              {filteredResults.length === 0 ? (
                <tr>
                  <td colSpan="7" className="px-6 py-12 text-center text-dark-600">
                    No results found
                  </td>
                </tr>
              ) : (
                filteredResults.map((result) => (
                  <tr key={result.id} className="hover:bg-dark-50 transition-colors">
                    <td className="px-6 py-4">
                      <div>
                        <div className="font-medium text-dark-900">{result.studentName}</div>
                        <div className="text-sm text-dark-600">{result.studentEmail}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-dark-900">{result.testTitle}</td>
                    <td className="px-6 py-4">
                      <div>
                        <div className="font-semibold text-dark-900">
                          {result.score}/{result.totalMarks}
                        </div>
                        <div className="text-sm text-dark-600">{result.percentage.toFixed(1)}%</div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`badge ${result.violations > 2 ? 'badge-danger' : result.violations > 0 ? 'badge-warning' : 'badge-success'}`}>
                        {result.violations}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-dark-200 rounded-full h-2 w-16">
                          <div
                            className={`h-2 rounded-full ${
                              result.riskScore > 50 ? 'bg-red-600' : result.riskScore > 25 ? 'bg-yellow-600' : 'bg-green-600'
                            }`}
                            style={{ width: `${result.riskScore}%` }}
                          />
                        </div>
                        <span className="text-sm text-dark-600">{result.riskScore}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-dark-600">
                      {new Date(result.submittedAt).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4">
                      <button
                        onClick={() => setSelectedResult(result)}
                        className="text-dark-600 hover:text-dark-900"
                      >
                        <Eye className="w-5 h-5" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Result Modal */}
      {selectedResult && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b-2 border-dark-100">
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-display font-bold text-dark-900">
                  Detailed Result
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
              {/* Student & Test Info */}
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-medium text-dark-600 mb-1">Student</h4>
                  <p className="font-semibold text-dark-900">{selectedResult.studentName}</p>
                  <p className="text-sm text-dark-600">{selectedResult.studentEmail}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-dark-600 mb-1">Test</h4>
                  <p className="font-semibold text-dark-900">{selectedResult.testTitle}</p>
                </div>
              </div>

              {/* Score Details */}
              <div className="grid grid-cols-3 gap-4 p-4 bg-dark-50 rounded-lg">
                <div>
                  <div className="text-sm text-dark-600 mb-1">Score</div>
                  <div className="text-2xl font-bold text-dark-900">
                    {selectedResult.score}/{selectedResult.totalMarks}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-dark-600 mb-1">Percentage</div>
                  <div className="text-2xl font-bold text-dark-900">
                    {selectedResult.percentage.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-sm text-dark-600 mb-1">Grade</div>
                  <div className="text-2xl font-bold text-dark-900">
                    {selectedResult.percentage >= 90 ? 'A+' : 
                     selectedResult.percentage >= 80 ? 'A' : 
                     selectedResult.percentage >= 70 ? 'B' : 
                     selectedResult.percentage >= 60 ? 'C' : 'D'}
                  </div>
                </div>
              </div>

              {/* Proctoring Data */}
              <div>
                <h4 className="font-semibold text-dark-900 mb-3">Proctoring Analysis</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 border-2 border-dark-100 rounded-lg">
                    <div className="text-sm text-dark-600 mb-1">Violations</div>
                    <div className="text-xl font-bold text-dark-900">{selectedResult.violations}</div>
                  </div>
                  <div className="p-4 border-2 border-dark-100 rounded-lg">
                    <div className="text-sm text-dark-600 mb-1">Risk Score</div>
                    <div className="flex items-center gap-2">
                      <div className="text-xl font-bold text-dark-900">{selectedResult.riskScore}%</div>
                      <span className={`badge ${
                        selectedResult.riskScore > 50 ? 'badge-danger' : 
                        selectedResult.riskScore > 25 ? 'badge-warning' : 
                        'badge-success'
                      }`}>
                        {selectedResult.riskScore > 50 ? 'High' : 
                         selectedResult.riskScore > 25 ? 'Medium' : 'Low'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Submission Details */}
              <div className="p-4 bg-dark-50 rounded-lg">
                <h4 className="text-sm font-medium text-dark-600 mb-2">Submission Details</h4>
                <p className="text-sm text-dark-900">
                  Submitted on {new Date(selectedResult.submittedAt).toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminResults;