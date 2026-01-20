// import React, { useState, useEffect } from 'react';
// import { Link } from 'react-router-dom';
// import { resultsAPI, testsAPI } from '../services/api';
// import {
//   FileText,
//   Users,
//   TrendingUp,
//   AlertTriangle,
//   Plus,
//   Calendar,
//   Clock
// } from 'lucide-react';
// import { mockTests, mockResults } from '../services/mockData';

// const AdminDashboard = () => {
//   const [stats, setStats] = useState({
//     totalTests: 0,
//     activeTests: 0,
//     totalStudents: 0,
//     averageScore: 0,
//     recentTests: [],
//     recentResults: []
//   });

//   // useEffect(() => {
//   //   // Calculate statistics from mock data
//   //   const activeTests = mockTests.filter(t => t.status === 'published').length;
//   //   const uniqueStudents = new Set(mockResults.map(r => r.studentEmail)).size;
//   //   const avgScore = mockResults.length > 0
//   //     ? mockResults.reduce((sum, r) => sum + r.percentage, 0) / mockResults.length
//   //     : 0;

//   //   setStats({
//   //     totalTests: mockTests.length,
//   //     activeTests,
//   //     totalStudents: uniqueStudents,
//   //     averageScore: avgScore.toFixed(1),
//   //     recentTests: mockTests.slice(0, 3),
//   //     recentResults: mockResults.slice(0, 5)
//   //   });
//   // }, []);

//   useEffect(() => {
//   const fetchDashboardData = async () => {
//     try {
//       const [testsRes, resultsRes] = await Promise.all([
//         testsAPI.getAllTests(),
//         resultsAPI.getAllResults()
//       ]);
//       setTests(testsRes.data);
//       setResults(resultsRes.data);
//     } catch (err) {
//       console.error('Failed to fetch dashboard data');
//     }
//   };
//   fetchDashboardData();
// }, []);

//   const statCards = [
//     {
//       title: 'Total Tests',
//       value: stats.totalTests,
//       icon: FileText,
//       color: 'bg-blue-500',
//       change: '+2 this week'
//     },
//     {
//       title: 'Active Tests',
//       value: stats.activeTests,
//       icon: TrendingUp,
//       color: 'bg-green-500',
//       change: 'Live now'
//     },
//     {
//       title: 'Total Students',
//       value: stats.totalStudents,
//       icon: Users,
//       color: 'bg-purple-500',
//       change: '+5 this month'
//     },
//     {
//       title: 'Average Score',
//       value: `${stats.averageScore}%`,
//       icon: AlertTriangle,
//       color: 'bg-orange-500',
//       change: '+3.2% from last week'
//     }
//   ];

//   return (
//     <div className="space-y-6 animate-fade-in">
//       {/* Welcome Section */}
//       <div className="bg-white rounded-2xl border-2 border-dark-100 p-8">
//         <div className="flex items-center justify-between">
//           <div>
//             <h2 className="text-3xl font-display font-bold text-dark-900 mb-2">
//               Welcome back, Admin
//             </h2>
//             <p className="text-dark-600">
//               Here's what's happening with your examination system today.
//             </p>
//           </div>
//           <Link
//             to="/admin/tests/create"
//             className="btn-primary flex items-center gap-2"
//           >
//             <Plus className="w-5 h-5" />
//             Create New Test
//           </Link>
//         </div>
//       </div>

//       {/* Stats Grid */}
//       <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
//         {statCards.map((stat, index) => {
//           const Icon = stat.icon;
//           return (
//             <div
//               key={stat.title}
//               className="card p-6 hover:shadow-lg animate-slide-up"
//               style={{ animationDelay: `${index * 100}ms` }}
//             >
//               <div className="flex items-center justify-between mb-4">
//                 <div className={`${stat.color} w-12 h-12 rounded-xl flex items-center justify-center`}>
//                   <Icon className="w-6 h-6 text-white" />
//                 </div>
//               </div>
//               <div className="text-3xl font-display font-bold text-dark-900 mb-1">
//                 {stat.value}
//               </div>
//               <div className="text-sm text-dark-600 mb-2">{stat.title}</div>
//               <div className="text-xs text-green-600 font-medium">{stat.change}</div>
//             </div>
//           );
//         })}
//       </div>

//       <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
//         {/* Recent Tests */}
//         <div className="card p-6">
//           <div className="flex items-center justify-between mb-6">
//             <h3 className="text-xl font-display font-bold text-dark-900">
//               Recent Tests
//             </h3>
//             <Link
//               to="/admin/tests"
//               className="text-sm text-dark-900 font-medium hover:underline"
//             >
//               View all
//             </Link>
//           </div>

//           <div className="space-y-4">
//             {stats.recentTests.map((test) => (
//               <div
//                 key={test.id}
//                 className="flex items-center justify-between p-4 bg-dark-50 rounded-lg hover:bg-dark-100 transition-colors"
//               >
//                 <div className="flex-1">
//                   <h4 className="font-semibold text-dark-900 mb-1">
//                     {test.title}
//                   </h4>
//                   <div className="flex items-center gap-4 text-sm text-dark-600">
//                     <span className="flex items-center gap-1">
//                       <Clock className="w-4 h-4" />
//                       {test.duration} min
//                     </span>
//                     <span>{test.totalQuestions} questions</span>
//                   </div>
//                 </div>
//                 <span
//                   className={`
//                     badge
//                     ${test.status === 'published' ? 'badge-success' : 'badge-warning'}
//                   `}
//                 >
//                   {test.status}
//                 </span>
//               </div>
//             ))}
//           </div>
//         </div>

//         {/* Recent Results */}
//         <div className="card p-6">
//           <div className="flex items-center justify-between mb-6">
//             <h3 className="text-xl font-display font-bold text-dark-900">
//               Recent Submissions
//             </h3>
//             <Link
//               to="/admin/results"
//               className="text-sm text-dark-900 font-medium hover:underline"
//             >
//               View all
//             </Link>
//           </div>

//           <div className="space-y-4">
//             {stats.recentResults.map((result) => (
//               <div
//                 key={result.id}
//                 className="flex items-center justify-between p-4 bg-dark-50 rounded-lg"
//               >
//                 <div className="flex-1">
//                   <h4 className="font-semibold text-dark-900 mb-1">
//                     {result.studentName}
//                   </h4>
//                   <p className="text-sm text-dark-600">{result.testTitle}</p>
//                 </div>
//                 <div className="text-right">
//                   <div className="text-lg font-bold text-dark-900">
//                     {result.percentage.toFixed(1)}%
//                   </div>
//                   <div className="text-xs text-dark-600">
//                     {result.violations} violations
//                   </div>
//                 </div>
//               </div>
//             ))}
//           </div>
//         </div>
//       </div>

//       {/* Quick Actions */}
//       <div className="card p-6">
//         <h3 className="text-xl font-display font-bold text-dark-900 mb-6">
//           Quick Actions
//         </h3>
//         <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
//           <Link
//             to="/admin/tests/create"
//             className="p-6 bg-dark-50 rounded-xl hover:bg-dark-100 transition-colors group"
//           >
//             <FileText className="w-8 h-8 text-dark-900 mb-3 group-hover:scale-110 transition-transform" />
//             <h4 className="font-semibold text-dark-900 mb-1">Create Test</h4>
//             <p className="text-sm text-dark-600">Set up a new examination</p>
//           </Link>
//           <Link
//             to="/admin/results"
//             className="p-6 bg-dark-50 rounded-xl hover:bg-dark-100 transition-colors group"
//           >
//             <TrendingUp className="w-8 h-8 text-dark-900 mb-3 group-hover:scale-110 transition-transform" />
//             <h4 className="font-semibold text-dark-900 mb-1">View Analytics</h4>
//             <p className="text-sm text-dark-600">Check performance metrics</p>
//           </Link>
//           <Link
//             to="/admin/tests"
//             className="p-6 bg-dark-50 rounded-xl hover:bg-dark-100 transition-colors group"
//           >
//             <Calendar className="w-8 h-8 text-dark-900 mb-3 group-hover:scale-110 transition-transform" />
//             <h4 className="font-semibold text-dark-900 mb-1">Manage Tests</h4>
//             <p className="text-sm text-dark-600">Edit or publish tests</p>
//           </Link>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default AdminDashboard;


import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { resultsAPI, testsAPI } from '../services/api';
import {
  FileText,
  Users,
  TrendingUp,
  AlertTriangle,
  Plus,
  Calendar,
  Clock
} from 'lucide-react';

const AdminDashboard = () => {
  const [stats, setStats] = useState({
    totalTests: 0,
    activeTests: 0,
    totalStudents: 0,
    averageScore: 0,
    recentTests: [],
    recentResults: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch tests and results from backend
        const [testsRes, resultsRes] = await Promise.all([
          testsAPI.getAllTests(),
          resultsAPI.getAllResults()
        ]);

        const tests = testsRes.data;
        const results = resultsRes.data;

        // Calculate statistics
        const activeTests = tests.filter(t => t.visibility === 'published').length;
        const uniqueStudents = new Set(results.map(r => r.student_id)).size;
        const avgScore = results.length > 0
          ? results.reduce((sum, r) => sum + (r.score / r.total_marks * 100), 0) / results.length
          : 0;

        setStats({
          totalTests: tests.length,
          activeTests: activeTests,
          totalStudents: uniqueStudents,
          averageScore: avgScore.toFixed(1),
          recentTests: tests
            .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
            .slice(0, 3),
          recentResults: results
            .sort((a, b) => new Date(b.submitted_at) - new Date(a.submitted_at))
            .slice(0, 5)
        });
      } catch (err) {
        console.error('Failed to fetch dashboard data:', err);
        setError('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const statCards = [
    {
      title: 'Total Tests',
      value: stats.totalTests,
      icon: FileText,
      color: 'bg-blue-500',
      change: '+2 this week'
    },
    {
      title: 'Active Tests',
      value: stats.activeTests,
      icon: TrendingUp,
      color: 'bg-green-500',
      change: 'Live now'
    },
    {
      title: 'Total Students',
      value: stats.totalStudents,
      icon: Users,
      color: 'bg-purple-500',
      change: '+5 this month'
    },
    {
      title: 'Average Score',
      value: `${stats.averageScore}%`,
      icon: AlertTriangle,
      color: 'bg-orange-500',
      change: '+3.2% from last week'
    }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-dark-900 border-t-transparent"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-600">{error}</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Welcome Section */}
      <div className="bg-white rounded-2xl border-2 border-dark-100 p-8">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-display font-bold text-dark-900 mb-2">
              Welcome back, Admin
            </h2>
            <p className="text-dark-600">
              Here's what's happening with your examination system today.
            </p>
          </div>
          <Link
            to="/admin/tests/create"
            className="btn-primary flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            Create New Test
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.title}
              className="card p-6 hover:shadow-lg animate-slide-up"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`${stat.color} w-12 h-12 rounded-xl flex items-center justify-center`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="text-3xl font-display font-bold text-dark-900 mb-1">
                {stat.value}
              </div>
              <div className="text-sm text-dark-600 mb-2">{stat.title}</div>
              <div className="text-xs text-green-600 font-medium">{stat.change}</div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Tests */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-display font-bold text-dark-900">
              Recent Tests
            </h3>
            <Link
              to="/admin/tests"
              className="text-sm text-dark-900 font-medium hover:underline"
            >
              View all
            </Link>
          </div>

          <div className="space-y-4">
            {stats.recentTests.length > 0 ? (
              stats.recentTests.map((test) => (
                <div
                  key={test.id}
                  className="flex items-center justify-between p-4 bg-dark-50 rounded-lg hover:bg-dark-100 transition-colors"
                >
                  <div className="flex-1">
                    <h4 className="font-semibold text-dark-900 mb-1">
                      {test.title}
                    </h4>
                    <div className="flex items-center gap-4 text-sm text-dark-600">
                      <span className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        {test.duration} min
                      </span>
                      <span>{test.total_questions} questions</span>
                    </div>
                  </div>
                  <span
                    className={`
                      badge
                      ${test.visibility === 'published' ? 'badge-success' : 'badge-warning'}
                    `}
                  >
                    {test.visibility}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-dark-600">
                No tests created yet
              </div>
            )}
          </div>
        </div>

        {/* Recent Results */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-display font-bold text-dark-900">
              Recent Submissions
            </h3>
            <Link
              to="/admin/results"
              className="text-sm text-dark-900 font-medium hover:underline"
            >
              View all
            </Link>
          </div>

          <div className="space-y-4">
            {stats.recentResults.length > 0 ? (
              stats.recentResults.map((result) => (
                <div
                  key={result.id}
                  className="flex items-center justify-between p-4 bg-dark-50 rounded-lg"
                >
                  <div className="flex-1">
                    <h4 className="font-semibold text-dark-900 mb-1">
                      {result.student_name || result.student_id}
                    </h4>
                    <p className="text-sm text-dark-600">{result.test_title || 'Test'}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-dark-900">
                      {((result.score / result.total_marks) * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-dark-600">
                      {result.violations?.length || 0} violations
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-dark-600">
                No submissions yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card p-6">
        <h3 className="text-xl font-display font-bold text-dark-900 mb-6">
          Quick Actions
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            to="/admin/tests/create"
            className="p-6 bg-dark-50 rounded-xl hover:bg-dark-100 transition-colors group"
          >
            <FileText className="w-8 h-8 text-dark-900 mb-3 group-hover:scale-110 transition-transform" />
            <h4 className="font-semibold text-dark-900 mb-1">Create Test</h4>
            <p className="text-sm text-dark-600">Set up a new examination</p>
          </Link>
          <Link
            to="/admin/results"
            className="p-6 bg-dark-50 rounded-xl hover:bg-dark-100 transition-colors group"
          >
            <TrendingUp className="w-8 h-8 text-dark-900 mb-3 group-hover:scale-110 transition-transform" />
            <h4 className="font-semibold text-dark-900 mb-1">View Analytics</h4>
            <p className="text-sm text-dark-600">Check performance metrics</p>
          </Link>
          <Link
            to="/admin/tests"
            className="p-6 bg-dark-50 rounded-xl hover:bg-dark-100 transition-colors group"
          >
            <Calendar className="w-8 h-8 text-dark-900 mb-3 group-hover:scale-110 transition-transform" />
            <h4 className="font-semibold text-dark-900 mb-1">Manage Tests</h4>
            <p className="text-sm text-dark-600">Edit or publish tests</p>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;