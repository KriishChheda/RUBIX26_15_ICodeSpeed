// import React, { useState, useEffect } from 'react';
// import { Link } from 'react-router-dom';
// import {
//   Plus,
//   Search,
//   Filter,
//   Edit,
//   Trash2,
//   Eye,
//   Clock,
//   FileText,
//   Send
// } from 'lucide-react';
// import { mockTests } from '../services/mockData';

// const AdminTests = () => {
//   const [tests, setTests] = useState([]);
//   const [searchTerm, setSearchTerm] = useState('');
//   const [filterStatus, setFilterStatus] = useState('all');
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     // Simulate API call
//     setTimeout(() => {
//       setTests(mockTests);
//       setLoading(false);
//     }, 500);
//   }, []);

//   const filteredTests = tests.filter(test => {
//     const matchesSearch = test.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
//                          test.description.toLowerCase().includes(searchTerm.toLowerCase());
//     const matchesFilter = filterStatus === 'all' || test.status === filterStatus;
//     return matchesSearch && matchesFilter;
//   });

//   const handleDelete = (testId) => {
//     if (window.confirm('Are you sure you want to delete this test?')) {
//       setTests(tests.filter(t => t.id !== testId));
//     }
//   };

//   const handlePublish = (testId) => {
//     setTests(tests.map(t => 
//       t.id === testId ? { ...t, status: 'published' } : t
//     ));
//   };

//   if (loading) {
//     return (
//       <div className="flex items-center justify-center h-64">
//         <div className="animate-spin rounded-full h-12 w-12 border-4 border-dark-900 border-t-transparent"></div>
//       </div>
//     );
//   }

//   return (
//     <div className="space-y-6 animate-fade-in">
//       {/* Header */}
//       <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
//         <div>
//           <h2 className="text-2xl font-display font-bold text-dark-900">
//             Manage Tests
//           </h2>
//           <p className="text-dark-600 mt-1">
//             Create, edit, and manage your examinations
//           </p>
//         </div>
//         <Link
//           to="/admin/tests/create"
//           className="btn-primary flex items-center gap-2"
//         >
//           <Plus className="w-5 h-5" />
//           Create New Test
//         </Link>
//       </div>

//       {/* Filters and Search */}
//       <div className="card p-6">
//         <div className="flex flex-col sm:flex-row gap-4">
//           <div className="flex-1 relative">
//             <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-400" />
//             <input
//               type="text"
//               placeholder="Search tests..."
//               value={searchTerm}
//               onChange={(e) => setSearchTerm(e.target.value)}
//               className="input-field pl-12 w-full"
//             />
//           </div>
//           <div className="sm:w-48">
//             <select
//               value={filterStatus}
//               onChange={(e) => setFilterStatus(e.target.value)}
//               className="input-field w-full"
//             >
//               <option value="all">All Status</option>
//               <option value="published">Published</option>
//               <option value="draft">Draft</option>
//             </select>
//           </div>
//         </div>
//       </div>

//       {/* Tests List */}
//       <div className="space-y-4">
//         {filteredTests.length === 0 ? (
//           <div className="card p-12 text-center">
//             <FileText className="w-16 h-16 text-dark-300 mx-auto mb-4" />
//             <h3 className="text-xl font-display font-bold text-dark-900 mb-2">
//               No tests found
//             </h3>
//             <p className="text-dark-600 mb-6">
//               {searchTerm || filterStatus !== 'all'
//                 ? 'Try adjusting your filters'
//                 : 'Create your first test to get started'}
//             </p>
//             {!searchTerm && filterStatus === 'all' && (
//               <Link to="/admin/tests/create" className="btn-primary inline-flex items-center gap-2">
//                 <Plus className="w-5 h-5" />
//                 Create Test
//               </Link>
//             )}
//           </div>
//         ) : (
//           filteredTests.map((test) => (
//             <div key={test.id} className="card p-6 hover:shadow-lg transition-all">
//               <div className="flex flex-col lg:flex-row lg:items-center gap-4">
//                 <div className="flex-1">
//                   <div className="flex items-center gap-3 mb-2">
//                     <h3 className="text-xl font-display font-bold text-dark-900">
//                       {test.title}
//                     </h3>
//                     <span
//                       className={`
//                         badge
//                         ${test.status === 'published' ? 'badge-success' : 'badge-warning'}
//                       `}
//                     >
//                       {test.status}
//                     </span>
//                   </div>
//                   <p className="text-dark-600 mb-4">{test.description}</p>
//                   <div className="flex flex-wrap gap-4 text-sm text-dark-600">
//                     <span className="flex items-center gap-1">
//                       <Clock className="w-4 h-4" />
//                       {test.duration} minutes
//                     </span>
//                     <span className="flex items-center gap-1">
//                       <FileText className="w-4 h-4" />
//                       {test.totalQuestions} questions
//                     </span>
//                     <span>
//                       Total Marks: {test.totalMarks}
//                     </span>
//                     {test.startDate && (
//                       <span>
//                         Start: {new Date(test.startDate).toLocaleDateString()}
//                       </span>
//                     )}
//                   </div>
//                 </div>

//                 <div className="flex lg:flex-col gap-2">
//                   <Link
//                     to={`/admin/tests/${test.id}`}
//                     className="btn-secondary flex items-center gap-2 justify-center"
//                   >
//                     <Eye className="w-4 h-4" />
//                     View
//                   </Link>
//                   <Link
//                     to={`/admin/tests/${test.id}/edit`}
//                     className="btn-secondary flex items-center gap-2 justify-center"
//                   >
//                     <Edit className="w-4 h-4" />
//                     Edit
//                   </Link>
//                   {test.status === 'draft' && (
//                     <button
//                       onClick={() => handlePublish(test.id)}
//                       className="px-4 py-2 bg-green-600 text-white rounded-lg font-medium 
//                                hover:bg-green-700 transition-all flex items-center gap-2 justify-center"
//                     >
//                       <Send className="w-4 h-4" />
//                       Publish
//                     </button>
//                   )}
//                   <button
//                     onClick={() => handleDelete(test.id)}
//                     className="px-4 py-2 bg-red-50 text-red-600 rounded-lg font-medium 
//                              hover:bg-red-100 transition-all flex items-center gap-2 justify-center"
//                   >
//                     <Trash2 className="w-4 h-4" />
//                     Delete
//                   </button>
//                 </div>
//               </div>
//             </div>
//           ))
//         )}
//       </div>
//     </div>
//   );
// };

// export default AdminTests;

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { testsAPI } from '../services/api';
import {
  Plus,
  Search,
  Filter,
  Edit,
  Trash2,
  Eye,
  Clock,
  FileText,
  Send
} from 'lucide-react';

const AdminTests = () => {
  const [tests, setTests] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTests();
  }, []);

  const fetchTests = async () => {
    try {
      setLoading(true);
      const response = await testsAPI.getAllTests();
      setTests(response.data);
    } catch (err) {
      console.error('Failed to fetch tests:', err);
      setError('Failed to load tests');
    } finally {
      setLoading(false);
    }
  };

  const filteredTests = tests.filter(test => {
    const matchesSearch = test.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         test.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === 'all' || test.visibility === filterStatus;
    return matchesSearch && matchesFilter;
  });

  const handleDelete = async (testId) => {
    if (window.confirm('Are you sure you want to delete this test?')) {
      try {
        await testsAPI.deleteTest(testId);
        setTests(tests.filter(t => t.id !== testId));
      } catch (err) {
        console.error('Failed to delete test:', err);
        alert('Failed to delete test: ' + (err.response?.data?.error || 'Please try again'));
      }
    }
  };

  const handlePublish = async (testId) => {
    try {
      await testsAPI.publishTest(testId);
      // Update local state
      setTests(tests.map(t => 
        t.id === testId ? { ...t, visibility: 'published' } : t
      ));
    } catch (err) {
      console.error('Failed to publish test:', err);
      alert('Failed to publish test: ' + (err.response?.data?.error || 'Please try again'));
    }
  };

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
        <div className="text-center">
          <p className="text-red-600 mb-4">{error}</p>
          <button onClick={fetchTests} className="btn-primary">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-display font-bold text-dark-900">
            Manage Tests
          </h2>
          <p className="text-dark-600 mt-1">
            Create, edit, and manage your examinations
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

      {/* Filters and Search */}
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
              <option value="all">All Status</option>
              <option value="published">Published</option>
              <option value="draft">Draft</option>
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
            <p className="text-dark-600 mb-6">
              {searchTerm || filterStatus !== 'all'
                ? 'Try adjusting your filters'
                : 'Create your first test to get started'}
            </p>
            {!searchTerm && filterStatus === 'all' && (
              <Link to="/admin/tests/create" className="btn-primary inline-flex items-center gap-2">
                <Plus className="w-5 h-5" />
                Create Test
              </Link>
            )}
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
                    <span
                      className={`
                        badge
                        ${test.visibility === 'published' ? 'badge-success' : 'badge-warning'}
                      `}
                    >
                      {test.visibility}
                    </span>
                  </div>
                  <p className="text-dark-600 mb-4">{test.description}</p>
                  <div className="flex flex-wrap gap-4 text-sm text-dark-600">
                    <span className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {test.duration} minutes
                    </span>
                    <span className="flex items-center gap-1">
                      <FileText className="w-4 h-4" />
                      {test.total_questions} questions
                    </span>
                    <span>
                      Total Marks: {test.total_marks}
                    </span>
                    {test.start_date && (
                      <span>
                        Start: {new Date(test.start_date).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex lg:flex-col gap-2">
                  <Link
                    to={`/admin/tests/${test.id}`}
                    className="btn-secondary flex items-center gap-2 justify-center"
                  >
                    <Eye className="w-4 h-4" />
                    View
                  </Link>
                  <Link
                    to={`/admin/tests/${test.id}/edit`}
                    className="btn-secondary flex items-center gap-2 justify-center"
                  >
                    <Edit className="w-4 h-4" />
                    Edit
                  </Link>
                  {test.visibility === 'draft' && (
                    <button
                      onClick={() => handlePublish(test.id)}
                      className="px-4 py-2 bg-green-600 text-white rounded-lg font-medium 
                               hover:bg-green-700 transition-all flex items-center gap-2 justify-center"
                    >
                      <Send className="w-4 h-4" />
                      Publish
                    </button>
                  )}
                  <button
                    onClick={() => handleDelete(test.id)}
                    className="px-4 py-2 bg-red-50 text-red-600 rounded-lg font-medium 
                             hover:bg-red-100 transition-all flex items-center gap-2 justify-center"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AdminTests;