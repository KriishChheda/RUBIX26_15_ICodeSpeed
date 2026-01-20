// import React, { useState } from 'react';
// import { useNavigate } from 'react-router-dom';
// import { Plus, Trash2, Save, X } from 'lucide-react';

// const CreateTest = () => {
//   const navigate = useNavigate();
//   const [testData, setTestData] = useState({
//     title: '',
//     description: '',
//     duration: 60,
//     totalMarks: 100,
//     startDate: '',
//     endDate: '',
//   });

//   const [questions, setQuestions] = useState([
//     {
//       id: 1,
//       question: '',
//       type: 'multiple-choice',
//       options: ['', '', '', ''],
//       correctAnswer: 0,
//       marks: 1,
//     }
//   ]);

//   const [saving, setSaving] = useState(false);

//   const handleTestDataChange = (e) => {
//     const { name, value } = e.target;
//     setTestData(prev => ({
//       ...prev,
//       [name]: value
//     }));
//   };

//   const handleQuestionChange = (questionId, field, value) => {
//     setQuestions(questions.map(q =>
//       q.id === questionId ? { ...q, [field]: value } : q
//     ));
//   };

//   const handleOptionChange = (questionId, optionIndex, value) => {
//     setQuestions(questions.map(q =>
//       q.id === questionId
//         ? { ...q, options: q.options.map((opt, idx) => idx === optionIndex ? value : opt) }
//         : q
//     ));
//   };

//   const addQuestion = () => {
//     const newQuestion = {
//       id: questions.length + 1,
//       question: '',
//       type: 'multiple-choice',
//       options: ['', '', '', ''],
//       correctAnswer: 0,
//       marks: 1,
//     };
//     setQuestions([...questions, newQuestion]);
//   };

//   const removeQuestion = (questionId) => {
//     if (questions.length > 1) {
//       setQuestions(questions.filter(q => q.id !== questionId));
//     }
//   };

//   const handleSubmit = async (e, isDraft = false) => {
//     e.preventDefault();
//     setSaving(true);

//     // TODO: Replace with actual API call
//     // const response = await testsAPI.createTest({
//     //   ...testData,
//     //   questions,
//     //   status: isDraft ? 'draft' : 'published'
//     // });

//     setTimeout(() => {
//       setSaving(false);
//       navigate('/admin/tests');
//     }, 1000);
//   };

//   return (
//     <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
//       {/* Header */}
//       <div className="flex items-center justify-between">
//         <div>
//           <h2 className="text-2xl font-display font-bold text-dark-900">
//             Create New Test
//           </h2>
//           <p className="text-dark-600 mt-1">
//             Set up your examination details and questions
//           </p>
//         </div>
//         <button
//           onClick={() => navigate('/admin/tests')}
//           className="btn-secondary flex items-center gap-2"
//         >
//           <X className="w-4 h-4" />
//           Cancel
//         </button>
//       </div>

//       {/* Test Details */}
//       <div className="card p-6">
//         <h3 className="text-xl font-display font-bold text-dark-900 mb-6">
//           Test Details
//         </h3>
//         <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//           <div className="md:col-span-2">
//             <label className="block text-sm font-medium text-dark-700 mb-2">
//               Test Title *
//             </label>
//             <input
//               type="text"
//               name="title"
//               value={testData.title}
//               onChange={handleTestDataChange}
//               className="input-field w-full"
//               placeholder="e.g., Introduction to Machine Learning"
//               required
//             />
//           </div>

//           <div className="md:col-span-2">
//             <label className="block text-sm font-medium text-dark-700 mb-2">
//               Description *
//             </label>
//             <textarea
//               name="description"
//               value={testData.description}
//               onChange={handleTestDataChange}
//               className="input-field w-full h-24 resize-none"
//               placeholder="Brief description of the test..."
//               required
//             />
//           </div>

//           <div>
//             <label className="block text-sm font-medium text-dark-700 mb-2">
//               Duration (minutes) *
//             </label>
//             <input
//               type="number"
//               name="duration"
//               value={testData.duration}
//               onChange={handleTestDataChange}
//               className="input-field w-full"
//               min="1"
//               required
//             />
//           </div>

//           <div>
//             <label className="block text-sm font-medium text-dark-700 mb-2">
//               Total Marks *
//             </label>
//             <input
//               type="number"
//               name="totalMarks"
//               value={testData.totalMarks}
//               onChange={handleTestDataChange}
//               className="input-field w-full"
//               min="1"
//               required
//             />
//           </div>

//           <div>
//             <label className="block text-sm font-medium text-dark-700 mb-2">
//               Start Date & Time
//             </label>
//             <input
//               type="datetime-local"
//               name="startDate"
//               value={testData.startDate}
//               onChange={handleTestDataChange}
//               className="input-field w-full"
//             />
//           </div>

//           <div>
//             <label className="block text-sm font-medium text-dark-700 mb-2">
//               End Date & Time
//             </label>
//             <input
//               type="datetime-local"
//               name="endDate"
//               value={testData.endDate}
//               onChange={handleTestDataChange}
//               className="input-field w-full"
//             />
//           </div>
//         </div>
//       </div>

//       {/* Questions */}
//       <div className="card p-6">
//         <div className="flex items-center justify-between mb-6">
//           <h3 className="text-xl font-display font-bold text-dark-900">
//             Questions ({questions.length})
//           </h3>
//           <button
//             onClick={addQuestion}
//             className="btn-primary flex items-center gap-2"
//           >
//             <Plus className="w-4 h-4" />
//             Add Question
//           </button>
//         </div>

//         <div className="space-y-6">
//           {questions.map((q, index) => (
//             <div key={q.id} className="p-6 bg-dark-50 rounded-xl">
//               <div className="flex items-center justify-between mb-4">
//                 <h4 className="font-semibold text-dark-900">
//                   Question {index + 1}
//                 </h4>
//                 {questions.length > 1 && (
//                   <button
//                     onClick={() => removeQuestion(q.id)}
//                     className="text-red-600 hover:text-red-700"
//                   >
//                     <Trash2 className="w-5 h-5" />
//                   </button>
//                 )}
//               </div>

//               <div className="space-y-4">
//                 <div>
//                   <label className="block text-sm font-medium text-dark-700 mb-2">
//                     Question Text *
//                   </label>
//                   <textarea
//                     value={q.question}
//                     onChange={(e) => handleQuestionChange(q.id, 'question', e.target.value)}
//                     className="input-field w-full h-20 resize-none"
//                     placeholder="Enter your question..."
//                     required
//                   />
//                 </div>

//                 <div className="grid grid-cols-2 gap-4">
//                   <div>
//                     <label className="block text-sm font-medium text-dark-700 mb-2">
//                       Question Type
//                     </label>
//                     <select
//                       value={q.type}
//                       onChange={(e) => handleQuestionChange(q.id, 'type', e.target.value)}
//                       className="input-field w-full"
//                     >
//                       <option value="multiple-choice">Multiple Choice</option>
//                       <option value="text">Text Answer</option>
//                       <option value="true-false">True/False</option>
//                     </select>
//                   </div>

//                   <div>
//                     <label className="block text-sm font-medium text-dark-700 mb-2">
//                       Marks *
//                     </label>
//                     <input
//                       type="number"
//                       value={q.marks}
//                       onChange={(e) => handleQuestionChange(q.id, 'marks', parseInt(e.target.value))}
//                       className="input-field w-full"
//                       min="1"
//                       required
//                     />
//                   </div>
//                 </div>

//                 {q.type === 'multiple-choice' && (
//                   <div>
//                     <label className="block text-sm font-medium text-dark-700 mb-2">
//                       Options
//                     </label>
//                     <div className="space-y-2">
//                       {q.options.map((option, optIndex) => (
//                         <div key={optIndex} className="flex items-center gap-2">
//                           <input
//                             type="radio"
//                             name={`correct-${q.id}`}
//                             checked={q.correctAnswer === optIndex}
//                             onChange={() => handleQuestionChange(q.id, 'correctAnswer', optIndex)}
//                             className="w-4 h-4"
//                           />
//                           <input
//                             type="text"
//                             value={option}
//                             onChange={(e) => handleOptionChange(q.id, optIndex, e.target.value)}
//                             className="input-field flex-1"
//                             placeholder={`Option ${optIndex + 1}`}
//                           />
//                         </div>
//                       ))}
//                     </div>
//                     <p className="text-xs text-dark-500 mt-2">
//                       Select the radio button for the correct answer
//                     </p>
//                   </div>
//                 )}
//               </div>
//             </div>
//           ))}
//         </div>
//       </div>

//       {/* Actions */}
//       <div className="flex gap-4 justify-end sticky bottom-6">
//         <button
//           onClick={(e) => handleSubmit(e, true)}
//           disabled={saving}
//           className="btn-secondary"
//         >
//           Save as Draft
//         </button>
//         <button
//           onClick={(e) => handleSubmit(e, false)}
//           disabled={saving}
//           className="btn-primary flex items-center gap-2"
//         >
//           <Save className="w-4 h-4" />
//           {saving ? 'Publishing...' : 'Publish Test'}
//         </button>
//       </div>
//     </div>
//   );
// };

// export default CreateTest;


import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { testsAPI } from '../services/api';
import { Plus, Trash2, Save, X } from 'lucide-react';

const CreateTest = () => {
  const navigate = useNavigate();
  const [testData, setTestData] = useState({
    title: '',
    description: '',
    duration: 60,
    totalMarks: 100,
    startDate: '',
    endDate: '',
  });

  const [questions, setQuestions] = useState([
    {
      id: 'q1',
      question: '',
      type: 'multiple-choice',
      options: ['', '', '', ''],
      correctAnswer: '0',
      marks: 1,
    }
  ]);

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const handleTestDataChange = (e) => {
    const { name, value } = e.target;
    setTestData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleQuestionChange = (questionId, field, value) => {
    setQuestions(questions.map(q =>
      q.id === questionId ? { ...q, [field]: value } : q
    ));
  };

  const handleOptionChange = (questionId, optionIndex, value) => {
    setQuestions(questions.map(q =>
      q.id === questionId
        ? { ...q, options: q.options.map((opt, idx) => idx === optionIndex ? value : opt) }
        : q
    ));
  };

  const addQuestion = () => {
    const newQuestion = {
      id: `q${questions.length + 1}`,
      question: '',
      type: 'multiple-choice',
      options: ['', '', '', ''],
      correctAnswer: '0',
      marks: 1,
    };
    setQuestions([...questions, newQuestion]);
  };

  const removeQuestion = (questionId) => {
    if (questions.length > 1) {
      setQuestions(questions.filter(q => q.id !== questionId));
    }
  };

  const validateForm = () => {
    if (!testData.title.trim()) {
      setError('Test title is required');
      return false;
    }
    if (!testData.description.trim()) {
      setError('Test description is required');
      return false;
    }
    if (testData.duration < 1) {
      setError('Duration must be at least 1 minute');
      return false;
    }
    if (testData.totalMarks < 1) {
      setError('Total marks must be at least 1');
      return false;
    }

    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];
      if (!q.question.trim()) {
        setError(`Question ${i + 1}: Question text is required`);
        return false;
      }
      if (q.type === 'multiple-choice') {
        const hasEmptyOption = q.options.some(opt => !opt.trim());
        if (hasEmptyOption) {
          setError(`Question ${i + 1}: All options must be filled`);
          return false;
        }
      }
    }

    setError(null);
    return true;
  };

  const handleSubmit = async (e, shouldPublish = false) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setSaving(true);
    setError(null);

    try {
      // Format questions for backend
      const formattedQuestions = questions.map(q => ({
        id: q.id,
        question: q.question,
        type: q.type,
        options: q.type === 'multiple-choice' ? q.options : [],
        correct_answer: String(q.correctAnswer),
        marks: q.marks
      }));

      // Format dates for backend (convert to ISO string or null)
      const startDate = testData.startDate ? new Date(testData.startDate).toISOString() : null;
      const endDate = testData.endDate ? new Date(testData.endDate).toISOString() : null;

      // Prepare payload
      const payload = {
        title: testData.title,
        description: testData.description,
        duration: parseInt(testData.duration),
        total_marks: parseInt(testData.totalMarks),
        questions: formattedQuestions,
        start_date: startDate,
        end_date: endDate
      };

      console.log('Creating test with payload:', payload);

      // Call API to create test
      const response = await testsAPI.createTest(payload);
      console.log('Test created successfully:', response.data);

      // If we want to publish immediately (not draft)
      if (shouldPublish && response.data.id) {
        try {
          await testsAPI.publishTest(response.data.id);
          console.log('Test published successfully');
        } catch (publishError) {
          console.error('Failed to publish test:', publishError);
          setError('Test created but failed to publish. You can publish it from the tests list.');
        }
      }

      // Navigate back to tests list
      setTimeout(() => {
        navigate('/admin/tests');
      }, 1000);
    } catch (err) {
      console.error('Failed to create test:', err);
      console.error('Error details:', err.response?.data);
      setError(err.response?.data?.error || err.response?.data?.details || 'Failed to create test. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-display font-bold text-dark-900">
            Create New Test
          </h2>
          <p className="text-dark-600 mt-1">
            Set up your examination details and questions
          </p>
        </div>
        <button
          onClick={() => navigate('/admin/tests')}
          className="btn-secondary flex items-center gap-2"
        >
          <X className="w-4 h-4" />
          Cancel
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="card p-4 border-2 border-red-200 bg-red-50">
          <p className="text-red-800 text-sm font-medium">{error}</p>
        </div>
      )}

      {/* Test Details */}
      <div className="card p-6">
        <h3 className="text-xl font-display font-bold text-dark-900 mb-6">
          Test Details
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-dark-700 mb-2">
              Test Title *
            </label>
            <input
              type="text"
              name="title"
              value={testData.title}
              onChange={handleTestDataChange}
              className="input-field w-full"
              placeholder="e.g., Introduction to Machine Learning"
              required
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-dark-700 mb-2">
              Description *
            </label>
            <textarea
              name="description"
              value={testData.description}
              onChange={handleTestDataChange}
              className="input-field w-full h-24 resize-none"
              placeholder="Brief description of the test..."
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-dark-700 mb-2">
              Duration (minutes) *
            </label>
            <input
              type="number"
              name="duration"
              value={testData.duration}
              onChange={handleTestDataChange}
              className="input-field w-full"
              min="1"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-dark-700 mb-2">
              Total Marks *
            </label>
            <input
              type="number"
              name="totalMarks"
              value={testData.totalMarks}
              onChange={handleTestDataChange}
              className="input-field w-full"
              min="1"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-dark-700 mb-2">
              Start Date & Time
            </label>
            <input
              type="datetime-local"
              name="startDate"
              value={testData.startDate}
              onChange={handleTestDataChange}
              className="input-field w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-dark-700 mb-2">
              End Date & Time
            </label>
            <input
              type="datetime-local"
              name="endDate"
              value={testData.endDate}
              onChange={handleTestDataChange}
              className="input-field w-full"
            />
          </div>
        </div>
      </div>

      {/* Questions */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-display font-bold text-dark-900">
            Questions ({questions.length})
          </h3>
          <button
            onClick={addQuestion}
            className="btn-primary flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Question
          </button>
        </div>

        <div className="space-y-6">
          {questions.map((q, index) => (
            <div key={q.id} className="p-6 bg-dark-50 rounded-xl">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-dark-900">
                  Question {index + 1}
                </h4>
                {questions.length > 1 && (
                  <button
                    onClick={() => removeQuestion(q.id)}
                    className="text-red-600 hover:text-red-700"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                )}
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-dark-700 mb-2">
                    Question Text *
                  </label>
                  <textarea
                    value={q.question}
                    onChange={(e) => handleQuestionChange(q.id, 'question', e.target.value)}
                    className="input-field w-full h-20 resize-none"
                    placeholder="Enter your question..."
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-dark-700 mb-2">
                      Question Type
                    </label>
                    <select
                      value={q.type}
                      onChange={(e) => handleQuestionChange(q.id, 'type', e.target.value)}
                      className="input-field w-full"
                    >
                      <option value="multiple-choice">Multiple Choice</option>
                      <option value="text">Text Answer</option>
                      <option value="true-false">True/False</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-dark-700 mb-2">
                      Marks *
                    </label>
                    <input
                      type="number"
                      value={q.marks}
                      onChange={(e) => handleQuestionChange(q.id, 'marks', parseInt(e.target.value) || 1)}
                      className="input-field w-full"
                      min="1"
                      required
                    />
                  </div>
                </div>

                {q.type === 'multiple-choice' && (
                  <div>
                    <label className="block text-sm font-medium text-dark-700 mb-2">
                      Options
                    </label>
                    <div className="space-y-2">
                      {q.options.map((option, optIndex) => (
                        <div key={optIndex} className="flex items-center gap-2">
                          <input
                            type="radio"
                            name={`correct-${q.id}`}
                            checked={q.correctAnswer === String(optIndex)}
                            onChange={() => handleQuestionChange(q.id, 'correctAnswer', String(optIndex))}
                            className="w-4 h-4"
                          />
                          <input
                            type="text"
                            value={option}
                            onChange={(e) => handleOptionChange(q.id, optIndex, e.target.value)}
                            className="input-field flex-1"
                            placeholder={`Option ${optIndex + 1}`}
                          />
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-dark-500 mt-2">
                      Select the radio button for the correct answer
                    </p>
                  </div>
                )}

                {q.type === 'text' && (
                  <div>
                    <label className="block text-sm font-medium text-dark-700 mb-2">
                      Expected Answer (for reference)
                    </label>
                    <textarea
                      value={q.correctAnswer}
                      onChange={(e) => handleQuestionChange(q.id, 'correctAnswer', e.target.value)}
                      className="input-field w-full h-20 resize-none"
                      placeholder="Enter expected answer keywords or criteria..."
                    />
                    <p className="text-xs text-dark-500 mt-1">
                      This will not be auto-graded. Manual grading required.
                    </p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-4 justify-end sticky bottom-6">
        <button
          onClick={(e) => handleSubmit(e, false)}
          disabled={saving}
          className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? 'Saving...' : 'Save as Draft'}
        </button>
        <button
          onClick={(e) => handleSubmit(e, true)}
          disabled={saving}
          className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Save className="w-4 h-4" />
          {saving ? 'Publishing...' : 'Publish Test'}
        </button>
      </div>
    </div>
  );
};

export default CreateTest;