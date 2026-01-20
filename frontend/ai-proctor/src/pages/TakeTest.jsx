// // import React, { useState, useEffect, useRef } from 'react';
// // import { useParams, useNavigate } from 'react-router-dom';
// // import Webcam from 'react-webcam';
// // import { studentTestsAPI, proctoringAPI } from '../services/api';
// // import {
// //   Camera,
// //   CameraOff,
// //   Clock,
// //   AlertTriangle,
// //   CheckCircle,
// //   XCircle,
// //   Eye,
// //   Users,
// //   Monitor
// // } from 'lucide-react';
// // import { mockQuestions } from '../services/mockData';

// // const TakeTest = () => {
// //   const { testId } = useParams();
// //   const navigate = useNavigate();
// //   const webcamRef = useRef(null);

// //   // Test state
// //   const [testStarted, setTestStarted] = useState(false);
// //   const [currentQuestion, setCurrentQuestion] = useState(0);
// //   const [answers, setAnswers] = useState({});
// //   const [timeRemaining, setTimeRemaining] = useState(3600); // 60 minutes in seconds
// //   const [testSubmitted, setTestSubmitted] = useState(false);

// //   // Proctoring state
// //   const [cameraEnabled, setCameraEnabled] = useState(false);
// //   const [cameraError, setCameraError] = useState(null);
// //   const [violations, setViolations] = useState([]);
// //   const [riskScore, setRiskScore] = useState(0);
// //   const [proctoringData, setProctoringData] = useState({
// //     faceDetected: false,
// //     multipleFaces: false,
// //     lookingAway: false,
// //     tabSwitches: 0
// //   });

// //   // Mock questions
// //   const questions = mockQuestions;

// //   // Timer
// //   useEffect(() => {
// //     if (testStarted && !testSubmitted && timeRemaining > 0) {
// //       const timer = setInterval(() => {
// //         setTimeRemaining(prev => {
// //           if (prev <= 1) {
// //             handleSubmitTest();
// //             return 0;
// //           }
// //           return prev - 1;
// //         });
// //       }, 1000);

// //       return () => clearInterval(timer);
// //     }
// //   }, [testStarted, testSubmitted, timeRemaining]);

// //   // Tab visibility monitoring
// //   useEffect(() => {
// //     if (testStarted && !testSubmitted) {
// //       const handleVisibilityChange = () => {
// //         if (document.hidden) {
// //           const newViolation = {
// //             type: 'tab-switch',
// //             timestamp: new Date().toISOString(),
// //             severity: 'medium'
// //           };
// //           setViolations(prev => [...prev, newViolation]);
// //           setProctoringData(prev => ({
// //             ...prev,
// //             tabSwitches: prev.tabSwitches + 1
// //           }));
// //           setRiskScore(prev => Math.min(100, prev + 10));
// //         }
// //       };

// //       document.addEventListener('visibilitychange', handleVisibilityChange);
// //       return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
// //     }
// //   }, [testStarted, testSubmitted]);

// //   // Simulated AI proctoring (replace with actual AI when backend is ready)
// //   useEffect(() => {
// //     if (testStarted && !testSubmitted && cameraEnabled) {
// //       const proctoringInterval = setInterval(() => {
// //         // Simulate random proctoring checks
// //         const randomCheck = Math.random();
        
// //         if (randomCheck < 0.05) { // 5% chance of detecting violation
// //           const violationType = Math.random() < 0.5 ? 'multiple-faces' : 'face-not-visible';
// //           const newViolation = {
// //             type: violationType,
// //             timestamp: new Date().toISOString(),
// //             severity: 'high'
// //           };
// //           setViolations(prev => [...prev, newViolation]);
// //           setRiskScore(prev => Math.min(100, prev + 15));
          
// //           setProctoringData(prev => ({
// //             ...prev,
// //             multipleFaces: violationType === 'multiple-faces',
// //             faceDetected: violationType !== 'face-not-visible'
// //           }));
// //         } else {
// //           setProctoringData(prev => ({
// //             ...prev,
// //             faceDetected: true,
// //             multipleFaces: false
// //           }));
// //         }
// //       }, 3000); // Check every 3 seconds

// //       return () => clearInterval(proctoringInterval);
// //     }
// //   }, [testStarted, testSubmitted, cameraEnabled]);

// //   const formatTime = (seconds) => {
// //     const mins = Math.floor(seconds / 60);
// //     const secs = seconds % 60;
// //     return `${mins}:${secs.toString().padStart(2, '0')}`;
// //   };

// //   const handleStartTest = async () => {
// //     try {
// //       // Request camera permission
// //       await navigator.mediaDevices.getUserMedia({ video: true });
// //       setCameraEnabled(true);
// //       setTestStarted(true);
// //     } catch (error) {
// //       setCameraError('Camera access denied. Please enable camera to start the test.');
// //     }
// //   };

// //   const handleAnswerChange = (questionId, answer) => {
// //     setAnswers(prev => ({
// //       ...prev,
// //       [questionId]: answer
// //     }));
// //   };

// //   const handleNextQuestion = () => {
// //     if (currentQuestion < questions.length - 1) {
// //       setCurrentQuestion(currentQuestion + 1);
// //     }
// //   };

// //   const handlePreviousQuestion = () => {
// //     if (currentQuestion > 0) {
// //       setCurrentQuestion(currentQuestion - 1);
// //     }
// //   };

// //   // const handleSubmitTest = async () => {
// //   //   if (window.confirm('Are you sure you want to submit the test? You cannot change answers after submission.')) {
// //   //     setTestSubmitted(true);
      
// //   //     // TODO: Submit to backend
// //   //     // const result = await studentTestsAPI.submitTest(testId, {
// //   //     //   answers,
// //   //     //   violations,
// //   //     //   riskScore
// //   //     // });
      
// //   //     setTimeout(() => {
// //   //       navigate('/student/results');
// //   //     }, 2000);
// //   //   }
// //   // };

// //   const handleSubmitTest = async () => {
// //   if (window.confirm('Are you sure you want to submit?')) {
// //     setTestSubmitted(true);
    
// //     try {
// //       // 1. Submit the actual answers
// //       await studentTestsAPI.submitTest(testId, answers);
      
// //       // 2. Submit the proctoring violations logged during the session
// //       // Assuming your backend expects a session ID or uses the testId
// //       await proctoringAPI.reportViolation(testId, {
// //         violations,
// //         finalRiskScore: riskScore
// //       });

// //       navigate('/student/results');
// //     } catch (error) {
// //       console.error("Submission failed", error);
// //     }
// //   }
// // };

// //   // Pre-test screen
// //   if (!testStarted) {
// //     return (
// //       <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
// //         <div className="card p-8">
// //           <h2 className="text-2xl font-display font-bold text-dark-900 mb-4">
// //             Test Instructions
// //           </h2>
          
// //           <div className="space-y-4 mb-6">
// //             <div className="flex items-start gap-3">
// //               <Camera className="w-5 h-5 text-dark-900 mt-0.5" />
// //               <div>
// //                 <h3 className="font-semibold text-dark-900 mb-1">Camera Monitoring</h3>
// //                 <p className="text-sm text-dark-600">
// //                   Your camera will monitor you throughout the test. Keep your face visible at all times.
// //                 </p>
// //               </div>
// //             </div>

// //             <div className="flex items-start gap-3">
// //               <Monitor className="w-5 h-5 text-dark-900 mt-0.5" />
// //               <div>
// //                 <h3 className="font-semibold text-dark-900 mb-1">Stay in Tab</h3>
// //                 <p className="text-sm text-dark-600">
// //                   Do not switch tabs or open other windows. This will be flagged as suspicious activity.
// //                 </p>
// //               </div>
// //             </div>

// //             <div className="flex items-start gap-3">
// //               <Users className="w-5 h-5 text-dark-900 mt-0.5" />
// //               <div>
// //                 <h3 className="font-semibold text-dark-900 mb-1">No Additional People</h3>
// //                 <p className="text-sm text-dark-600">
// //                   Ensure no other person is visible in the camera frame during the test.
// //                 </p>
// //               </div>
// //             </div>
// //           </div>

// //           {cameraError && (
// //             <div className="mb-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg">
// //               <p className="text-red-800 text-sm">{cameraError}</p>
// //             </div>
// //           )}

// //           <div className="bg-yellow-50 border-2 border-yellow-200 rounded-lg p-4 mb-6">
// //             <div className="flex items-start gap-3">
// //               <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
// //               <div className="text-sm text-yellow-800">
// //                 <p className="font-semibold mb-1">Important:</p>
// //                 <p>Once you start, the timer will begin and cannot be paused. Make sure you're ready.</p>
// //               </div>
// //             </div>
// //           </div>

// //           <button
// //             onClick={handleStartTest}
// //             className="btn-primary w-full"
// //           >
// //             I'm Ready, Start Test
// //           </button>
// //         </div>
// //       </div>
// //     );
// //   }

// //   // Post-test screen
// //   if (testSubmitted) {
// //     return (
// //       <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
// //         <div className="card p-8 text-center">
// //           <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
// //             <CheckCircle className="w-8 h-8 text-green-600" />
// //           </div>
// //           <h2 className="text-2xl font-display font-bold text-dark-900 mb-2">
// //             Test Submitted Successfully!
// //           </h2>
// //           <p className="text-dark-600 mb-6">
// //             Your answers have been recorded. You will be redirected to your results shortly.
// //           </p>
// //           <div className="animate-spin rounded-full h-8 w-8 border-4 border-dark-900 border-t-transparent mx-auto"></div>
// //         </div>
// //       </div>
// //     );
// //   }

// //   const question = questions[currentQuestion];

// //   return (
// //     <div className="space-y-6">
// //       {/* Top Bar */}
// //       <div className="bg-white border-2 border-dark-100 rounded-xl p-4 sticky top-20 z-20">
// //         <div className="flex items-center justify-between">
// //           <div className="flex items-center gap-6">
// //             <div className="flex items-center gap-2">
// //               <Clock className="w-5 h-5 text-dark-600" />
// //               <span className={`text-lg font-bold ${timeRemaining < 300 ? 'text-red-600' : 'text-dark-900'}`}>
// //                 {formatTime(timeRemaining)}
// //               </span>
// //             </div>
            
// //             <div className="flex items-center gap-2">
// //               <span className="text-sm text-dark-600">
// //                 Question {currentQuestion + 1} of {questions.length}
// //               </span>
// //             </div>
// //           </div>

// //           <button
// //             onClick={handleSubmitTest}
// //             className="btn-primary"
// //           >
// //             Submit Test
// //           </button>
// //         </div>
// //       </div>

// //       <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
// //         {/* Question Area */}
// //         <div className="lg:col-span-3 space-y-6">
// //           <div className="card p-8">
// //             <div className="mb-6">
// //               <div className="text-sm text-dark-600 mb-2">
// //                 Question {currentQuestion + 1}
// //               </div>
// //               <h3 className="text-xl font-display font-semibold text-dark-900 mb-4">
// //                 {question.question}
// //               </h3>
// //             </div>

// //             {question.type === 'multiple-choice' ? (
// //               <div className="space-y-3">
// //                 {question.options.map((option, index) => (
// //                   <label
// //                     key={index}
// //                     className="flex items-center gap-3 p-4 border-2 border-dark-200 rounded-lg cursor-pointer hover:border-dark-900 transition-colors"
// //                   >
// //                     <input
// //                       type="radio"
// //                       name={`question-${question.id}`}
// //                       checked={answers[question.id] === index}
// //                       onChange={() => handleAnswerChange(question.id, index)}
// //                       className="w-5 h-5"
// //                     />
// //                     <span className="text-dark-900">{option}</span>
// //                   </label>
// //                 ))}
// //               </div>
// //             ) : (
// //               <textarea
// //                 value={answers[question.id] || ''}
// //                 onChange={(e) => handleAnswerChange(question.id, e.target.value)}
// //                 className="input-field w-full h-48 resize-none"
// //                 placeholder="Type your answer here..."
// //               />
// //             )}

// //             <div className="flex justify-between mt-8">
// //               <button
// //                 onClick={handlePreviousQuestion}
// //                 disabled={currentQuestion === 0}
// //                 className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
// //               >
// //                 Previous
// //               </button>
// //               <button
// //                 onClick={handleNextQuestion}
// //                 disabled={currentQuestion === questions.length - 1}
// //                 className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
// //               >
// //                 Next
// //               </button>
// //             </div>
// //           </div>
// //         </div>

// //         {/* Proctoring Sidebar */}
// //         <div className="space-y-4">
// //           {/* Camera Feed */}
// //           <div className="card p-4">
// //             <h4 className="font-semibold text-dark-900 mb-3 flex items-center gap-2">
// //               <Camera className="w-4 h-4" />
// //               Camera Monitor
// //             </h4>
// //             <div className="aspect-video bg-dark-900 rounded-lg overflow-hidden mb-3">
// //               {cameraEnabled ? (
// //                 <Webcam
// //                   ref={webcamRef}
// //                   audio={false}
// //                   screenshotFormat="image/jpeg"
// //                   className="w-full h-full object-cover"
// //                 />
// //               ) : (
// //                 <div className="w-full h-full flex items-center justify-center">
// //                   <CameraOff className="w-8 h-8 text-dark-400" />
// //                 </div>
// //               )}
// //             </div>
// //             <div className="space-y-2 text-xs">
// //               <div className="flex items-center justify-between">
// //                 <span className="text-dark-600">Face Detected:</span>
// //                 {proctoringData.faceDetected ? (
// //                   <CheckCircle className="w-4 h-4 text-green-600" />
// //                 ) : (
// //                   <XCircle className="w-4 h-4 text-red-600" />
// //                 )}
// //               </div>
// //               <div className="flex items-center justify-between">
// //                 <span className="text-dark-600">Multiple Faces:</span>
// //                 {proctoringData.multipleFaces ? (
// //                   <XCircle className="w-4 h-4 text-red-600" />
// //                 ) : (
// //                   <CheckCircle className="w-4 h-4 text-green-600" />
// //                 )}
// //               </div>
// //             </div>
// //           </div>

// //           {/* Risk Score */}
// //           <div className="card p-4">
// //             <h4 className="font-semibold text-dark-900 mb-3">Risk Score</h4>
// //             <div className="mb-2">
// //               <div className="flex justify-between text-sm mb-1">
// //                 <span className="text-dark-600">Current Risk</span>
// //                 <span className={`font-bold ${riskScore > 50 ? 'text-red-600' : 'text-green-600'}`}>
// //                   {riskScore}%
// //                 </span>
// //               </div>
// //               <div className="w-full bg-dark-200 rounded-full h-2">
// //                 <div
// //                   className={`h-2 rounded-full transition-all ${
// //                     riskScore > 50 ? 'bg-red-600' : 'bg-green-600'
// //                   }`}
// //                   style={{ width: `${riskScore}%` }}
// //                 />
// //               </div>
// //             </div>
// //             <p className="text-xs text-dark-600">
// //               {violations.length} violation(s) detected
// //             </p>
// //           </div>

// //           {/* Navigation Grid */}
// //           <div className="card p-4">
// //             <h4 className="font-semibold text-dark-900 mb-3">Questions</h4>
// //             <div className="grid grid-cols-5 gap-2">
// //               {questions.map((q, index) => (
// //                 <button
// //                   key={q.id}
// //                   onClick={() => setCurrentQuestion(index)}
// //                   className={`
// //                     aspect-square rounded-lg text-sm font-medium transition-all
// //                     ${currentQuestion === index
// //                       ? 'bg-dark-900 text-white'
// //                       : answers[q.id] !== undefined
// //                         ? 'bg-green-100 text-green-800'
// //                         : 'bg-dark-100 text-dark-600 hover:bg-dark-200'
// //                     }
// //                   `}
// //                 >
// //                   {index + 1}
// //                 </button>
// //               ))}
// //             </div>
// //           </div>
// //         </div>
// //       </div>
// //     </div>
// //   );
// // };

// // export default TakeTest;



// import React, { useState, useEffect, useRef } from 'react';
// import { useParams, useNavigate } from 'react-router-dom';
// import Webcam from 'react-webcam';
// import {
//   Camera,
//   CameraOff,
//   Clock,
//   AlertTriangle,
//   CheckCircle,
//   XCircle,
//   Eye,
//   Users,
//   Monitor
// } from 'lucide-react';
// import { mockQuestions } from '../services/mockData';

// const TakeTest = () => {
//   const { testId } = useParams();
//   const navigate = useNavigate();
//   const webcamRef = useRef(null);
//   const isSubmittingRef = useRef(false); 

//   // Test state
//   const [testStarted, setTestStarted] = useState(false);
//   const [currentQuestion, setCurrentQuestion] = useState(0);
//   const [answers, setAnswers] = useState({});
//   const [timeRemaining, setTimeRemaining] = useState(3600); // 60 minutes in seconds
//   const [testSubmitted, setTestSubmitted] = useState(false);

//   // Proctoring state
//   const [cameraEnabled, setCameraEnabled] = useState(false);
//   const [cameraError, setCameraError] = useState(null);
//   const [violations, setViolations] = useState([]);
//   const [riskScore, setRiskScore] = useState(0);
//   const [proctoringData, setProctoringData] = useState({
//     faceDetected: false,
//     multipleFaces: false,
//     lookingAway: false,
//     tabSwitches: 0
//   });
//   const [showWarning, setShowWarning] = useState(false);
//   const MAX_SWITCH_LIMIT = 3;

//   // Mock questions
//   const questions = mockQuestions;

//   // Timer
//   useEffect(() => {
//     if (testStarted && !testSubmitted && timeRemaining > 0) {
//       const timer = setInterval(() => {
//         setTimeRemaining(prev => {
//           if (prev <= 1) {
//             handleSubmitTest();
//             return 0;
//           }
//           return prev - 1;
//         });
//       }, 1000);

//       return () => clearInterval(timer);
//     }
//   }, [testStarted, testSubmitted, timeRemaining]);

//   // Tab visibility monitoring
// useEffect(() => {
//   if (testStarted && !testSubmitted) {
//     const handleVisibilityChange = () => {
//       // Check the ref to see if we are already in the process of submitting
//       if (document.hidden && !isSubmittingRef.current) {
//         const newViolation = {
//           type: 'tab-switch',
//           timestamp: new Date().toISOString(),
//           severity: 'medium'
//         };

//         setViolations(prev => {
//           const updatedViolations = [...prev, newViolation];
//           const switchCount = updatedViolations.filter(v => v.type === 'tab-switch').length;

//           // Check if limit is reached AND we haven't started submitting yet
//           if (switchCount >= 3 && !isSubmittingRef.current) {
//             isSubmittingRef.current = true; // Lock the submission process immediately
//             alert("Maximum tab switches reached. Your test is being submitted.");
            
//             // Bypass the window.confirm in handleSubmitTest for auto-submission
//             setTestSubmitted(true); 
//             setTimeout(() => {
//               navigate('/student/results');
//             }, 2000);
//           } else if (switchCount < 3) {
//             setShowWarning(true);
//           }
          
//           return updatedViolations;
//         });

//         setProctoringData(prev => ({
//           ...prev,
//           tabSwitches: prev.tabSwitches + 1
//         }));
//         setRiskScore(prev => Math.min(100, prev + 20));
//       }
//     };

//     document.addEventListener('visibilitychange', handleVisibilityChange);
//     return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
//   }
// }, [testStarted, testSubmitted, navigate]);

//   // Simulated AI proctoring (replace with actual AI when backend is ready)
//   useEffect(() => {
//     if (testStarted && !testSubmitted && cameraEnabled) {
//       const proctoringInterval = setInterval(() => {
//         // Simulate random proctoring checks
//         const randomCheck = Math.random();
        
//         if (randomCheck < 0.05) { // 5% chance of detecting violation
//           const violationType = Math.random() < 0.5 ? 'multiple-faces' : 'face-not-visible';
//           const newViolation = {
//             type: violationType,
//             timestamp: new Date().toISOString(),
//             severity: 'high'
//           };
//           setViolations(prev => [...prev, newViolation]);
//           setRiskScore(prev => Math.min(100, prev + 15));
          
//           setProctoringData(prev => ({
//             ...prev,
//             multipleFaces: violationType === 'multiple-faces',
//             faceDetected: violationType !== 'face-not-visible'
//           }));
//         } else {
//           setProctoringData(prev => ({
//             ...prev,
//             faceDetected: true,
//             multipleFaces: false
//           }));
//         }
//       }, 3000); // Check every 3 seconds

//       return () => clearInterval(proctoringInterval);
//     }
//   }, [testStarted, testSubmitted, cameraEnabled]);

//   const formatTime = (seconds) => {
//     const mins = Math.floor(seconds / 60);
//     const secs = seconds % 60;
//     return `${mins}:${secs.toString().padStart(2, '0')}`;
//   };

//   const handleStartTest = async () => {
//     try {
//       // Request camera permission
//       await navigator.mediaDevices.getUserMedia({ video: true });
//       setCameraEnabled(true);
//       setTestStarted(true);
//     } catch (error) {
//       setCameraError('Camera access denied. Please enable camera to start the test.');
//     }
//   };

//   const handleAnswerChange = (questionId, answer) => {
//     setAnswers(prev => ({
//       ...prev,
//       [questionId]: answer
//     }));
//   };

//   const handleNextQuestion = () => {
//     if (currentQuestion < questions.length - 1) {
//       setCurrentQuestion(currentQuestion + 1);
//     }
//   };

//   const handlePreviousQuestion = () => {
//     if (currentQuestion > 0) {
//       setCurrentQuestion(currentQuestion - 1);
//     }
//   };

//   const handleSubmitTest = async () => {
//     if (window.confirm('Are you sure you want to submit the test? You cannot change answers after submission.')) {
//       setTestSubmitted(true);
      
//       // TODO: Submit to backend
//       // const result = await studentTestsAPI.submitTest(testId, {
//       //   answers,
//       //   violations,
//       //   riskScore
//       // });
      
//       setTimeout(() => {
//         navigate('/student/results');
//       }, 2000);
//     }
//   };

//   // Pre-test screen
//   if (!testStarted) {
//     return (
//       <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
//         <div className="card p-8">
//           <h2 className="text-2xl font-display font-bold text-dark-900 mb-4">
//             Test Instructions
//           </h2>
          
//           <div className="space-y-4 mb-6">
//             <div className="flex items-start gap-3">
//               <Camera className="w-5 h-5 text-dark-900 mt-0.5" />
//               <div>
//                 <h3 className="font-semibold text-dark-900 mb-1">Camera Monitoring</h3>
//                 <p className="text-sm text-dark-600">
//                   Your camera will monitor you throughout the test. Keep your face visible at all times.
//                 </p>
//               </div>
//             </div>

//             <div className="flex items-start gap-3">
//               <Monitor className="w-5 h-5 text-dark-900 mt-0.5" />
//               <div>
//                 <h3 className="font-semibold text-dark-900 mb-1">Stay in Tab</h3>
//                 <p className="text-sm text-dark-600">
//                   Do not switch tabs or open other windows. This will be flagged as suspicious activity.
//                 </p>
//               </div>
//             </div>

//             <div className="flex items-start gap-3">
//               <Users className="w-5 h-5 text-dark-900 mt-0.5" />
//               <div>
//                 <h3 className="font-semibold text-dark-900 mb-1">No Additional People</h3>
//                 <p className="text-sm text-dark-600">
//                   Ensure no other person is visible in the camera frame during the test.
//                 </p>
//               </div>
//             </div>
//           </div>

//           {cameraError && (
//             <div className="mb-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg">
//               <p className="text-red-800 text-sm">{cameraError}</p>
//             </div>
//           )}

//           <div className="bg-yellow-50 border-2 border-yellow-200 rounded-lg p-4 mb-6">
//             <div className="flex items-start gap-3">
//               <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
//               <div className="text-sm text-yellow-800">
//                 <p className="font-semibold mb-1">Important:</p>
//                 <p>Once you start, the timer will begin and cannot be paused. Make sure you're ready.</p>
//               </div>
//             </div>
//           </div>

//           <button
//             onClick={handleStartTest}
//             className="btn-primary w-full"
//           >
//             I'm Ready, Start Test
//           </button>
//         </div>
//       </div>
//     );
//   }

//   // Post-test screen
//   if (testSubmitted) {
//     return (
//       <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
//         <div className="card p-8 text-center">
//           <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
//             <CheckCircle className="w-8 h-8 text-green-600" />
//           </div>
//           <h2 className="text-2xl font-display font-bold text-dark-900 mb-2">
//             Test Submitted Successfully!
//           </h2>
//           <p className="text-dark-600 mb-6">
//             Your answers have been recorded. You will be redirected to your results shortly.
//           </p>
//           <div className="animate-spin rounded-full h-8 w-8 border-4 border-dark-900 border-t-transparent mx-auto"></div>
//         </div>
//       </div>
//     );
//   }

//   const question = questions[currentQuestion];

//   return (
//     <div className="space-y-6">
//       {/* Top Bar */}
//       <div className="bg-white border-2 border-dark-100 rounded-xl p-4 sticky top-20 z-20">
//         <div className="flex items-center justify-between">
//           <div className="flex items-center gap-6">
//             <div className="flex items-center gap-2">
//               <Clock className="w-5 h-5 text-dark-600" />
//               <span className={`text-lg font-bold ${timeRemaining < 300 ? 'text-red-600' : 'text-dark-900'}`}>
//                 {formatTime(timeRemaining)}
//               </span>
//             </div>
            
//             <div className="flex items-center gap-2">
//               <span className="text-sm text-dark-600">
//                 Question {currentQuestion + 1} of {questions.length}
//               </span>
//             </div>
//           </div>

//           <button
//             onClick={handleSubmitTest}
//             className="btn-primary"
//           >
//             Submit Test
//           </button>
//         </div>
//       </div>

//       <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
//         {/* Question Area */}
//         <div className="lg:col-span-3 space-y-6">
//           <div className="card p-8">
//             <div className="mb-6">
//               <div className="text-sm text-dark-600 mb-2">
//                 Question {currentQuestion + 1}
//               </div>
//               <h3 className="text-xl font-display font-semibold text-dark-900 mb-4">
//                 {question.question}
//               </h3>
//             </div>

//             {question.type === 'multiple-choice' ? (
//               <div className="space-y-3">
//                 {question.options.map((option, index) => (
//                   <label
//                     key={index}
//                     className="flex items-center gap-3 p-4 border-2 border-dark-200 rounded-lg cursor-pointer hover:border-dark-900 transition-colors"
//                   >
//                     <input
//                       type="radio"
//                       name={`question-${question.id}`}
//                       checked={answers[question.id] === index}
//                       onChange={() => handleAnswerChange(question.id, index)}
//                       className="w-5 h-5"
//                     />
//                     <span className="text-dark-900">{option}</span>
//                   </label>
//                 ))}
//               </div>
//             ) : (
//               <textarea
//                 value={answers[question.id] || ''}
//                 onChange={(e) => handleAnswerChange(question.id, e.target.value)}
//                 className="input-field w-full h-48 resize-none"
//                 placeholder="Type your answer here..."
//               />
//             )}

//             <div className="flex justify-between mt-8">
//               <button
//                 onClick={handlePreviousQuestion}
//                 disabled={currentQuestion === 0}
//                 className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
//               >
//                 Previous
//               </button>
//               <button
//                 onClick={handleNextQuestion}
//                 disabled={currentQuestion === questions.length - 1}
//                 className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
//               >
//                 Next
//               </button>
//             </div>
//           </div>
//         </div>

//         {/* Proctoring Sidebar */}
//         <div className="space-y-4">
//           {/* Camera Feed */}
//           <div className="card p-4">
//             <h4 className="font-semibold text-dark-900 mb-3 flex items-center gap-2">
//               <Camera className="w-4 h-4" />
//               Camera Monitor
//             </h4>
//             <div className="aspect-video bg-dark-900 rounded-lg overflow-hidden mb-3">
//               {cameraEnabled ? (
//                 <Webcam
//                   ref={webcamRef}
//                   audio={false}
//                   screenshotFormat="image/jpeg"
//                   className="w-full h-full object-cover"
//                 />
//               ) : (
//                 <div className="w-full h-full flex items-center justify-center">
//                   <CameraOff className="w-8 h-8 text-dark-400" />
//                 </div>
//               )}
//             </div>
//             <div className="space-y-2 text-xs">
//               <div className="flex items-center justify-between">
//                 <span className="text-dark-600">Face Detected:</span>
//                 {proctoringData.faceDetected ? (
//                   <CheckCircle className="w-4 h-4 text-green-600" />
//                 ) : (
//                   <XCircle className="w-4 h-4 text-red-600" />
//                 )}
//               </div>
//               <div className="flex items-center justify-between">
//                 <span className="text-dark-600">Multiple Faces:</span>
//                 {proctoringData.multipleFaces ? (
//                   <XCircle className="w-4 h-4 text-red-600" />
//                 ) : (
//                   <CheckCircle className="w-4 h-4 text-green-600" />
//                 )}
//               </div>
//             </div>
//           </div>

//           {/* Risk Score */}
//           <div className="card p-4">
//             <h4 className="font-semibold text-dark-900 mb-3">Risk Score</h4>
//             <div className="mb-2">
//               <div className="flex justify-between text-sm mb-1">
//                 <span className="text-dark-600">Current Risk</span>
//                 <span className={`font-bold ${riskScore > 50 ? 'text-red-600' : 'text-green-600'}`}>
//                   {riskScore}%
//                 </span>
//               </div>
//               <div className="w-full bg-dark-200 rounded-full h-2">
//                 <div
//                   className={`h-2 rounded-full transition-all ${
//                     riskScore > 50 ? 'bg-red-600' : 'bg-green-600'
//                   }`}
//                   style={{ width: `${riskScore}%` }}
//                 />
//               </div>
//             </div>
//             <p className="text-xs text-dark-600">
//               {violations.length} violation(s) detected
//             </p>
//           </div>

//           {/* Navigation Grid */}
//           <div className="card p-4">
//             <h4 className="font-semibold text-dark-900 mb-3">Questions</h4>
//             <div className="grid grid-cols-5 gap-2">
//               {questions.map((q, index) => (
//                 <button
//                   key={q.id}
//                   onClick={() => setCurrentQuestion(index)}
//                   className={`
//                     aspect-square rounded-lg text-sm font-medium transition-all
//                     ${currentQuestion === index
//                       ? 'bg-dark-900 text-white'
//                       : answers[q.id] !== undefined
//                         ? 'bg-green-100 text-green-800'
//                         : 'bg-dark-100 text-dark-600 hover:bg-dark-200'
//                     }
//                   `}
//                 >
//                   {index + 1}
//                 </button>
//               ))}
//             </div>
//           </div>
//         </div>
//       </div>
//       {/* Tab Switch Warning Modal */}
//       {showWarning && (
//         <div className="fixed inset-0 bg-dark-900/90 backdrop-blur-sm flex items-center justify-center p-4 z-[100]">
//           <div className="card max-w-md w-full p-8 border-red-500 border-4">
//             <div className="text-center">
//               <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
//                 <AlertTriangle className="w-8 h-8 text-red-600" />
//               </div>
//               <h2 className="text-2xl font-display font-bold text-red-600 mb-2">
//                 Warning: Tab Switch Detected!
//               </h2>
//               <p className="text-dark-700 mb-6">
//                 You have switched tabs {proctoringData.tabSwitches} time(s). 
//                 Switching tabs more than <strong>{MAX_SWITCH_LIMIT}</strong> times will result in automatic submission.
//               </p>
//               <button
//                 onClick={() => setShowWarning(false)}
//                 className="btn-danger w-full py-4 text-lg font-bold"
//               >
//                 I Understand, Return to Test
//               </button>
//             </div>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default TakeTest;

import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Webcam from 'react-webcam';
import { studentTestsAPI } from '../services/api';
import {
  Camera,
  CameraOff,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Eye,
  Users,
  Monitor
} from 'lucide-react';

const TakeTest = () => {
  const { testId } = useParams();
  const navigate = useNavigate();
  const webcamRef = useRef(null);
  const isSubmittingRef = useRef(false); 

  // Test state
  const [testData, setTestData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [testStarted, setTestStarted] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [testSubmitted, setTestSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Proctoring state
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [violations, setViolations] = useState([]);
  const [riskScore, setRiskScore] = useState(0);
  const [proctoringData, setProctoringData] = useState({
    faceDetected: false,
    multipleFaces: false,
    lookingAway: false,
    tabSwitches: 0
  });
  const [showWarning, setShowWarning] = useState(false);
  const MAX_SWITCH_LIMIT = 3;

  // Timer
  useEffect(() => {
    if (testStarted && !testSubmitted && timeRemaining > 0) {
      const timer = setInterval(() => {
        setTimeRemaining(prev => {
          if (prev <= 1) {
            handleSubmitTest();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [testStarted, testSubmitted, timeRemaining]);

  // Tab visibility monitoring
  useEffect(() => {
    if (testStarted && !testSubmitted) {
      const handleVisibilityChange = () => {
        if (document.hidden && !isSubmittingRef.current) {
          const newViolation = {
            type: 'tab-switch',
            timestamp: new Date().toISOString(),
            severity: 'medium'
          };

          setViolations(prev => {
            const updatedViolations = [...prev, newViolation];
            const switchCount = updatedViolations.filter(v => v.type === 'tab-switch').length;

            if (switchCount >= 3 && !isSubmittingRef.current) {
              isSubmittingRef.current = true;
              alert("Maximum tab switches reached. Your test is being submitted.");
              
              setTestSubmitted(true); 
              setTimeout(() => {
                navigate('/student/results');
              }, 2000);
            } else if (switchCount < 3) {
              setShowWarning(true);
            }
            
            return updatedViolations;
          });

          setProctoringData(prev => ({
            ...prev,
            tabSwitches: prev.tabSwitches + 1
          }));
          setRiskScore(prev => Math.min(100, prev + 20));
        }
      };

      document.addEventListener('visibilitychange', handleVisibilityChange);
      return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
    }
  }, [testStarted, testSubmitted, navigate]);

  // Simulated AI proctoring
  useEffect(() => {
    if (testStarted && !testSubmitted && cameraEnabled) {
      const proctoringInterval = setInterval(() => {
        const randomCheck = Math.random();
        
        if (randomCheck < 0.05) {
          const violationType = Math.random() < 0.5 ? 'multiple-faces' : 'face-not-visible';
          const newViolation = {
            type: violationType,
            timestamp: new Date().toISOString(),
            severity: 'high'
          };
          setViolations(prev => [...prev, newViolation]);
          setRiskScore(prev => Math.min(100, prev + 15));
          
          setProctoringData(prev => ({
            ...prev,
            multipleFaces: violationType === 'multiple-faces',
            faceDetected: violationType !== 'face-not-visible'
          }));
        } else {
          setProctoringData(prev => ({
            ...prev,
            faceDetected: true,
            multipleFaces: false
          }));
        }
      }, 3000);

      return () => clearInterval(proctoringInterval);
    }
  }, [testStarted, testSubmitted, cameraEnabled]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStartTest = async () => {
    try {
      setLoading(true);
      
      // Request camera permission
      await navigator.mediaDevices.getUserMedia({ video: true });
      setCameraEnabled(true);
      
      // Call backend API to start test session
      const response = await studentTestsAPI.startTest(testId);
      const { session, test } = response.data;
      
      // Set test data and session info
      setTestData(test);
      setSessionId(session.id);
      setTimeRemaining(test.duration * 60); // Convert minutes to seconds
      setTestStarted(true);
      
    } catch (error) {
      console.error('Failed to start test:', error);
      if (error.message?.includes('Camera')) {
        setCameraError('Camera access denied. Please enable camera to start the test.');
      } else {
        setError(error.response?.data?.error || 'Failed to start test. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerChange = (questionId, answer) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
  };

  const handleNextQuestion = () => {
    if (currentQuestion < testData.questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    }
  };

  const handlePreviousQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  // const handleSubmitTest = async () => {
  //   if (window.confirm('Are you sure you want to submit the test? You cannot change answers after submission.')) {
  //     setTestSubmitted(true);
      
  //     try {
  //       await studentTestsAPI.submitTest(testId, {
  //         answers,
  //         violations,
  //         risk_score: riskScore
  //       });
        
  //       setTimeout(() => {
  //         navigate('/student/results');
  //       }, 2000);
  //     } catch (error) {
  //       console.error('Failed to submit test:', error);
  //       alert('Failed to submit test. Please try again.');
  //       setTestSubmitted(false);
  //     }
  //   }
  // };

//   const handleSubmitTest = async () => {
//   if (window.confirm('Are you sure you want to submit the test? You cannot change answers after submission.')) {
//     setTestSubmitted(true);
    
//     try {
//       // Format answers - convert all to strings as backend expects
//       const formattedAnswers = {};
//       Object.keys(answers).forEach(key => {
//         const answer = answers[key];
//         formattedAnswers[key] = typeof answer === 'number' ? String(answer) : String(answer);
//       });

//       // Format violations to ensure each is a proper object/dict
//       const formattedViolations = violations.map(v => ({
//         type: String(v.type || ''),
//         timestamp: String(v.timestamp || new Date().toISOString()),
//         severity: String(v.severity || 'medium')
//       }));

//       const submissionData = {
//         answers: JSON.stringify(formattedAnswers),  // Stringify the answers object
//         violations: JSON.stringify(formattedViolations),  // Stringify the violations array
//         risk_score: Number(riskScore)
//       };

//       console.log('Submitting test with data:', submissionData);
      
//       await studentTestsAPI.submitTest(testId, submissionData);
      
//       setTimeout(() => {
//         navigate('/student/results');
//       }, 2000);
//     } catch (error) {
//       console.error('Failed to submit test:', error);
//       console.error('Error response:', error.response?.data);
//       alert(`Failed to submit test: ${error.response?.data?.error || 'Please try again.'}`);
//       setTestSubmitted(false);
//     }
//   }
// };

const handleSubmitTest = async () => {
  if (window.confirm('Are you sure you want to submit the test? You cannot change answers after submission.')) {
    setTestSubmitted(true);
    
    try {
      // Format answers - keep as object, don't stringify
      const formattedAnswers = {};
      Object.keys(answers).forEach(key => {
        const answer = answers[key];
        // Convert to string for consistency
        formattedAnswers[key] = typeof answer === 'number' ? String(answer) : String(answer);
      });

      // Format violations - keep as array of objects, don't stringify
      const formattedViolations = violations.map(v => ({
        type: v.type || '',
        timestamp: v.timestamp || new Date().toISOString(),
        severity: v.severity || 'medium'
      }));

      const submissionData = {
        answers: formattedAnswers,
        violations: formattedViolations,
        risk_score: Number(riskScore)
      };

      console.log('Submitting test with data:', submissionData);
      console.log('Stringified:', JSON.stringify(submissionData, null, 2));
      
      await studentTestsAPI.submitTest(testId, submissionData);
      
      alert('Test submitted successfully!');
      setTimeout(() => {
        navigate('/student/results');
      }, 2000);
    } catch (error) {
      console.error('Failed to submit test:', error);
      console.error('Error response:', error.response?.data);
      console.error('Error details:', JSON.stringify(error.response?.data, null, 2));
      alert(`Failed to submit test: ${JSON.stringify(error.response?.data)}`);
      setTestSubmitted(false);
    }
  }
};

  // Pre-test screen
  if (!testStarted) {
    return (
      <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
        <div className="card p-8">
          <h2 className="text-2xl font-display font-bold text-dark-900 mb-4">
            Test Instructions
          </h2>
          
          <div className="space-y-4 mb-6">
            <div className="flex items-start gap-3">
              <Camera className="w-5 h-5 text-dark-900 mt-0.5" />
              <div>
                <h3 className="font-semibold text-dark-900 mb-1">Camera Monitoring</h3>
                <p className="text-sm text-dark-600">
                  Your camera will monitor you throughout the test. Keep your face visible at all times.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <Monitor className="w-5 h-5 text-dark-900 mt-0.5" />
              <div>
                <h3 className="font-semibold text-dark-900 mb-1">Stay in Tab</h3>
                <p className="text-sm text-dark-600">
                  Do not switch tabs or open other windows. This will be flagged as suspicious activity.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <Users className="w-5 h-5 text-dark-900 mt-0.5" />
              <div>
                <h3 className="font-semibold text-dark-900 mb-1">No Additional People</h3>
                <p className="text-sm text-dark-600">
                  Ensure no other person is visible in the camera frame during the test.
                </p>
              </div>
            </div>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg">
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          {cameraError && (
            <div className="mb-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg">
              <p className="text-red-800 text-sm">{cameraError}</p>
            </div>
          )}

          <div className="bg-yellow-50 border-2 border-yellow-200 rounded-lg p-4 mb-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
              <div className="text-sm text-yellow-800">
                <p className="font-semibold mb-1">Important:</p>
                <p>Once you start, the timer will begin and cannot be paused. Make sure you're ready.</p>
              </div>
            </div>
          </div>

          <button
            onClick={handleStartTest}
            disabled={loading}
            className="btn-primary w-full disabled:opacity-50"
          >
            {loading ? 'Starting Test...' : "I'm Ready, Start Test"}
          </button>
        </div>
      </div>
    );
  }

  // Post-test screen
  if (testSubmitted) {
    return (
      <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
        <div className="card p-8 text-center">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
          <h2 className="text-2xl font-display font-bold text-dark-900 mb-2">
            Test Submitted Successfully!
          </h2>
          <p className="text-dark-600 mb-6">
            Your answers have been recorded. You will be redirected to your results shortly.
          </p>
          <div className="animate-spin rounded-full h-8 w-8 border-4 border-dark-900 border-t-transparent mx-auto"></div>
        </div>
      </div>
    );
  }

  if (!testData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-dark-900 border-t-transparent"></div>
      </div>
    );
  }

  const question = testData.questions[currentQuestion];

  return (
    <div className="space-y-6">
      {/* Top Bar */}
      <div className="bg-white border-2 border-dark-100 rounded-xl p-4 sticky top-20 z-20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-dark-600" />
              <span className={`text-lg font-bold ${timeRemaining < 300 ? 'text-red-600' : 'text-dark-900'}`}>
                {formatTime(timeRemaining)}
              </span>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="text-sm text-dark-600">
                Question {currentQuestion + 1} of {testData.questions.length}
              </span>
            </div>
          </div>

          <button
            onClick={handleSubmitTest}
            className="btn-primary"
          >
            Submit Test
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Question Area */}
        <div className="lg:col-span-3 space-y-6">
          <div className="card p-8">
            <div className="mb-6">
              <div className="text-sm text-dark-600 mb-2">
                Question {currentQuestion + 1}
              </div>
              <h3 className="text-xl font-display font-semibold text-dark-900 mb-4">
                {question.question}
              </h3>
            </div>

            {question.type === 'multiple-choice' ? (
              <div className="space-y-3">
                {question.options.map((option, index) => (
                  <label
                    key={index}
                    className="flex items-center gap-3 p-4 border-2 border-dark-200 rounded-lg cursor-pointer hover:border-dark-900 transition-colors"
                  >
                    <input
                      type="radio"
                      name={`question-${question.id}`}
                      checked={answers[question.id] === index}
                      onChange={() => handleAnswerChange(question.id, index)}
                      className="w-5 h-5"
                    />
                    <span className="text-dark-900">{option}</span>
                  </label>
                ))}
              </div>
            ) : (
              <textarea
                value={answers[question.id] || ''}
                onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                className="input-field w-full h-48 resize-none"
                placeholder="Type your answer here..."
              />
            )}

            <div className="flex justify-between mt-8">
              <button
                onClick={handlePreviousQuestion}
                disabled={currentQuestion === 0}
                className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <button
                onClick={handleNextQuestion}
                disabled={currentQuestion === testData.questions.length - 1}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        </div>

        {/* Proctoring Sidebar - Same as before */}
        <div className="space-y-4">
          {/* Camera Feed */}
          <div className="card p-4">
            <h4 className="font-semibold text-dark-900 mb-3 flex items-center gap-2">
              <Camera className="w-4 h-4" />
              Camera Monitor
            </h4>
            <div className="aspect-video bg-dark-900 rounded-lg overflow-hidden mb-3">
              {cameraEnabled ? (
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <CameraOff className="w-8 h-8 text-dark-400" />
                </div>
              )}
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-dark-600">Face Detected:</span>
                {proctoringData.faceDetected ? (
                  <CheckCircle className="w-4 h-4 text-green-600" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-600" />
                )}
              </div>
              <div className="flex items-center justify-between">
                <span className="text-dark-600">Multiple Faces:</span>
                {proctoringData.multipleFaces ? (
                  <XCircle className="w-4 h-4 text-red-600" />
                ) : (
                  <CheckCircle className="w-4 h-4 text-green-600" />
                )}
              </div>
            </div>
          </div>

          {/* Risk Score */}
          <div className="card p-4">
            <h4 className="font-semibold text-dark-900 mb-3">Risk Score</h4>
            <div className="mb-2">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-dark-600">Current Risk</span>
                <span className={`font-bold ${riskScore > 50 ? 'text-red-600' : 'text-green-600'}`}>
                  {riskScore}%
                </span>
              </div>
              <div className="w-full bg-dark-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all ${
                    riskScore > 50 ? 'bg-red-600' : 'bg-green-600'
                  }`}
                  style={{ width: `${riskScore}%` }}
                />
              </div>
            </div>
            <p className="text-xs text-dark-600">
              {violations.length} violation(s) detected
            </p>
          </div>

          {/* Navigation Grid */}
          <div className="card p-4">
            <h4 className="font-semibold text-dark-900 mb-3">Questions</h4>
            <div className="grid grid-cols-5 gap-2">
              {testData.questions.map((q, index) => (
                <button
                  key={q.id}
                  onClick={() => setCurrentQuestion(index)}
                  className={`
                    aspect-square rounded-lg text-sm font-medium transition-all
                    ${currentQuestion === index
                      ? 'bg-dark-900 text-white'
                      : answers[q.id] !== undefined
                        ? 'bg-green-100 text-green-800'
                        : 'bg-dark-100 text-dark-600 hover:bg-dark-200'
                    }
                  `}
                >
                  {index + 1}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Tab Switch Warning Modal */}
      {showWarning && (
        <div className="fixed inset-0 bg-dark-900/90 backdrop-blur-sm flex items-center justify-center p-4 z-[100]">
          <div className="card max-w-md w-full p-8 border-red-500 border-4">
            <div className="text-center">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <AlertTriangle className="w-8 h-8 text-red-600" />
              </div>
              <h2 className="text-2xl font-display font-bold text-red-600 mb-2">
                Warning: Tab Switch Detected!
              </h2>
              <p className="text-dark-700 mb-6">
                You have switched tabs {proctoringData.tabSwitches} time(s). 
                Switching tabs more than <strong>{MAX_SWITCH_LIMIT}</strong> times will result in automatic submission.
              </p>
              <button
                onClick={() => setShowWarning(false)}
                className="btn-danger w-full py-4 text-lg font-bold"
              >
                I Understand, Return to Test
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TakeTest;