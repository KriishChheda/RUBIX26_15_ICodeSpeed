// Mock data for development without backend

export const mockTests = [
  {
    id: '1',
    title: 'Introduction to Machine Learning',
    description: 'Comprehensive test covering basic ML concepts, algorithms, and applications.',
    duration: 60,
    totalQuestions: 30,
    totalMarks: 100,
    status: 'published',
    createdAt: '2024-01-15T10:00:00Z',
    startDate: '2024-01-20T09:00:00Z',
    endDate: '2024-01-20T18:00:00Z',
  },
  {
    id: '2',
    title: 'Data Structures and Algorithms',
    description: 'Test your knowledge on arrays, linked lists, trees, graphs, and sorting algorithms.',
    duration: 90,
    totalQuestions: 40,
    totalMarks: 150,
    status: 'published',
    createdAt: '2024-01-16T14:00:00Z',
    startDate: '2024-01-22T10:00:00Z',
    endDate: '2024-01-22T20:00:00Z',
  },
  {
    id: '3',
    title: 'Web Development Fundamentals',
    description: 'HTML, CSS, JavaScript, and modern web development practices.',
    duration: 45,
    totalQuestions: 25,
    totalMarks: 75,
    status: 'draft',
    createdAt: '2024-01-17T11:00:00Z',
    startDate: null,
    endDate: null,
  },
];

export const mockStudentTests = [
  {
    id: '1',
    title: 'Introduction to Machine Learning',
    duration: 60,
    totalQuestions: 30,
    totalMarks: 100,
    status: 'available',
    startDate: '2024-01-20T09:00:00Z',
    endDate: '2024-01-20T18:00:00Z',
    attempts: 0,
    maxAttempts: 1,
  },
  {
    id: '2',
    title: 'Data Structures and Algorithms',
    duration: 90,
    totalQuestions: 40,
    totalMarks: 150,
    status: 'completed',
    startDate: '2024-01-22T10:00:00Z',
    endDate: '2024-01-22T20:00:00Z',
    attempts: 1,
    maxAttempts: 1,
    score: 125,
    completedAt: '2024-01-22T11:15:00Z',
  },
];

export const mockResults = [
  {
    id: '1',
    studentName: 'John Doe',
    studentEmail: 'john.doe@example.com',
    testTitle: 'Introduction to Machine Learning',
    score: 85,
    totalMarks: 100,
    percentage: 85,
    submittedAt: '2024-01-20T10:45:00Z',
    violations: 2,
    riskScore: 35,
  },
  {
    id: '2',
    studentName: 'Jane Smith',
    studentEmail: 'jane.smith@example.com',
    testTitle: 'Introduction to Machine Learning',
    score: 92,
    totalMarks: 100,
    percentage: 92,
    submittedAt: '2024-01-20T10:30:00Z',
    violations: 0,
    riskScore: 5,
  },
  {
    id: '3',
    studentName: 'Mike Johnson',
    studentEmail: 'mike.j@example.com',
    testTitle: 'Data Structures and Algorithms',
    score: 125,
    totalMarks: 150,
    percentage: 83.3,
    submittedAt: '2024-01-22T11:15:00Z',
    violations: 1,
    riskScore: 20,
  },
];

export const mockQuestions = [
  {
    id: 'q1',
    question: 'What is supervised learning?',
    type: 'multiple-choice',
    options: [
      'Learning with labeled data',
      'Learning without labels',
      'Reinforcement-based learning',
      'Clustering technique'
    ],
    correctAnswer: 0,
    marks: 4,
  },
  {
    id: 'q2',
    question: 'Explain the difference between classification and regression.',
    type: 'text',
    marks: 10,
  },
];

export const mockViolations = [
  {
    id: '1',
    type: 'multiple-faces',
    timestamp: '2024-01-20T10:15:23Z',
    severity: 'high',
    description: 'Multiple faces detected in frame',
  },
  {
    id: '2',
    type: 'tab-switch',
    timestamp: '2024-01-20T10:25:45Z',
    severity: 'medium',
    description: 'Browser tab switched',
  },
  {
    id: '3',
    type: 'face-not-visible',
    timestamp: '2024-01-20T10:35:12Z',
    severity: 'high',
    description: 'Face not visible in camera',
  },
];

// Mock API calls with delays to simulate network requests
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export const mockAPI = {
  getTests: async () => {
    await delay(500);
    return { data: mockTests };
  },
  
  getStudentTests: async () => {
    await delay(500);
    return { data: mockStudentTests };
  },
  
  getResults: async () => {
    await delay(500);
    return { data: mockResults };
  },
  
  getTestById: async (testId) => {
    await delay(300);
    const test = mockTests.find(t => t.id === testId);
    return { data: test };
  },
  
  createTest: async (testData) => {
    await delay(500);
    const newTest = {
      id: Date.now().toString(),
      ...testData,
      createdAt: new Date().toISOString(),
      status: 'draft',
    };
    mockTests.push(newTest);
    return { data: newTest };
  },
  
  submitTest: async (testId, answers) => {
    await delay(1000);
    return {
      data: {
        score: Math.floor(Math.random() * 30) + 70,
        totalMarks: 100,
        submittedAt: new Date().toISOString(),
      }
    };
  },
};