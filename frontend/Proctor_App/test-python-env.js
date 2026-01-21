/**
 * Test Script for Python Environment Validation
 * Run this to verify your Python venv setup
 * 
 * Usage: node test-python-env.js
 */

const path = require('path');
const fs = require('fs');
const PythonValidator = require('./electron/utils/pythonValidator');

// Simulate finding Python path (same logic as camMonitorSpawn.js)
function findPythonPath() {
  const projectRoot = path.join(__dirname, '../../..');
  const platform = process.platform;
  const pythonExe = platform === 'win32' ? 'python.exe' : 'python';
  
  // Check different venv locations in priority order
  const venvPaths = [
    path.join(projectRoot, 'HD_ML_stuff', '.venv', platform === 'win32' ? 'Scripts' : 'bin', pythonExe),
    path.join(projectRoot, 'HD_ML_stuff', 'venv', platform === 'win32' ? 'Scripts' : 'bin', pythonExe),
    path.join(projectRoot, 'HD_ML_stuff', 'env', platform === 'win32' ? 'Scripts' : 'bin', pythonExe),
  ];

  // Check for conda environment
  if (process.env.CONDA_PREFIX) {
    const condaPython = path.join(process.env.CONDA_PREFIX, platform === 'win32' ? 'python.exe' : 'bin/python');
    venvPaths.unshift(condaPython);
  }

  console.log('\nSearching for Python in virtual environments...\n');
  
  for (const venvPath of venvPaths) {
    console.log(`Checking: ${venvPath}`);
    if (fs.existsSync(venvPath)) {
      console.log(`✓ Found Python at: ${venvPath}\n`);
      return venvPath;
    } else {
      console.log(`  Not found`);
    }
  }

  console.log(`\nNo venv found, falling back to system Python: ${pythonExe}\n`);
  return pythonExe;
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║   Python Environment Validator for AI Proctor             ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // Find Python
  const pythonPath = findPythonPath();

  // Validate environment
  console.log('Running validation checks...\n');
  const validation = await PythonValidator.validateEnvironment(pythonPath);

  // Print report
  console.log(PythonValidator.getReport(validation));

  // Exit with appropriate code
  process.exit(validation.overall ? 0 : 1);
}

main().catch(err => {
  console.error('Validation failed with error:', err);
  process.exit(1);
});
