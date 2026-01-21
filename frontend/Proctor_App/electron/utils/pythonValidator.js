/**
 * Python Environment Validator
 * Utility to test and verify Python environment setup for proctoring
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

class PythonValidator {
  /**
   * Test if Python executable works and check version
   * @param {string} pythonPath - Path to Python executable
   * @returns {Promise<Object>} Result with version info
   */
  static async testPythonExecutable(pythonPath) {
    return new Promise((resolve) => {
      const process = spawn(pythonPath, ['--version']);
      let output = '';
      let error = '';

      process.stdout.on('data', (data) => {
        output += data.toString();
      });

      process.stderr.on('data', (data) => {
        error += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0 || output.includes('Python') || error.includes('Python')) {
          const versionMatch = (output + error).match(/Python (\d+\.\d+\.\d+)/);
          resolve({
            success: true,
            version: versionMatch ? versionMatch[1] : 'unknown',
            output: output + error
          });
        } else {
          resolve({
            success: false,
            error: 'Failed to execute Python',
            output: output + error
          });
        }
      });

      process.on('error', (err) => {
        resolve({
          success: false,
          error: err.message
        });
      });
    });
  }

  /**
   * Check if required Python packages are installed
   * @param {string} pythonPath - Path to Python executable
   * @param {Array<string>} packages - Package names to check
   * @returns {Promise<Object>} Result with package status
   */
  static async checkPackages(pythonPath, packages = ['opencv-python', 'mediapipe', 'deepface']) {
    return new Promise((resolve) => {
      const checkScript = `
import sys
import importlib.metadata

packages = ${JSON.stringify(packages)}
results = {}

for pkg in packages:
    try:
        # Try different package name formats
        pkg_check = pkg.replace('-', '_')
        version = importlib.metadata.version(pkg_check)
        results[pkg] = {'installed': True, 'version': version}
    except:
        try:
            version = importlib.metadata.version(pkg)
            results[pkg] = {'installed': True, 'version': version}
        except:
            results[pkg] = {'installed': False, 'version': None}

import json
print(json.dumps(results))
`;

      const process = spawn(pythonPath, ['-c', checkScript]);
      let output = '';
      let error = '';

      process.stdout.on('data', (data) => {
        output += data.toString();
      });

      process.stderr.on('data', (data) => {
        error += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0 && output.trim()) {
          try {
            const results = JSON.parse(output.trim());
            const allInstalled = Object.values(results).every(pkg => pkg.installed);
            
            resolve({
              success: true,
              allInstalled,
              packages: results
            });
          } catch (err) {
            resolve({
              success: false,
              error: 'Failed to parse package check results',
              output,
              stderr: error
            });
          }
        } else {
          resolve({
            success: false,
            error: 'Failed to check packages',
            output,
            stderr: error
          });
        }
      });

      process.on('error', (err) => {
        resolve({
          success: false,
          error: err.message
        });
      });
    });
  }

  /**
   * Validate entire Python environment setup
   * @param {string} pythonPath - Path to Python executable
   * @returns {Promise<Object>} Complete validation result
   */
  static async validateEnvironment(pythonPath) {
    const results = {
      pythonPath,
      executable: null,
      packages: null,
      overall: false
    };

    // Test Python executable
    results.executable = await this.testPythonExecutable(pythonPath);
    
    if (!results.executable.success) {
      return results;
    }

    // Check packages
    results.packages = await this.checkPackages(pythonPath);

    // Overall success
    results.overall = results.executable.success && 
                     results.packages.success && 
                     results.packages.allInstalled;

    return results;
  }

  /**
   * Get detailed report string
   * @param {Object} validation - Validation result
   * @returns {string} Formatted report
   */
  static getReport(validation) {
    let report = '\n=== Python Environment Validation ===\n\n';
    
    report += `Python Path: ${validation.pythonPath}\n\n`;

    // Executable check
    report += '1. Python Executable:\n';
    if (validation.executable.success) {
      report += `   ✓ Working (Version: ${validation.executable.version})\n`;
    } else {
      report += `   ✗ Failed: ${validation.executable.error}\n`;
    }
    report += '\n';

    // Package check
    if (validation.packages) {
      report += '2. Required Packages:\n';
      if (validation.packages.success) {
        for (const [pkg, info] of Object.entries(validation.packages.packages)) {
          if (info.installed) {
            report += `   ✓ ${pkg} (${info.version})\n`;
          } else {
            report += `   ✗ ${pkg} - NOT INSTALLED\n`;
          }
        }
      } else {
        report += `   ✗ Failed to check packages: ${validation.packages.error}\n`;
      }
      report += '\n';
    }

    // Overall status
    report += '3. Overall Status:\n';
    if (validation.overall) {
      report += '   ✓ Environment is ready for proctoring\n';
    } else {
      report += '   ✗ Environment setup incomplete\n';
      report += '\n   Action Required:\n';
      if (!validation.executable.success) {
        report += '   - Fix Python executable path\n';
      }
      if (!validation.packages.allInstalled) {
        report += '   - Install missing packages in your venv:\n';
        report += '     cd HD_ML_stuff\n';
        report += '     .venv\\Scripts\\activate (Windows) or source .venv/bin/activate (Linux/Mac)\n';
        report += '     pip install -r requirements.txt\n';
      }
    }

    report += '\n====================================\n';
    return report;
  }
}

module.exports = PythonValidator;
