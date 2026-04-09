// Helper to run backend/app.py using venv Python on Windows.
// Falls back to system python if venv not found.
const { spawn } = require('child_process');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const venvPy = path.join(projectRoot, 'venv', 'Scripts', 'python.exe');
const appFile = path.join(__dirname, 'app.py');

const pythonExecutable = venvPy;

const child = spawn(pythonExecutable, [appFile], {
  stdio: 'inherit'
});

child.on('exit', (code) => {
  if (code !== 0) {
    console.error(`Backend exited with code ${code}`);
  }
});
