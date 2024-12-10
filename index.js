import express from 'express';
import { PythonShell } from 'python-shell';

const app = express();

const pythonPath = '/'; 

app.use(express.json());

const inputData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
let output = '';

app.post('/', (req, res) => {
  console.log('Get POST*\n');
});

let pyshell = new PythonShell('dist_data_info.py', {
  pythonPath: pythonPath,
  args: [JSON.stringify(inputData)], 
});

pyshell.on('message', function (message) {
  output += message; 
});

pyshell.end(function (err) {
  if (err) {
    console.error('Error executing Python-script:', err.message);
    return;
  }
  console.log(output); 
});

app.listen(3000, () => {
  console.log('localhost \n');
});