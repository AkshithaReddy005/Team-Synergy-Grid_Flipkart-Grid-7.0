const cron = require('node-cron');
const { spawn } = require('child_process');
const path = require('path');

const runDataSync = () => {
  console.log('Running scheduled data synchronization...');
  const child = spawn('node', [path.join(__dirname, '..', 'indexData.js')], { stdio: 'inherit' });

  child.on('close', (code) => {
    if (code !== 0) {
      console.error(`Data synchronization process exited with code ${code}`);
    } else {
      console.log('Data synchronization completed successfully.');
    }
  });
};

const startSyncService = () => {
  // Schedule the task to run every hour
  cron.schedule('0 * * * *', () => {
    runDataSync();
  });

  console.log('Real-time data synchronization service has been scheduled. It will run every hour.');

  // Kicking off initial data sync in the background
  console.log('Kicking off initial data sync in the background...');
  runDataSync();
};

module.exports = { startSyncService };
