require('dotenv').config();
const express = require('express');
const connectDB = require('./config/db');
const cors = require('cors');
const indexData = require('./indexData'); // Import the indexing function
const { startSyncService } = require('./services/syncService');
const retrainScheduler = require('./services/retrainScheduler');

const app = express();

// Init Middleware
app.use(cors());
app.use(express.json());

// Define Routes
const searchRoutes = require('./routes/searchRoutes');
const analyticsRoutes = require('./routes/analyticsRoutes');
const mlRoutes = require('./routes/mlRoutes');
const userHistoryRoutes = require('./routes/userHistoryRoutes');

app.use('/api/search', searchRoutes);
app.use('/api/srp', require('./routes/srpDynamicRoutes'));
app.use('/api/analytics', analyticsRoutes);
app.use('/api/test', require('./routes/testRoutes')); 
app.use('/api/ml', mlRoutes);
app.use('/api/userHistory', userHistoryRoutes);
app.get('/', (req, res) => res.send('API Running'));

const PORT = process.env.PORT || 5001;

const startServer = async () => {
  try {
    // Connect to Database
    await connectDB();
    
    // Run the initial data indexing on startup (commented out for now)
    // console.log('Kicking off initial data sync...');
    // await indexData();
    
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
      // Schedule the recurring sync service after the initial setup (disabled for now)
      // console.log('Scheduling recurring data synchronization...');
      // startSyncService();
      // Start retraining scheduler
      retrainScheduler.start();
    });

  } catch (error) {
    console.error('Failed to start the server:', error);
    process.exit(1);
  }
};

startServer();
