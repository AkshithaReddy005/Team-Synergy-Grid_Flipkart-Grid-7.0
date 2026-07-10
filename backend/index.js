require('dotenv').config();
const express = require('express');
const connectDB = require('./config/db');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const indexData = require('./indexData');
const { startSyncService } = require('./services/syncService');
const retrainScheduler = require('./services/retrainScheduler');

const app = express();

// --- Enterprise-Grade Middleware ---
// 1. Security Headers
app.use(helmet());

// 2. Cross-Origin Resource Sharing
app.use(cors());

// 3. Request Payload Parsing
app.use(express.json());

// 4. Professional Request Logging
app.use(morgan('combined'));

// 5. API Rate Limiting (Protects against DDoS/Brute Force)
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 200, // Limit each IP to 200 requests per window
  message: { success: false, message: 'Too many requests from this IP, please try again after 15 minutes' }
});
app.use('/api/', apiLimiter);

// --- Health Check Endpoint ---
app.get('/health', (req, res) => {
  res.status(200).json({ success: true, message: 'API is healthy and running' });
});

// --- Routes ---
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
app.get('/', (req, res) => res.send('Flipkart Grid 7.0 API Running'));

// --- Global Error Handler ---
app.use((err, req, res, next) => {
  console.error('[Global Error]', err.stack);
  res.status(500).json({
    success: false,
    message: 'Internal Server Error',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

const PORT = process.env.PORT || 5001;
let server;

const startServer = async () => {
  try {
    // Connect to Database
    await connectDB();
    
    server = app.listen(PORT, () => {
      console.log(`[INFO] Server is running on port ${PORT}`);
      // retrainScheduler.start(); // Uncomment when ml-service is ready
    });
  } catch (error) {
    console.error('[ERROR] Failed to start the server:', error);
    process.exit(1);
  }
};

startServer();

// --- Graceful Shutdown ---
process.on('SIGTERM', () => {
  console.log('[INFO] SIGTERM signal received: closing HTTP server');
  if (server) {
    server.close(() => {
      console.log('[INFO] HTTP server closed');
      // Here you would also disconnect mongoose/elasticsearch if needed
      process.exit(0);
    });
  }
});
