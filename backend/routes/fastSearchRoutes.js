const express = require('express');
const { fastSearch, getPerformanceStats } = require('../controllers/fastSearchController');

const router = express.Router();

// ⚡ Ultra-fast search endpoint with parallel processing and caching
router.get('/fast-search', fastSearch);

// 📊 Performance monitoring endpoint
router.get('/performance', getPerformanceStats);

module.exports = router;
