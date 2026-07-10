const express = require('express');
const { strictRelevanceSearch } = require('../controllers/strictRelevanceController');

const router = express.Router();

// 🎯 Strict relevance search endpoint - NO IRRELEVANT PRODUCTS
router.get('/strict-search', strictRelevanceSearch);

module.exports = router;
