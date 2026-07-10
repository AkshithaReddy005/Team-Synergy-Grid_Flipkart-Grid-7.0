const express = require('express');
const router = express.Router();
const searchController = require('../controllers/searchController');
const enhancedSearchController = require('../controllers/enhancedSearchController');
const { updateAndSyncQuery } = require('../services/popularityService');
const { getCorrectedSpelling } = require('../services/geminiService');

// Basic search routes
router.get('/autosuggest', searchController.getAutosuggestions);
router.get('/products', searchController.searchProducts);

// Enhanced search routes with Gemini AI
router.get('/enhanced', enhancedSearchController.enhancedSearch);
router.get('/recommendations', enhancedSearchController.getSmartRecommendations);
router.get('/insights', enhancedSearchController.getQueryInsights);

// Log search queries for analytics (with AI correction)
router.post('/log', async (req, res) => {
  try {
    const { query } = req.body;
    if (query && query.trim()) {
      // Use AI to correct the query before storing
      const correctedQuery = await getCorrectedSpelling(query.trim());
      const queryToStore = correctedQuery && correctedQuery !== query.trim() ? correctedQuery : query.trim();
      
      await updateAndSyncQuery(queryToStore);
      console.log(`[Search Log] Logged enhanced query: "${queryToStore}" (original: "${query.trim()}")`);
      res.json({ success: true, message: 'Query logged successfully' });
    } else {
      res.status(400).json({ error: 'Query is required' });
    }
  } catch (error) {
    console.error('[Search Log] Error:', error);
    res.status(500).json({ error: 'Failed to log query' });
  }
});

module.exports = router;
