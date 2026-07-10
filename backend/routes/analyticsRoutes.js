const express = require('express');
const router = express.Router();
const { updateAndSyncQuery } = require('../services/popularityService');
const { getCorrectedSpelling } = require('../services/geminiService');

// @route   POST /api/analytics/autosuggest-click
// @desc    Logs a click and updates its popularity in real-time (with AI correction).
// @access  Public
router.post('/autosuggest-click', async (req, res) => {
    const { query } = req.body;

    if (!query) {
        return res.status(400).json({ msg: 'Query is required.' });
    }

    try {
        // Use AI to correct the query before storing
        const correctedQuery = await getCorrectedSpelling(query.trim());
        const queryToStore = correctedQuery && correctedQuery !== query.trim() ? correctedQuery : query.trim();
        
        // Don't wait for the update to complete. Respond to the user immediately.
        updateAndSyncQuery(queryToStore);
        console.log(`[Analytics] Autosuggest click - stored enhanced query: "${queryToStore}" (original: "${query.trim()}")`);

        res.status(202).json({ msg: 'Click event accepted for processing.' });
    } catch (error) {
        console.error('[Analytics] Autosuggest click error:', error);
        // Fallback to original query if AI correction fails
        updateAndSyncQuery(query.trim());
        res.status(202).json({ msg: 'Click event accepted for processing.' });
    }
});

// @route   POST /api/analytics/log-search
// @desc    Logs a search and updates its popularity in real-time (with AI correction).
// @access  Public
router.post('/log-search', async (req, res) => {
    const { query } = req.body;

    if (!query) {
        return res.status(400).json({ msg: 'Query is required.' });
    }

    try {
        // Use AI to correct the query before storing
        const correctedQuery = await getCorrectedSpelling(query.trim());
        const queryToStore = correctedQuery && correctedQuery !== query.trim() ? correctedQuery : query.trim();
        
        // Don't wait for the update to complete. Respond to the user immediately.
        updateAndSyncQuery(queryToStore);
        console.log(`[Analytics] Log search - stored enhanced query: "${queryToStore}" (original: "${query.trim()}")`);

        res.status(202).json({ msg: 'Search event accepted for processing.' });
    } catch (error) {
        console.error('[Analytics] Log search error:', error);
        // Fallback to original query if AI correction fails
        updateAndSyncQuery(query.trim());
        res.status(202).json({ msg: 'Search event accepted for processing.' });
    }
});

module.exports = router;
