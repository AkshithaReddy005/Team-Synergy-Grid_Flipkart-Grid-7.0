const express = require('express');
const router = express.Router();
const UserHistoryService = require('../services/userHistoryService');

router.post('/event', async (req, res) => {
  const { user_id, session_id, product_id, action, metadata } = req.body;
  if (!user_id || !session_id || !product_id || !action) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  await UserHistoryService.recordEvent({ user_id, session_id, product_id, action, metadata });
  res.json({ success: true });
});

router.get('/:user_id', async (req, res) => {
  const { user_id } = req.params;
  const { limit = 20 } = req.query;
  const history = await UserHistoryService.fetchHistory(user_id, parseInt(limit));
  res.json(history);
});

module.exports = router;
