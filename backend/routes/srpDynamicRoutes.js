const express = require('express');
const router = express.Router();
const { dynamicSearch } = require('../controllers/srpDynamicController');

router.get('/search', dynamicSearch);

module.exports = router;
