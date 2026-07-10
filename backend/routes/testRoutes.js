const express = require('express');
const router = express.Router();
const { testSearch } = require('../controllers/testSearchController');

router.get('/search', testSearch);

module.exports = router;
