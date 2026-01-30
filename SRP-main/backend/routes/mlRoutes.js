const express = require('express');
const router = express.Router();
const { gbertRecommend, personalizeScore, personalizeScoreBatch, personalizeTrain } = require('../controllers/mlController');

router.post('/gbert', gbertRecommend);
router.post('/personalize', personalizeScore);
router.post('/personalize/score-batch', personalizeScoreBatch);
router.post('/personalize/train', personalizeTrain);

module.exports = router;
