const express = require('express');
const router = express.Router();
const { gbertRecommend, gbertRerank, personalizeScore, personalizeScoreBatch, personalizeTrain } = require('../controllers/mlController');

router.post('/gbert', gbertRecommend);
router.post('/gbert/rerank', gbertRerank);
router.post('/personalize', personalizeScore);
router.post('/personalize/score-batch', personalizeScoreBatch);
router.post('/personalize/train', personalizeTrain);

module.exports = router;
