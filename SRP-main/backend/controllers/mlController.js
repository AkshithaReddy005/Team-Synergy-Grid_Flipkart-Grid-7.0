const axios = require('axios');
const MonitoringService = require('../services/monitoringService');
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8001';

exports.gbertRecommend = async (req, res) => {
  const start = Date.now();
  try {
    const { user_id, history = [], k = 10 } = req.body || {};
    const cacheKey = `gbert:${user_id}:${JSON.stringify(history)}:${k}`;
    const cached = await MonitoringService.cacheGet(cacheKey);
    if (cached) return res.json(cached);

    const { data } = await axios.post(`${ML_SERVICE_URL}/recommend/gbert`, {
      user_id,
      history,
      k,
    });
    await MonitoringService.cacheSet(cacheKey, data, 600);
    MonitoringService.recordLatency('gbert_recommend', Date.now() - start);
    res.json(data);
  } catch (err) {
    MonitoringService.recordError();
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data || 'ML service error' });
  }
};

exports.gbertRerank = async (req, res) => {
  const start = Date.now();
  try {
    const { user_id, history = [], candidate_pids = [] } = req.body || {};
    const cacheKey = `gbert_rerank:${user_id}:${JSON.stringify(candidate_pids)}`;
    const cached = await MonitoringService.cacheGet(cacheKey);
    if (cached) return res.json(cached);

    const { data } = await axios.post(`${ML_SERVICE_URL}/recommend/gbert/rerank`, {
      user_id,
      history,
      candidate_pids,
    });
    await MonitoringService.cacheSet(cacheKey, data, 300);
    MonitoringService.recordLatency('gbert_rerank', Date.now() - start);
    res.json(data);
  } catch (err) {
    MonitoringService.recordError();
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data || 'ML service error' });
  }
};

exports.personalizeScore = async (req, res) => {
  try {
    const { features = [] } = req.body || {};
    const { data } = await axios.post(`${ML_SERVICE_URL}/personalize/score`, {
      features,
    });
    res.json(data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data || 'ML service error' });
  }
};

exports.personalizeScoreBatch = async (req, res) => {
  try {
    const { features_list = [] } = req.body || {};
    const { data } = await axios.post(`${ML_SERVICE_URL}/personalize/score-batch`, {
      features_list,
    });
    res.json(data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data || 'ML service error' });
  }
};

exports.personalizeTrain = async (req, res) => {
  try {
    const { dataset_path } = req.body || {};
    const { data } = await axios.post(`${ML_SERVICE_URL}/personalize/train`, {
      dataset_path,
    });
    res.json(data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data || 'ML service error' });
  }
};
