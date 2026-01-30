const axios = require('axios');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

exports.gbertRecommend = async (req, res) => {
  try {
    const { user_id, history = [], k = 10 } = req.body || {};
    const { data } = await axios.post(`${ML_SERVICE_URL}/recommend/gbert`, {
      user_id,
      history,
      k,
    });
    res.json(data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data || 'ML service error' });
  }
};

exports.gbertRerank = async (req, res) => {
  try {
    const { user_id, history = [], candidate_pids = [] } = req.body || {};
    const { data } = await axios.post(`${ML_SERVICE_URL}/recommend/gbert/rerank`, {
      user_id,
      history,
      candidate_pids,
    });
    res.json(data);
  } catch (err) {
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
