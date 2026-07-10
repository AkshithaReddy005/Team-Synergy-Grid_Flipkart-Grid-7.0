const cron = require('node-cron');
const axios = require('axios');
const path = require('path');
const fs = require('fs').promises;

class RetrainScheduler {
  constructor() {
    this.mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8001';
    this.catalogPath = path.resolve(__dirname, '../../Dataset_Final_TeamSynergyGrid.csv');
  }

  async triggerPersonalizationRetrain() {
    try {
      console.log('[RetrainScheduler] Starting personalization retraining...');
      const { data } = await axios.post(`${this.mlServiceUrl}/personalize/train`, {
        dataset_path: this.catalogPath,
      });
      console.log('[RetrainScheduler] Personalization retrain completed:', data);
    } catch (error) {
      console.error('[RetrainScheduler] Personalization retrain failed:', error?.response?.data || error.message);
    }
  }

  async triggerGbertRetrain() {
    try {
      console.log('[RetrainScheduler] Starting G-BERT retraining...');
      const sessionsPath = path.resolve(__dirname, '../../../ml-service/data/synthetic_sessions.csv');
      const modelOutPath = path.resolve(__dirname, '../../../ml-service/models/gbert');
      await fs.mkdir(path.dirname(modelOutPath), { recursive: true });
      // Trigger training script inside ML service container (if using Docker) or locally
      const { data } = await axios.post(`${this.mlServiceUrl}/retrain/gbert`, {
        sessions_path: sessionsPath,
        catalog_path: this.catalogPath,
        output_path: modelOutPath,
        epochs: 2,
      });
      console.log('[RetrainScheduler] G-BERT retrain completed:', data);
    } catch (error) {
      console.error('[RetrainScheduler] G-BERT retrain failed:', error?.response?.data || error.message);
    }
  }

  start() {
    // Personalization: every Sunday at 2 AM
    cron.schedule('0 2 * * 0', () => {
      this.triggerPersonalizationRetrain();
    });

    // G-BERT: every Monday at 3 AM
    cron.schedule('0 3 * * 1', () => {
      this.triggerGbertRetrain();
    });

    console.log('[RetrainScheduler] Schedulers started');
  }
}

module.exports = new RetrainScheduler();
