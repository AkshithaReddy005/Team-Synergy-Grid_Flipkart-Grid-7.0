const request = require('supertest');
const app = require('../index');

describe('ML Endpoints', () => {
  it('POST /api/ml/gbert should return recommendations', async () => {
    const res = await request(app)
      .post('/api/ml/gbert')
      .send({
        user_id: 'test_user',
        history: [{ product_id: 'pid1', title: 'iPhone 15' }],
        k: 5,
      });
    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('recommendations');
    expect(Array.isArray(res.body.recommendations)).toBe(true);
  });

  it('POST /api/ml/gbert/rerank should score candidates', async () => {
    const res = await request(app)
      .post('/api/ml/gbert/rerank')
      .send({
        user_id: 'test_user',
        history: [{ product_id: 'pid1', title: 'iPhone 15' }],
        candidate_pids: ['pid1', 'pid2'],
      });
    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('scores');
    expect(Array.isArray(res.body.scores)).toBe(true);
  });

  it('POST /api/ml/personalize should return a score', async () => {
    const res = await request(app)
      .post('/api/ml/personalize')
      .send({
        features: [9.5, 3.2, 0.15, 4.2, 7.1, 1],
      });
    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('score');
    expect(typeof res.body.score).toBe('number');
  });

  it('POST /api/ml/personalize/score-batch should return scores', async () => {
    const res = await request(app)
      .post('/api/ml/personalize/score-batch')
      .send({
        features_list: [
          [9.5, 3.2, 0.15, 4.2, 7.1, 1],
          [8.0, 2.5, 0.10, 4.0, 6.5, 1],
        ],
      });
    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('scores');
    expect(Array.isArray(res.body.scores)).toBe(true);
  });
});
