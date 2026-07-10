const crypto = require('crypto');

class ABTestService {
  constructor() {
    this.configs = {
      searchRerank: {
        control: { gbertWeight: 0, personalWeight: 0 },
        variantA: { gbertWeight: 0.3, personalWeight: 0.2 },
        variantB: { gbertWeight: 0.5, personalWeight: 0.3 },
      },
    };
  }

  hashUserId(user_id) {
    return crypto.createHash('md5').update(user_id).digest('hex');
  }

  getVariant(user_id, experiment) {
    const hash = this.hashUserId(user_id);
    const bucket = parseInt(hash.substring(0, 8), 16) % 100;
    if (bucket < 34) return 'control';
    if (bucket < 67) return 'variantA';
    return 'variantB';
  }

  getWeights(user_id) {
    const variant = this.getVariant(user_id, 'searchRerank');
    return this.configs.searchRerank[variant];
  }

  logExposure(user_id, experiment, variant) {
    console.log(`[ABTest] User ${user_id} exposed to ${experiment}:${variant}`);
  }
}

module.exports = new ABTestService();
