const client = require('redis').createClient();
const fs = require('fs').promises;
const path = require('path');

class MonitoringService {
  constructor() {
    this.metrics = {
      mlLatency: [],
      cacheHits: 0,
      cacheMisses: 0,
      errors: 0,
    };
    this.metricsFile = path.resolve(__dirname, '../../logs/metrics.json');
    this.initRedis();
  }

  async initRedis() {
    try {
      await client.connect();
      console.log('[MonitoringService] Redis connected');
    } catch (err) {
      console.warn('[MonitoringService] Redis not available, using in-memory cache');
    }
  }

  async cacheGet(key) {
    try {
      const cached = await client.get(key);
      if (cached) {
        this.metrics.cacheHits++;
        return JSON.parse(cached);
      }
      this.metrics.cacheMisses++;
      return null;
    } catch {
      this.metrics.cacheMisses++;
      return null;
    }
  }

  async cacheSet(key, value, ttlSeconds = 300) {
    try {
      await client.setEx(key, ttlSeconds, JSON.stringify(value));
    } catch {
      // ignore Redis errors
    }
  }

  recordLatency(service, ms) {
    this.metrics.mlLatency.push({ service, ms, ts: Date.now() });
    if (this.metrics.mlLatency.length > 1000) this.metrics.mlLatency.shift();
  }

  recordError() {
    this.metrics.errors++;
  }

  async getMetrics() {
    const recent = this.metrics.mlLatency.slice(-100);
    const avgLatency = recent.length ? recent.reduce((s, e) => s + e.ms, 0) / recent.length : 0;
    return {
      avgLatencyMs: Math.round(avgLatency),
      cacheHitRate: this.metrics.cacheHits / (this.metrics.cacheHits + this.metrics.cacheMisses) || 0,
      errors: this.metrics.errors,
      totalRequests: this.metrics.cacheHits + this.metrics.cacheMisses,
    };
  }

  async flushMetrics() {
    try {
      await fs.mkdir(path.dirname(this.metricsFile), { recursive: true });
      await fs.writeFile(this.metricsFile, JSON.stringify(await this.getMetrics(), null, 2));
    } catch (err) {
      console.error('[MonitoringService] Failed to write metrics:', err);
    }
  }
}

module.exports = new MonitoringService();
