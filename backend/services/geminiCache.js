const NodeCache = require('node-cache');

class GeminiCache {
  constructor() {
    // Cache for 1 hour (3600 seconds)
    this.cache = new NodeCache({ stdTTL: 3600 });
    this.hitCount = 0;
    this.missCount = 0;
  }

  generateKey(operation, query) {
    return `${operation}:${query.toLowerCase().trim()}`;
  }

  get(operation, query) {
    const key = this.generateKey(operation, query);
    const result = this.cache.get(key);
    
    if (result) {
      this.hitCount++;
      console.log(`💾 Cache HIT for ${operation}: "${query}" (${this.hitCount} hits, ${this.missCount} misses)`);
      return result;
    }
    
    this.missCount++;
    return null;
  }

  set(operation, query, result) {
    const key = this.generateKey(operation, query);
    this.cache.set(key, result);
    console.log(`💾 Cached result for ${operation}: "${query}"`);
  }

  getStats() {
    return {
      hitCount: this.hitCount,
      missCount: this.missCount,
      hitRate: this.hitCount / (this.hitCount + this.missCount) * 100,
      cacheSize: this.cache.keys().length
    };
  }

  clear() {
    this.cache.flushAll();
    this.hitCount = 0;
    this.missCount = 0;
    console.log('🗑️ Cache cleared');
  }
}

// Singleton instance
const geminiCache = new GeminiCache();

module.exports = geminiCache;
