class CacheService {
  constructor() {
    // In-memory cache for fast lookups
    this.translationCache = new Map();
    this.typoCache = new Map();
    this.categoryCache = new Map();
    this.searchResultsCache = new Map();
    
    // Cache settings
    this.maxCacheSize = 1000;
    this.cacheExpiry = 30 * 60 * 1000; // 30 minutes
    
    console.log('🚀 Cache service initialized');
  }

  // Generic cache methods
  set(cache, key, value) {
    // Implement LRU eviction if cache is full
    if (cache.size >= this.maxCacheSize) {
      const firstKey = cache.keys().next().value;
      cache.delete(firstKey);
    }
    
    cache.set(key, {
      value,
      timestamp: Date.now()
    });
  }

  get(cache, key) {
    const cached = cache.get(key);
    if (!cached) return null;
    
    // Check if expired
    if (Date.now() - cached.timestamp > this.cacheExpiry) {
      cache.delete(key);
      return null;
    }
    
    return cached.value;
  }

  // Translation cache
  getCachedTranslation(query) {
    return this.get(this.translationCache, query.toLowerCase());
  }

  setCachedTranslation(query, translation) {
    this.set(this.translationCache, query.toLowerCase(), translation);
  }

  // Typo correction cache
  getCachedTypoCorrection(query) {
    return this.get(this.typoCache, query.toLowerCase());
  }

  setCachedTypoCorrection(query, correction) {
    this.set(this.typoCache, query.toLowerCase(), correction);
  }

  // Category cache
  getCachedCategories(query) {
    return this.get(this.categoryCache, query.toLowerCase());
  }

  setCachedCategories(query, categories) {
    this.set(this.categoryCache, query.toLowerCase(), categories);
  }

  // Search results cache
  getCachedSearchResults(cacheKey) {
    return this.get(this.searchResultsCache, cacheKey);
  }

  setCachedSearchResults(cacheKey, results) {
    this.set(this.searchResultsCache, cacheKey, results);
  }

  // Generate cache key for search results
  generateSearchCacheKey(query, filters = {}) {
    const key = {
      query: query.toLowerCase(),
      ...filters
    };
    return JSON.stringify(key);
  }

  // Clear expired entries
  clearExpired() {
    const now = Date.now();
    const caches = [this.translationCache, this.typoCache, this.categoryCache, this.searchResultsCache];
    
    caches.forEach(cache => {
      for (const [key, value] of cache.entries()) {
        if (now - value.timestamp > this.cacheExpiry) {
          cache.delete(key);
        }
      }
    });
  }

  // Get cache statistics
  getStats() {
    return {
      translation: this.translationCache.size,
      typo: this.typoCache.size,
      category: this.categoryCache.size,
      searchResults: this.searchResultsCache.size,
      total: this.translationCache.size + this.typoCache.size + 
             this.categoryCache.size + this.searchResultsCache.size
    };
  }

  // Clear all caches
  clearAll() {
    this.translationCache.clear();
    this.typoCache.clear();
    this.categoryCache.clear();
    this.searchResultsCache.clear();
    console.log('🧹 All caches cleared');
  }
}

// Singleton instance
const cacheService = new CacheService();

// Clean expired entries every 10 minutes
setInterval(() => {
  cacheService.clearExpired();
}, 10 * 60 * 1000);

module.exports = cacheService;
