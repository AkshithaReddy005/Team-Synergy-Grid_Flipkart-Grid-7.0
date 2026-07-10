class ApiKeyManager {
  constructor() {
    // Multiple API keys for rotation
    this.apiKeys = [
      process.env.GEMINI_API_KEY_1 || process.env.GEMINI_API_KEY,
      process.env.GEMINI_API_KEY_2,
      process.env.GEMINI_API_KEY_3,
      process.env.GEMINI_API_KEY_4
    ].filter(key => key && key.trim() !== ''); // Remove empty/null keys

    this.currentKeyIndex = 0;
    this.keyUsageCount = new Map(); // Track usage per key
    this.keyErrorCount = new Map(); // Track errors per key
    this.maxUsagePerKey = 50; // Max requests per key before rotation
    this.maxErrorsPerKey = 3; // Max errors before marking key as bad

    // Initialize usage counters
    this.apiKeys.forEach((key, index) => {
      this.keyUsageCount.set(index, 0);
      this.keyErrorCount.set(index, 0);
    });

    console.log(`🔑 API Key Manager initialized with ${this.apiKeys.length} keys`);
  }

  // Get the next available API key
  getNextApiKey() {
    if (this.apiKeys.length === 0) {
      throw new Error('No valid API keys available');
    }

    // Find a key that hasn't exceeded usage or error limits
    let attempts = 0;
    while (attempts < this.apiKeys.length) {
      const currentUsage = this.keyUsageCount.get(this.currentKeyIndex) || 0;
      const currentErrors = this.keyErrorCount.get(this.currentKeyIndex) || 0;

      // Check if current key is still usable
      if (currentUsage < this.maxUsagePerKey && currentErrors < this.maxErrorsPerKey) {
        const selectedKey = this.apiKeys[this.currentKeyIndex];
        this.keyUsageCount.set(this.currentKeyIndex, currentUsage + 1);
        
        console.log(`🔑 Using API key ${this.currentKeyIndex + 1}/${this.apiKeys.length} (Usage: ${currentUsage + 1}/${this.maxUsagePerKey})`);
        return selectedKey;
      }

      // Move to next key
      this.currentKeyIndex = (this.currentKeyIndex + 1) % this.apiKeys.length;
      attempts++;
    }

    // If all keys are exhausted, reset counters and start over
    console.log('🔄 All API keys exhausted, resetting counters...');
    this.resetCounters();
    return this.getNextApiKey();
  }

  // Report an error for the current key
  reportError(apiKey) {
    const keyIndex = this.apiKeys.indexOf(apiKey);
    if (keyIndex !== -1) {
      const currentErrors = this.keyErrorCount.get(keyIndex) || 0;
      this.keyErrorCount.set(keyIndex, currentErrors + 1);
      
      console.log(`❌ Error reported for API key ${keyIndex + 1} (Errors: ${currentErrors + 1}/${this.maxErrorsPerKey})`);
      
      // If this key has too many errors, rotate to next
      if (currentErrors + 1 >= this.maxErrorsPerKey) {
        console.log(`🚫 API key ${keyIndex + 1} marked as problematic, rotating...`);
        this.currentKeyIndex = (keyIndex + 1) % this.apiKeys.length;
      }
    }
  }

  // Reset all counters (useful when all keys are exhausted)
  resetCounters() {
    this.apiKeys.forEach((key, index) => {
      this.keyUsageCount.set(index, 0);
      this.keyErrorCount.set(index, 0);
    });
    this.currentKeyIndex = 0;
    console.log('🔄 API key usage counters reset');
  }

  // Get statistics about key usage
  getStats() {
    const stats = this.apiKeys.map((key, index) => ({
      keyIndex: index + 1,
      usage: this.keyUsageCount.get(index) || 0,
      errors: this.keyErrorCount.get(index) || 0,
      available: (this.keyUsageCount.get(index) || 0) < this.maxUsagePerKey && 
                 (this.keyErrorCount.get(index) || 0) < this.maxErrorsPerKey
    }));

    return {
      totalKeys: this.apiKeys.length,
      currentKey: this.currentKeyIndex + 1,
      keys: stats
    };
  }

  // Force rotation to next key
  forceRotate() {
    this.currentKeyIndex = (this.currentKeyIndex + 1) % this.apiKeys.length;
    console.log(`🔄 Forced rotation to API key ${this.currentKeyIndex + 1}/${this.apiKeys.length}`);
  }
}

// Singleton instance
const apiKeyManager = new ApiKeyManager();

module.exports = apiKeyManager;
