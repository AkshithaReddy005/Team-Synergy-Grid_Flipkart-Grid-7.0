const { getCorrectedSpelling, getExpectedCategories, translateToEnglish } = require('./geminiService');
const cacheService = require('./cacheService');

class ParallelGeminiService {
  constructor() {
    console.log('⚡ Parallel Gemini service initialized');
  }

  // Fast parallel processing of all AI tasks
  async processQueryParallel(originalQuery) {
    const startTime = Date.now();
    
    // Check if we can skip some AI processing for simple queries
    const isSimpleEnglishQuery = this.isSimpleEnglishQuery(originalQuery);
    
    try {
      let translatedQuery, correctedQuery, categories;

      if (isSimpleEnglishQuery) {
        // Skip translation for obvious English queries
        console.log('⚡ Skipping translation for simple English query');
        translatedQuery = originalQuery;
        
        // Run typo correction and category detection in parallel
        const [typoResult, categoryResult] = await Promise.allSettled([
          this.getCachedOrFetchTypoCorrection(originalQuery),
          this.getCachedOrFetchCategories(originalQuery)
        ]);

        correctedQuery = typoResult.status === 'fulfilled' ? typoResult.value : originalQuery;
        categories = categoryResult.status === 'fulfilled' ? categoryResult.value : [];
        
      } else {
        // Run all three AI tasks in parallel for non-English queries
        const [translationResult, typoResult, categoryResult] = await Promise.allSettled([
          this.getCachedOrFetchTranslation(originalQuery),
          this.getCachedOrFetchTypoCorrection(originalQuery),
          this.getCachedOrFetchCategories(originalQuery)
        ]);

        translatedQuery = translationResult.status === 'fulfilled' ? translationResult.value : originalQuery;
        correctedQuery = typoResult.status === 'fulfilled' ? typoResult.value : originalQuery;
        categories = categoryResult.status === 'fulfilled' ? categoryResult.value : [];
      }

      const processingTime = Date.now() - startTime;
      console.log(`⚡ Parallel AI processing completed in ${processingTime}ms`);

      return {
        originalQuery,
        translatedQuery,
        correctedQuery: correctedQuery || translatedQuery,
        categories: categories || [],
        processingTime
      };

    } catch (error) {
      console.error('❌ Parallel processing error:', error);
      // Return fallback values
      return {
        originalQuery,
        translatedQuery: originalQuery,
        correctedQuery: originalQuery,
        categories: [],
        processingTime: Date.now() - startTime
      };
    }
  }

  // Check if query is simple English (skip translation)
  isSimpleEnglishQuery(query) {
    // Simple heuristic: if query contains only ASCII characters and common English words
    const englishPattern = /^[a-zA-Z0-9\s\-_.,!?'"()]+$/;
    const commonWords = ['shoes', 'mobile', 'laptop', 'phone', 'watch', 'shirt', 'under', 'above', 'below', 'for', 'with'];
    
    if (!englishPattern.test(query)) {
      return false; // Contains non-ASCII characters, likely non-English
    }

    const words = query.toLowerCase().split(/\s+/);
    const englishWordCount = words.filter(word => 
      word.length <= 3 || // Short words are usually English
      commonWords.some(common => word.includes(common)) ||
      /^[a-z]+$/.test(word) // Simple alphabetic words
    ).length;

    return englishWordCount / words.length > 0.7; // 70% English words
  }

  // Cached translation with fallback
  async getCachedOrFetchTranslation(query) {
    const cached = cacheService.getCachedTranslation(query);
    if (cached) {
      console.log('💾 Using cached translation');
      return cached;
    }

    try {
      const translation = await translateToEnglish(query);
      cacheService.setCachedTranslation(query, translation);
      return translation;
    } catch (error) {
      console.error('Translation error:', error);
      return query; // Fallback to original
    }
  }

  // Cached typo correction with fallback
  async getCachedOrFetchTypoCorrection(query) {
    const cached = cacheService.getCachedTypoCorrection(query);
    if (cached) {
      console.log('💾 Using cached typo correction');
      return cached;
    }

    try {
      const correction = await getCorrectedSpelling(query);
      cacheService.setCachedTypoCorrection(query, correction);
      return correction;
    } catch (error) {
      console.error('Typo correction error:', error);
      return query; // Fallback to original
    }
  }

  // Cached category detection with fallback
  async getCachedOrFetchCategories(query) {
    const cached = cacheService.getCachedCategories(query);
    if (cached) {
      console.log('💾 Using cached categories');
      return cached;
    }

    try {
      const categories = await getExpectedCategories(query);
      cacheService.setCachedCategories(query, categories);
      return categories;
    } catch (error) {
      console.error('Category detection error:', error);
      return []; // Fallback to empty array
    }
  }

  // Get processing statistics
  getStats() {
    return {
      cache: cacheService.getStats(),
      timestamp: new Date().toISOString()
    };
  }
}

// Singleton instance
const parallelGeminiService = new ParallelGeminiService();

module.exports = parallelGeminiService;
