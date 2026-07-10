const elasticClient = require('../elasticClient');
const extractFilters = require('../utils/dynamicQueryParser');
const { filterRelevantProducts } = require('../services/geminiService');
const parallelGeminiService = require('../services/parallelGeminiService');
const cacheService = require('../services/cacheService');
const { updateAndSyncQuery } = require('../services/popularityService');

// ⚡ ULTRA-FAST SEARCH CONTROLLER WITH PARALLEL PROCESSING
const fastSearch = async (req, res) => {
  const searchStartTime = Date.now();
  
  try {
    const { 
      q, 
      page = 1, 
      category, 
      brand, 
      price_gt, 
      price_lt, 
      rating_gte, 
      sortBy = 'relevance' 
    } = req.query;

    if (!q || q.trim() === '') {
      return res.status(400).json({ error: "Query is required" });
    }

    // Check cache first for instant results
    const cacheKey = cacheService.generateSearchCacheKey(q, { category, brand, price_gt, price_lt, rating_gte, sortBy });
    const cachedResults = cacheService.getCachedSearchResults(cacheKey);
    
    if (cachedResults) {
      console.log(`⚡ INSTANT CACHE HIT - returning results in ${Date.now() - searchStartTime}ms`);
      return res.json({
        ...cachedResults,
        cached: true,
        cache_time_ms: Date.now() - searchStartTime
      });
    }

    // ⚡ PARALLEL AI PROCESSING - Much faster than sequential calls
    const aiResults = await parallelGeminiService.processQueryParallel(q);
    const { originalQuery, translatedQuery, correctedQuery, categories: expectedCategories } = aiResults;
    
    // Extract price constraints using fast dynamic parser (no AI needed)
    const parsedFilters = extractFilters(translatedQuery);
    const extractedPriceConstraints = parsedFilters.price || {};
    
    console.log(`⚡ Fast parallel processing: ${aiResults.processingTime}ms`);
    console.log(`🌍 Original: "${originalQuery}", 🔄 Translated: "${translatedQuery}", 🤖 Corrected: "${correctedQuery}"`);
    
    // Merge extracted price constraints with URL parameters (URL params take precedence)
    const finalPriceMin = price_gt || extractedPriceConstraints.gte;
    const finalPriceMax = price_lt || extractedPriceConstraints.lte;
    
    const searchQuery = correctedQuery || translatedQuery || originalQuery;

    // Store the AI-corrected query in database for better suggestions
    try {
      await updateAndSyncQuery(correctedQuery || translatedQuery);
    } catch (error) {
      console.error('Error updating popular query:', error);
    }

    // Build optimized Elasticsearch query
    const elasticQuery = buildOptimizedElasticsearchQuery(
      searchQuery, 
      { category, brand, finalPriceMin, finalPriceMax, rating_gte },
      sortBy
    );

    console.log(`🔍 Searching for: "${originalQuery}" (corrected: "${searchQuery}")`);

    const esStartTime = Date.now();
    const response = await elasticClient.search({
      index: 'products',
      body: elasticQuery
    });
    const esTime = Date.now() - esStartTime;

    const total = response.hits.total.value;
    let results = response.hits.hits.map(hit => ({
      productId: hit._source.productId,
      name: hit._source.name,
      brand: hit._source.brand,
      category: hit._source.category,
      price: hit._source.price,
      rating: hit._source.rating,
      image: hit._source.image,
      url: hit._source.url,
      _score: hit._score
    }));

    // Smart AI filtering - only for ambiguous queries
    const obviousCategoryQueries = ['shoes', 'mobile', 'phone', 'laptop', 'watch', 'shirt', 'dress', 'jeans'];
    const isObviousCategory = obviousCategoryQueries.some(cat => searchQuery.toLowerCase().includes(cat));
    
    if (!isObviousCategory && results.length < 30 && results.length > 5) {
      try {
        console.log(`🤖 Applying smart AI filtering...`);
        const relevantProducts = await filterRelevantProducts(searchQuery, results, 30);
        
        if (relevantProducts && relevantProducts.length > 0) {
          results = relevantProducts;
          console.log(`🎯 AI filtered: ${results.length} relevant products`);
        }
      } catch (error) {
        console.error('AI filtering error:', error);
      }
    } else {
      console.log(`⚡ Skipping AI filtering for speed - ${isObviousCategory ? 'obvious category' : 'optimal result count'}`);
    }

    // Remove duplicates
    const uniqueResults = Array.from(new Map(results.map(item => [item.productId, item])).values());

    const totalSearchTime = Date.now() - searchStartTime;
    console.log(`🚀 ULTRA-FAST search completed in ${totalSearchTime}ms (AI: ${aiResults.processingTime}ms, ES: ${esTime}ms)`);

    const searchResults = {
      products: uniqueResults,
      total,
      page: parseInt(page),
      limit: 100,
      query_analysis: {
        original_query: originalQuery,
        corrected_query: searchQuery,
        search_time: new Date().toISOString(),
        performance: {
          total_time_ms: totalSearchTime,
          ai_processing_ms: aiResults.processingTime,
          elasticsearch_ms: esTime,
          cache_used: false
        }
      }
    };

    // Cache results for future instant access
    cacheService.setCachedSearchResults(cacheKey, searchResults);

    res.json(searchResults);

  } catch (error) {
    console.error('Error during fast search:', error);
    res.status(500).json({ 
      message: 'Error performing search', 
      details: error.message,
      search_time_ms: Date.now() - searchStartTime
    });
  }
};

// Optimized Elasticsearch query builder
function buildOptimizedElasticsearchQuery(searchQuery, filters, sortBy) {
  const { category, brand, finalPriceMin, finalPriceMax, rating_gte } = filters;
  
  // Build filter array
  const filterArray = [];
  
  if (category) filterArray.push({ term: { "category.keyword": category } });
  if (brand) filterArray.push({ term: { "brand.keyword": brand } });
  if (rating_gte) filterArray.push({ range: { rating: { gte: parseFloat(rating_gte) } } });
  
  // Add price filters
  if (finalPriceMin || finalPriceMax) {
    const priceFilter = { range: { price: {} } };
    if (finalPriceMin) priceFilter.range.price.gte = parseFloat(finalPriceMin);
    if (finalPriceMax) priceFilter.range.price.lte = parseFloat(finalPriceMax);
    filterArray.push(priceFilter);
  }

  // Optimized search query with boosted relevance
  const query = {
    size: 100,
    query: {
      bool: {
        must: [
          {
            multi_match: {
              query: searchQuery,
              fields: ['name^4', 'brand^3', 'category^2', 'description'],
              type: 'best_fields',
              minimum_should_match: '60%',
              fuzziness: 'AUTO'
            }
          }
        ],
        filter: filterArray,
        should: [
          { match: { name: { query: searchQuery, boost: 3 } } },
          { match: { brand: { query: searchQuery, boost: 2 } } },
          { match: { category: { query: searchQuery, boost: 1.5 } } }
        ]
      }
    }
  };

  // Add sorting
  if (sortBy === 'price-low-high') {
    query.sort = [{ price: { order: 'asc' } }];
  } else if (sortBy === 'price-high-low') {
    query.sort = [{ price: { order: 'desc' } }];
  } else if (sortBy === 'rating') {
    query.sort = [{ rating: { order: 'desc' } }];
  }
  // Default is relevance (_score)

  return query;
}

// Performance monitoring endpoint
const getPerformanceStats = async (req, res) => {
  const stats = parallelGeminiService.getStats();
  res.json({
    message: "Search performance statistics",
    timestamp: new Date().toISOString(),
    ...stats
  });
};

module.exports = {
  fastSearch,
  getPerformanceStats
};
