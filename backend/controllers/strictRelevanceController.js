const elasticClient = require('../elasticClient');
const extractFilters = require('../utils/dynamicQueryParser');
const parallelGeminiService = require('../services/parallelGeminiService');
const relevanceFilter = require('../services/relevanceFilter');
const { updateAndSyncQuery } = require('../services/popularityService');

// 🎯 STRICT RELEVANCE CONTROLLER - NO IRRELEVANT PRODUCTS ALLOWED
const strictRelevanceSearch = async (req, res) => {
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

    console.log(`🎯 STRICT RELEVANCE SEARCH: "${q}" - NO IRRELEVANT PRODUCTS ALLOWED`);

    // ⚡ PARALLEL AI PROCESSING
    const aiResults = await parallelGeminiService.processQueryParallel(q);
    const { originalQuery, translatedQuery, correctedQuery } = aiResults;
    
    // Extract price constraints using dynamic parser
    const parsedFilters = extractFilters(translatedQuery);
    const extractedPriceConstraints = parsedFilters.price || {};
    
    console.log(`⚡ AI processing: ${aiResults.processingTime}ms`);
    console.log(`🌍 Query: "${originalQuery}" → "${correctedQuery}"`);
    
    // Merge price constraints
    const finalPriceMin = price_gt || extractedPriceConstraints.gte;
    const finalPriceMax = price_lt || extractedPriceConstraints.lte;
    
    const searchQuery = correctedQuery || translatedQuery || originalQuery;

    // Store corrected query
    try {
      await updateAndSyncQuery(correctedQuery || translatedQuery);
    } catch (error) {
      console.error('Error updating popular query:', error);
    }

    // 🔍 BUILD STRICT ELASTICSEARCH QUERY
    const elasticQuery = buildStrictElasticsearchQuery(
      searchQuery, 
      { category, brand, finalPriceMin, finalPriceMax, rating_gte },
      sortBy
    );

    console.log(`🔍 Executing strict Elasticsearch query...`);
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

    console.log(`📊 Elasticsearch returned ${results.length} initial results`);

    // 🎯 APPLY STRICT MULTI-LAYER RELEVANCE FILTERING
    const originalCount = results.length;
    
    if (results.length > 0) {
      console.log(`🎯 APPLYING STRICT RELEVANCE FILTERING - ELIMINATING IRRELEVANT PRODUCTS`);
      
      try {
        results = await relevanceFilter.applyStrictRelevanceFilter(searchQuery, results, 50);
        console.log(`✅ STRICT FILTERING SUCCESS: ${results.length}/${originalCount} highly relevant products`);
        
        // If too few results, apply moderate filtering
        if (results.length < 3 && originalCount > 5) {
          console.log(`⚠️ Too few results (${results.length}), applying moderate relevance filter...`);
          
          const allResults = response.hits.hits.map(hit => ({
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
          
          // Apply basic keyword filtering as fallback
          results = relevanceFilter.basicKeywordFilter(searchQuery, allResults);
          results = results.slice(0, 20); // Limit to top 20 most relevant
          console.log(`🔄 Moderate filtering applied: ${results.length} relevant products`);
        }
        
      } catch (error) {
        console.error('❌ Relevance filtering error:', error);
        // Apply basic filtering as absolute fallback
        results = relevanceFilter.basicKeywordFilter(searchQuery, results).slice(0, 30);
        console.log(`🔄 Fallback filtering applied: ${results.length} products`);
      }
    }

    // Remove duplicates and ensure uniqueness
    const uniqueResults = Array.from(new Map(results.map(item => [item.productId, item])).values());

    // Final relevance check - remove obviously irrelevant products
    const finalResults = uniqueResults.filter(product => {
      return !relevanceFilter.isObviouslyIrrelevant(searchQuery, product);
    });

    const totalSearchTime = Date.now() - searchStartTime;
    
    console.log(`🎯 FINAL RESULT: ${finalResults.length} HIGHLY RELEVANT products (${totalSearchTime}ms)`);
    console.log(`📊 Filtering Stats: ${response.hits.hits.length} → ${originalCount} → ${finalResults.length}`);

    // Aggregations for filters
    const categoriesAggs = response.aggregations?.categories?.buckets || [];
    const brandsAggs = response.aggregations?.brands?.buckets || [];

    res.json({
      products: finalResults,
      total: finalResults.length, // Return actual filtered count
      page: parseInt(page),
      limit: 50,
      query_analysis: {
        original_query: originalQuery,
        corrected_query: searchQuery,
        search_time: new Date().toISOString(),
        relevance_filtering: {
          elasticsearch_results: response.hits.hits.length,
          after_basic_filtering: originalCount,
          final_relevant_products: finalResults.length,
          filtering_success_rate: `${Math.round((finalResults.length / response.hits.hits.length) * 100)}%`
        },
        performance: {
          total_time_ms: totalSearchTime,
          ai_processing_ms: aiResults.processingTime,
          elasticsearch_ms: esTime,
          relevance_filtering_ms: totalSearchTime - aiResults.processingTime - esTime
        }
      },
      filters: {
        categories: categoriesAggs,
        brands: brandsAggs
      }
    });

  } catch (error) {
    console.error('❌ Strict relevance search error:', error);
    res.status(500).json({ 
      message: 'Error performing strict relevance search', 
      details: error.message,
      search_time_ms: Date.now() - searchStartTime
    });
  }
};

// Build strict Elasticsearch query with high relevance thresholds
function buildStrictElasticsearchQuery(searchQuery, filters, sortBy) {
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

  // STRICT query with high relevance requirements
  const query = {
    size: 100,
    query: {
      bool: {
        must: [
          {
            multi_match: {
              query: searchQuery,
              fields: ['name^5', 'brand^4', 'category^3', 'description^1'],
              type: 'best_fields',
              minimum_should_match: '70%', // STRICT: 70% of words must match
              fuzziness: 'AUTO',
              boost: 2.0
            }
          }
        ],
        filter: filterArray,
        should: [
          // Exact phrase match (highest boost)
          { 
            match_phrase: { 
              name: { 
                query: searchQuery, 
                boost: 10.0 
              } 
            } 
          },
          // Name match with high boost
          { 
            match: { 
              name: { 
                query: searchQuery, 
                boost: 5.0,
                minimum_should_match: '60%'
              } 
            } 
          },
          // Brand match
          { 
            match: { 
              brand: { 
                query: searchQuery, 
                boost: 3.0 
              } 
            } 
          },
          // Category match
          { 
            match: { 
              category: { 
                query: searchQuery, 
                boost: 2.0 
              } 
            } 
          }
        ],
        minimum_should_match: 1 // At least one should clause must match
      }
    },
    // Add aggregations for filtering
    aggs: {
      categories: {
        terms: { field: "category.keyword", size: 20 }
      },
      brands: {
        terms: { field: "brand.keyword", size: 20 }
      }
    }
  };

  // Add sorting
  if (sortBy === 'price-low-high') {
    query.sort = [{ price: { order: 'asc' } }, { _score: { order: 'desc' } }];
  } else if (sortBy === 'price-high-low') {
    query.sort = [{ price: { order: 'desc' } }, { _score: { order: 'desc' } }];
  } else if (sortBy === 'rating') {
    query.sort = [{ rating: { order: 'desc' } }, { _score: { order: 'desc' } }];
  } else {
    // Default: relevance first, then rating
    query.sort = [{ _score: { order: 'desc' } }, { rating: { order: 'desc' } }];
  }

  return query;
}

module.exports = {
  strictRelevanceSearch
};
