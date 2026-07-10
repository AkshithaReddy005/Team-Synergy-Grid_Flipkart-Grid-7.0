const elasticClient = require('../elasticClient');
const extractFilters = require('../utils/dynamicQueryParser');
const { getCorrectedSpelling, getExpectedCategories, filterRelevantProducts, translateToEnglish } = require('../services/geminiService');
const { updateAndSyncQuery } = require('../services/popularityService');

// Known categories for category-specific searches
const knownCategories = ['clothing', 'jewellery', 'footwear', 'mobile phones', 'mobiles & accessories', 
  'automotive', 'home decor', 'home decor & festive needs', 'home furnishing', 'computers'];


// Typo correction is now handled entirely by Gemini AI service
// No static mappings needed - AI detects and corrects any typing mistakes

// All typo correction is now handled by Gemini AI - no static function needed

const isTopRatedQuery = (query) => {
  const topRatedKeywords = ['top rated', 'best rated', 'highest rated', 'top', 'best'];
  return topRatedKeywords.some(keyword => query.toLowerCase().includes(keyword));
};

// Build should clauses for search query
const buildShouldClauses = (searchQuery, originalQuery) => {
  const clauses = [
    // Exact phrase match with corrected query (highest boost)
    {
      multi_match: {
        query: searchQuery,
        fields: ['name^4', 'brand^3', 'category^2'],
        type: 'phrase',
        boost: 10.0
      }
    },
    // Best fields match with corrected query
    {
      multi_match: {
        query: searchQuery,
        fields: ['name^3', 'brand^2', 'category^1'],
        type: 'best_fields',
        boost: 6.0
      }
    },
    // Cross fields match for better relevance
    {
      multi_match: {
        query: searchQuery,
        fields: ['name^2', 'brand^2', 'category^1'],
        type: 'cross_fields',
        boost: 4.0
      }
    },
    // Fuzzy matching for typos (Level 1 - Conservative)
    {
      multi_match: {
        query: searchQuery,
        fields: ['name^2', 'brand^2'],
        fuzziness: 'AUTO',
        prefix_length: 0,
        max_expansions: 20,
        boost: 3.0
      }
    },
    // Fuzzy matching for typos (Level 2 - Medium)
    {
      multi_match: {
        query: searchQuery,
        fields: ['name^1.5', 'brand^1.5'],
        fuzziness: 2,
        prefix_length: 0,
        max_expansions: 25,
        boost: 2.0
      }
    },
    // Fuzzy matching for typos (Level 3 - Lenient)
    {
      multi_match: {
        query: searchQuery,
        fields: ['name^1', 'brand^1', 'category^1'],
        fuzziness: 3,
        prefix_length: 0,
        max_expansions: 30,
        boost: 1.0
      }
    }
  ];
  
  // Add original query phrase match if different from corrected
  if (originalQuery !== searchQuery) {
    clauses.push({
      multi_match: {
        query: originalQuery,
        fields: ['name^3', 'brand^2', 'category^1'],
        type: 'phrase',
        boost: 8.0
      }
    });
  }
  
  return clauses;
};

const extractRatingFromQuery = (query) => {
  const lowerQuery = query.toLowerCase();
  const underMatch = lowerQuery.match(/rating\s+(under|below|less\s+than)\s+(\d+(?:\.\d+)?)/);
  if (underMatch) return { lt: parseFloat(underMatch[2]) };
  
  const aboveMatch = lowerQuery.match(/rating\s+(above|over|greater\s+than)\s+(\d+(?:\.\d+)?)/);
  if (aboveMatch) return { gte: parseFloat(aboveMatch[2]) };

  const starAboveMatch = lowerQuery.match(/(\d+(?:\.\d+)?)\s+(star|rating)\s+(and\s+)?above/);
  if (starAboveMatch) return { gte: parseFloat(starAboveMatch[1]) };

  const exactStarMatch = lowerQuery.match(/(\d+(?:\.\d+)?)\s+star(?!s)/);
  if (exactStarMatch) return { gte: parseFloat(exactStarMatch[1]) };
  
  return null;
};

// Enhanced sorting function with BM25 and relevance support
const getSortOptions = (sortBy, queryKeywords) => {
  switch (sortBy) {
    case 'price-low-high':
      return [{ price: { order: 'asc' } }];
    case 'price-high-low':
      return [{ price: { order: 'desc' } }];
    case 'rating':
      return [{ rating: { order: 'desc' } }];
    case 'popularity':
      return [{ popularity: { order: 'desc' } }];
    case 'relevance':
    default:
      // BM25 relevance scoring with custom boost
      return [{ _score: { order: 'desc' } }];
  }
};

exports.dynamicSearch = async (req, res) => {
  const { q, page = 1, category, brand, sortBy = 'relevance', price_lt, price_gt, rating_gte } = req.query;

  console.log('Dynamic search received sortBy:', sortBy);
  console.log('All query parameters:', req.query);
  console.log('Query string:', req.url);

  if (!q) {
    return res.status(400).json({ error: "Query is required" });
  }

  // Check if explicit filters are applied (skip AI for efficiency)
  const hasUrlFilters = !!(category || brand || price_lt || price_gt || rating_gte);
  const isSimpleQuery = q.trim().split(' ').length <= 2; // Simple 1-2 word queries
  
  console.log(`🔍 Filter Analysis: hasUrlFilters=${hasUrlFilters}, isSimpleQuery=${isSimpleQuery}`);

  let translatedQuery = q;
  let correctedQuery = q;
  let extractedPriceConstraints = {};
  let expectedCategories = null;

  // Always extract price constraints first to determine if we have implicit filters
  const parsedFilters = extractFilters(q);
  const queryKeywords = parsedFilters.keywords.join(' ');
  extractedPriceConstraints = parsedFilters.price || {};
  
  // Check for both URL filters AND extracted price constraints
  const hasExplicitFilters = hasUrlFilters || Object.keys(extractedPriceConstraints).length > 0;
  
  console.log(`💰 Extracted price constraints:`, extractedPriceConstraints);
  console.log(`🔍 Final filter analysis: hasExplicitFilters=${hasExplicitFilters} (URL: ${hasUrlFilters}, Price: ${Object.keys(extractedPriceConstraints).length > 0})`);

  // If there are no explicit filters, use AI to enhance the keywords
  if (!hasExplicitFilters) {
    console.log(`🤖 Using Gemini AI for query enhancement on keywords: "${queryKeywords}"`);

    // Step 1: Translate the core keywords to English if needed
    const translatedKeywords = await translateToEnglish(queryKeywords);

    // Step 2: Apply typo correction to the translated keywords
    correctedQuery = await getCorrectedSpelling(translatedKeywords);

    console.log(`🌍 Original keywords: "${queryKeywords}", 🔄 Translated: "${translatedKeywords}", 🤖 Corrected: "${correctedQuery}"`);
  } else {
    // If filters are present, skip AI and use the extracted keywords directly
    console.log(`⚡ Skipping Gemini AI - explicit filters detected. Using keywords: "${queryKeywords}"`);
    correctedQuery = queryKeywords;
  }

  // The final search query is the corrected set of keywords
  const searchQuery = correctedQuery;
  
  // Merge extracted price constraints with URL parameters (URL params take precedence)
  const finalPriceMin = price_gt || extractedPriceConstraints.gte;
  const finalPriceMax = price_lt || extractedPriceConstraints.lte;

  // Debug: Show extracted, merged, and raw price constraints
  console.log('🔎 [DEBUG] Extracted price constraints:', extractedPriceConstraints);
  console.log('🔎 [DEBUG] URL price_gt:', price_gt, 'URL price_lt:', price_lt);
  console.log('🔎 [DEBUG] Final price filter: Min:', finalPriceMin, 'Max:', finalPriceMax);

  if (finalPriceMin || finalPriceMax) {
    if (finalPriceMin && finalPriceMax && Number(finalPriceMax) < Number(finalPriceMin)) {
      console.warn('⚠️ [WARNING] Price max is less than min. Check extraction logic!');
    }
    if (!finalPriceMin || !finalPriceMax) {
      console.warn('⚠️ [WARNING] Only one bound present for a "between" query.');
    }
    console.log(`💰 Final price constraints - Min: ${finalPriceMin || 'none'}, Max: ${finalPriceMax || 'none'}`);
  }

  // Store the AI-corrected query in database for better suggestions
  try {
    const queryToStore = correctedQuery && correctedQuery !== q ? correctedQuery : q;
    await updateAndSyncQuery(queryToStore.trim());
    console.log(`[Dynamic Search] Stored enhanced query: "${queryToStore.trim()}" (original: "${q.trim()}")`);
  } catch (error) {
    console.error(`[Dynamic Search] Failed to store query "${queryToStore.trim()}":`, error);
    // Don't fail the search if storing fails
  }

  // Fetch all available products - no pagination limit
  const pageSize = 10000; // Large number to fetch all products
  const from = 0; // Always start from 0 to get all products
  
  // Minimal filtering for testing
  const filter_clauses = [];
  
  // Add basic filters only if provided
  if (category) {
    filter_clauses.push({ term: { 'category.keyword': category } });
  }
  if (brand) {
    filter_clauses.push({ term: { 'brand.keyword': brand } });
  }
  
  // Apply merged price constraints (from URL params or extracted from query)
  if (finalPriceMin || finalPriceMax) {
    const priceRange = {};
    if (finalPriceMin) {
      priceRange.gte = parseFloat(finalPriceMin);
    }
    if (finalPriceMax) {
      priceRange.lte = parseFloat(finalPriceMax);
    }
    filter_clauses.push({ range: { price: priceRange } });
    console.log(`💰 [DEBUG] Applied price filter:`, { range: { price: priceRange } });
  }
  
  if (rating_gte) {
    filter_clauses.push({ range: { rating: { gte: parseFloat(rating_gte) } } });
  }

  // Get sort options based on sortBy parameter
  const sort_options = getSortOptions(sortBy, '');

  try {
    console.log(`🔍 Searching for: "${q}" (corrected: "${searchQuery}")`);
    
    // Multi-layered Category Detection
    let expectedCategories = [];
    // 1. Prioritize local parser for speed and reliability
    if (parsedFilters.category) {
      expectedCategories = [parsedFilters.category];
      console.log(`⚡️ [Dynamic Search] Using locally parsed category: ${parsedFilters.category}`);
    } 
    // 2. Fallback to AI for ambiguous queries if no local category and no explicit filters
    else if (!hasExplicitFilters) {
      try {
        console.log(`🤖 [Dynamic Search] No local category found. Calling Gemini AI for category detection...`);
        const detected = await getExpectedCategories(queryKeywords);
        if (detected && detected.length > 0) {
          expectedCategories = detected;
          console.log(`🎯 [Dynamic Search] Gemini AI detected categories: ${expectedCategories.join(', ')}`);
        }
      } catch (error) {
        console.error('[Dynamic Search] Gemini AI category detection error:', error);
        // Gracefully continue without AI categories if the API fails
      }
    }
    // (DEBUG) Print final filter_clauses before query
    console.log('🔎 [DEBUG] Final filter_clauses:', JSON.stringify(filter_clauses, null, 2));
    const shouldClauses = [
      // Exact phrase match with corrected query (highest boost)
      {
        multi_match: {
          query: searchQuery,
          fields: ['name^5', 'brand^4', 'category^3'],
          type: 'phrase',
          boost: 15.0
        }
      },
      // Best fields match with higher minimum should match
      {
        multi_match: {
          query: searchQuery,
          fields: ['name^4', 'brand^3', 'category^2'],
          type: 'best_fields',
          minimum_should_match: '75%',
          boost: 10.0
        }
      },
      // Cross fields match for better relevance
      {
        multi_match: {
          query: searchQuery,
          fields: ['name^3', 'brand^2'],
          type: 'cross_fields',
          minimum_should_match: '60%',
          boost: 8.0
        }
      },
      // Conservative fuzzy matching (only for minor typos)
      {
        multi_match: {
          query: searchQuery,
          fields: ['name^2', 'brand^2'],
          fuzziness: 1, // Reduced from 'AUTO' to be more strict
          prefix_length: 1, // Require at least 1 character prefix match
          max_expansions: 10, // Reduced expansions
          boost: 4.0
        }
      }
    ];
    
    // Add original query if different from corrected
    if (q !== searchQuery) {
      shouldClauses.push({
        multi_match: {
          query: q,
          fields: ['name^3', 'brand^2', 'category^1'],
          type: 'phrase',
          boost: 8.0
        }
      });
    }
    
    const must_clauses = [
      // Base relevance: ensure at least one strong match in name, brand, or category
      {
        multi_match: {
          query: searchQuery,
          fields: ['name^3', 'brand^2', 'category^1'],
          type: 'best_fields',
          minimum_should_match: '50%'
        }
      }
    ];

    // Add AI-detected categories to the SHOULD clause to boost relevance, not as a strict filter
    if (expectedCategories.length > 0) {
      shouldClauses.push({
        "bool": {
          "should": expectedCategories.map(cat => (
            { "match": { "category": cat } }
          )),
          "minimum_should_match": 1,
          "boost": 5.0 // Add a moderate boost for matching the detected category
        }
      });
    }

    // Strict search with relevance threshold and must clause
    const searchBody = {
      from: (page - 1) * 100,
      size: 100,
      min_score: 2.0, // Set minimum score threshold to filter out irrelevant results
      query: {
        bool: {
          must: must_clauses,
          should: shouldClauses,
          minimum_should_match: 1,
          filter: filter_clauses
        }
      },
      sort: getSortOptions(sortBy, queryKeywords)
    };

    console.log('Search body:', JSON.stringify(searchBody, null, 2));

    const response = await elasticClient.search({
      index: 'products_index',
      body: searchBody
    });

    if (!response || !response.hits) {
      throw new Error('Invalid response structure from Elasticsearch');
    }

    let results = response.hits.hits.map(hit => ({ ...hit._source, _score: hit._score }));
    const total = response.hits.total.value;
    
    console.log(`Found ${results.length} results for query: "${q}"`);
    
    // Smart AI filter: filter_clausesches OR when explicit filters are applied
    const obviousCategoryQueries = ['shoes', 'mobile', 'phone', 'laptop', 'watch', 'shirt', 'dress', 'jeans', 'headphones', 'earphones'];
    const isObviousCategory = obviousCategoryQueries.some(cat => searchQuery.toLowerCase().includes(cat));
    
    // Skip AI filtering if explicit filters are applied OR obvious category OR too many results
    if (!hasExplicitFilters && !isObviousCategory && results.length > 0 && results.length < 50) {
      console.log(`🤖 Applying Gemini AI relevance filtering for ambiguous query...`);
      
      try {
        const relevantProducts = await filterRelevantProducts(searchQuery, results, 50);
        
        if (relevantProducts && relevantProducts.length > 0) {
          results = relevantProducts;
          console.log(`🎯 Gemini AI filtered results: ${results.length} relevant products (from ${response.hits.hits.length} original)`);
        } else {
          console.log(`⚠️ Gemini AI found no relevant products, keeping original results`);
        }
      } catch (error) {
        console.error('Gemini AI relevance filtering error:', error);
        console.log(`⚠️ Falling back to original Elasticsearch results`);
      }
    } else if (hasExplicitFilters) {
      console.log(`⚡ Skipping AI filtering - explicit filters applied, results already filtered by user preferences`);
    } else if (isObviousCategory) {
      console.log(`⚡ Skipping AI filtering for obvious category query "${searchQuery}" - keeping all ${results.length} results for better speed and inclusivity`);
    } else {
      console.log(`⚡ Skipping AI filtering - too many results (${results.length}) for efficient processing`);
    }
    
    // Safely handle aggregations
    const categoriesAggs = response.aggregations?.categories?.buckets || [];
    const brandsAggs = response.aggregations?.brands?.buckets || [];

    const uniqueResults = Array.from(new Map(results.map(item => [item.productId, item])).values());

    res.json({
      products: uniqueResults,
      total,
      page,
      pageSize,
      categories: categoriesAggs,
      brands: brandsAggs
    });

  } catch (error) {
    console.error('Error during Elasticsearch search:', error.meta?.body || error);
    res.status(500).json({ message: 'Error performing search', details: error.meta?.body?.error || error.message });
  }
};
