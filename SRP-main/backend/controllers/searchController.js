const elasticClient = require('../elasticClient');
const { getCorrectedSpelling, getConceptualSearchKeywords, getExpectedCategories } = require('../services/geminiService');
const extractFilters = require('../utils/dynamicQueryParser');
const axios = require('axios');
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Common typo mappings for better search results
const commonTypos = {
  'wach': 'watch',
  'fone': 'phone',
  'lapto': 'laptop',
  'earpods': 'earbuds',
  'blututh': 'bluetooth',
  'raning': 'running',
  'gents': 'men',
  'ladies': 'women',
  'chajjar': 'coffee',
  'powerbank': 'power bank',
  'samsun': 'samsung',
  'iphne': 'iphone',
  'nke': 'nike',
  'addidas': 'adidas',
  'puma': 'puma',
  'fastrack': 'fastrack',
  'titan': 'titan'
};

// Function to correct common typos
const getCorrectedQuery = (query) => {
  const lowerQuery = query.toLowerCase();
  for (const [typo, correction] of Object.entries(commonTypos)) {
    if (lowerQuery.includes(typo)) {
      return query.replace(new RegExp(typo, 'gi'), correction);
    }
  }
  return query;
};

// Popular queries for gift suggestions and common searches
const POPULAR_QUERIES = [
  'gift for sister', 'gift for best friend', 'gift for wife', 'gift for husband',
  'gift for mom', 'gift for dad', 'gift for girlfriend', 'gift for boyfriend',
  'birthday gift', 'anniversary gift', 'wedding gift', 'housewarming gift',
  'diwali gift', 'rakhi gift', 'christmas gift', 'valentine gift',
  'mobile phones', 'laptops', 'smartphones', 'headphones', 'watches',
  'shoes', 'clothing', 'jewellery', 'home decor', 'kitchen appliances',
  'books', 'toys', 'sports equipment', 'fitness equipment', 'beauty products'
];

exports.getAutosuggestions = async (req, res) => {
  const query = req.query.q;
  if (!query) {
    return res.json([]);
  }

  try {
    const finalSuggestions = new Map();
    const queryLower = query.toLowerCase();

    // 1. PRIORITY 1: Get frequency-based suggestions from database (highest priority)
    try {
      const frequencySuggestionPromise = elasticClient.search({
        index: 'search_queries',
        body: {
          suggest: {
            'frequency-suggester': {
              prefix: queryLower,
              completion: {
                field: 'suggest',
                size: 8,
                skip_duplicates: true,
                fuzzy: { fuzziness: 'AUTO' }
              }
            }
          }
        }
      });

      const frequencyResponse = await frequencySuggestionPromise;
      
      if (frequencyResponse.suggest && 
          frequencyResponse.suggest['frequency-suggester'] && 
          frequencyResponse.suggest['frequency-suggester'][0] && 
          frequencyResponse.suggest['frequency-suggester'][0].options.length > 0) {
        
        frequencyResponse.suggest['frequency-suggester'][0].options.forEach(option => {
          if (option._source && option._source.queryText) {
            const suggestion = option._source.queryText;
            if (!finalSuggestions.has(suggestion.toLowerCase())) {
              finalSuggestions.set(suggestion.toLowerCase(), { 
                text: suggestion, 
                type: 'frequency',
                weight: option._source.weight || 1
              });
            }
          }
        });
      }
    } catch (error) {
      console.error('[Autosuggest] Error fetching frequency-based suggestions:', error);
    }

    // 2. PRIORITY 2: Get popular queries that match the prefix
    const matchingPopularQueries = POPULAR_QUERIES.filter(popularQuery => 
      popularQuery.toLowerCase().startsWith(queryLower) || 
      popularQuery.toLowerCase().includes(queryLower)
    );

    matchingPopularQueries.forEach(popularQuery => {
      if (!finalSuggestions.has(popularQuery.toLowerCase())) {
        finalSuggestions.set(popularQuery.toLowerCase(), { 
          text: popularQuery, 
          type: 'popular',
          weight: 5 // Medium weight for popular queries
        });
      }
    });

    // 3. PRIORITY 3: Get product-based suggestions (lowest priority)
    try {
      const productSuggestionPromise = elasticClient.search({
        index: 'products_index',
        body: {
          size: 15,
          query: {
            bool: {
              should: [
                {
                  multi_match: {
                    query: query,
                    fields: ['name^3', 'brand^2', 'category^1'],
                    type: 'bool_prefix',
                    boost: 2.0
                  }
                },
                {
                  multi_match: {
                    query: query,
                    fields: ['name^2', 'brand^1'],
                    fuzziness: 'AUTO',
                    boost: 1.0
                  }
                }
              ],
              minimum_should_match: 1
            }
          }
        }
      });

      const productResponse = await productSuggestionPromise;
      
      productResponse.hits.hits.forEach(hit => {
        if (hit._source) {
          let suggestion = '';
          let type = 'product';
          
          // Prefer name, then brand, then category
          if (hit._source.name) {
            suggestion = hit._source.name;
          } else if (hit._source.brand) {
            suggestion = hit._source.brand;
            type = 'brand';
          } else if (hit._source.category) {
            suggestion = hit._source.category;
            type = 'category';
          }
          
          if (suggestion && !finalSuggestions.has(suggestion.toLowerCase())) {
            finalSuggestions.set(suggestion.toLowerCase(), { 
              text: suggestion, 
              type: type,
              weight: 1 // Lowest weight for product suggestions
            });
          }
        }
      });
    } catch (error) {
      console.error('[Autosuggest] Error fetching product-based suggestions:', error);
    }

    // 4. Sort suggestions by priority and weight
    const suggestions = Array.from(finalSuggestions.values())
      .sort((a, b) => {
        // First sort by type priority: frequency > popular > product
        const typePriority = { frequency: 3, popular: 2, product: 1, brand: 1, category: 1 };
        const aPriority = typePriority[a.type] || 0;
        const bPriority = typePriority[b.type] || 0;
        
        if (aPriority !== bPriority) {
          return bPriority - aPriority;
        }
        
        // Then sort by weight
        return (b.weight || 0) - (a.weight || 0);
      })
      .slice(0, 10) // Limit to 10 suggestions
      .map(suggestion => ({
        text: suggestion.text,
        type: suggestion.type
      }));

    res.json(suggestions);

  } catch (error) {
    console.error('[Autosuggest] Error:', error);
    res.status(500).json({ message: 'Error fetching autosuggestions' });
  }
};

exports.searchProducts = async (req, res) => {
  const { q, page = 1, category, brand, sortBy = 'relevance', price_lt, price_gt, rating_gte } = req.query;

  console.log('[Search] Original query:', q, ', Corrected query:', q);

  if (!q) {
    return res.status(400).json({ error: "Query is required" });
  }

  // Extract price constraints from natural language query
  const extractedPriceConstraints = extractFilters(q).price || {};
  console.log(' [DEBUG] Search Controller - Extracted price constraints:', extractedPriceConstraints);
  
  // Merge extracted price constraints with URL parameters (URL params take precedence)
  const finalPriceMin = price_gt || extractedPriceConstraints.gte;
  const finalPriceMax = price_lt || extractedPriceConstraints.lte;
  
  console.log(' [DEBUG] Search Controller - Final price filter: Min:', finalPriceMin, 'Max:', finalPriceMax);

  // Get corrected query for common typos
  const correctedQuery = getCorrectedQuery(q);
  const originalQuery = q;
  const searchQuery = correctedQuery !== originalQuery ? correctedQuery : originalQuery;
  
  console.log(`[Search] Original query: "${originalQuery}", Corrected query: "${correctedQuery}"`);

  // Enhanced category detection using the improved parser

  const parsedFilters = extractFilters(searchQuery);
  const queryKeywords = parsedFilters.keywords.join(' ');
  const isCategoryQuery = parsedFilters.category !== null;

  let searchQueryBody;
  if (isCategoryQuery) {
    // Enhanced category-specific search with better filtering
    searchQueryBody = {
      bool: {
        must: [
          // Category-specific boost
          {
            multi_match: {
              query: parsedFilters.category,
              fields: ['category^4', 'name^3', 'description^2'],
              type: 'phrase',
              boost: 3.0
            }
          }
        ],
        should: [
          // General keyword matching within the category
          {
            multi_match: {
              query: q,
              fields: ['name^3', 'brand^2', 'description^1'],
              type: 'best_fields',
              minimum_should_match: '50%'
            }
          }
        ],
        filter: [
          // Ensure results are relevant to the detected category
          {
            multi_match: {
              query: parsedFilters.category,
              fields: ['category', 'name', 'description'],
              type: 'best_fields',
              minimum_should_match: '30%'
            }
          }
        ]
      }
    };
  } else {
    // Ultra-strict search query for compound queries
    searchQueryBody = {
      bool: {
        must: [
          // Ultra-strict relevance requirement
          {
            multi_match: {
              query: q,
              fields: ['name^4', 'brand^3', 'category^2'],
              type: 'best_fields',
              minimum_should_match: '75%' // Much higher requirement
            }
          }
        ],
        should: [
          // Exact phrase match (highest boost)
          {
            multi_match: {
              query: queryKeywords, // Use extracted keywords
              fields: ['name^5', 'brand^4', 'category^2'],
              type: 'phrase',
              boost: 10.0
            }
          },
          // Standard best_fields match for general relevance
          {
            multi_match: {
              query: queryKeywords, // Use extracted keywords
              fields: ['name^3', 'brand^2', 'description', 'category'],
              type: 'best_fields',
              fuzziness: 'AUTO',
              prefix_length: 1,
              boost: 3.0
            }
          },
          // Very conservative fuzzy matching for typos (minimal boost, very strict)
          {
            multi_match: {
              query: queryKeywords, // Use extracted keywords
              fields: ['name^2', 'brand^2'],
              fuzziness: 'AUTO',
              prefix_length: 2, // Require longer prefix match
              max_expansions: 3, // Very limited expansions
              boost: 0.5
            }
          }
        ],
        minimum_should_match: 1
      }
    };
  }

  try {
    // For category searches, return all products. For other searches, limit to reasonable number
    const isCategorySearch = isCategoryQuery;
    const size = 10000; // Fetch all available products
    
    // Default sort by relevance score (client-side sorting will handle the rest)
    let sortOptions = [{ _score: { order: 'desc' } }];

    // Build filter clauses for price and other filters
    const filter_clauses = [];
    
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
      console.log('💰 [DEBUG] Search Controller - Applied price filter:', { range: { price: priceRange } });
    }
    
    // Add category filter if specified
    if (category) {
      filter_clauses.push({ term: { 'category.keyword': category } });
    }
    
    // Add brand filter if specified
    if (brand) {
      filter_clauses.push({ term: { 'brand.keyword': brand } });
    }
    
    // Add rating filter if specified
    if (rating_gte) {
      filter_clauses.push({ range: { rating: { gte: parseFloat(rating_gte) } } });
    }

    // AI-powered Category Detection (fully dynamic)
    let expectedCategories = [];
    if (!category) { // Only run AI if no manual category filter is set
      try {
        console.log(`🤖 [Search Controller] Calling Gemini AI for category detection...`);
        const detected = await getExpectedCategories(queryKeywords);
        if (detected && detected.length > 0) {
          expectedCategories = detected;
          console.log(`🎯 [Search Controller] Gemini AI detected categories: ${expectedCategories.join(', ')}`);
        }
      } catch (error) {
        console.error('[Search Controller] Gemini AI category detection error:', error);
      }
    }

    const must_clauses = [
      {
        function_score: {
          query: searchQueryBody, // Use the dynamically built search query
          functions: [
            {
              filter: { exists: { field: "rating" } },
              field_value_factor: {
                field: "rating",
                factor: 1.5,
                modifier: "ln1p",
                missing: 1
              },
              weight: 1.5
            },
            {
              filter: { exists: { field: "popularity" } },
              field_value_factor: {
                field: "popularity",
                factor: 0.2,
                modifier: "ln1p",
                missing: 1
              },
              weight: 2.5
            }
          ],
          score_mode: "sum",
          boost_mode: "multiply"
        }
      }
    ];

    // Add AI-detected category filter to the FILTER clause for strict, non-scoring, but case-insensitive filtering
    if (expectedCategories.length > 0) {
      filter_clauses.push({
        "bool": {
          "should": expectedCategories.map(cat => (
            { "match": { "category": cat } }
          )),
          "minimum_should_match": 1
        }
      });
    }
    
    console.log('🔎 [DEBUG] Search Controller - Final filter_clauses:', JSON.stringify(filter_clauses, null, 2));

    const searchBody = {
      size: size,
      min_score: 2.0, // Much higher threshold to eliminate irrelevant results
      query: {
        bool: {
          must: must_clauses,
          filter: filter_clauses
        }
      }
    };

    // Add sort if specified
    if (sortOptions.length > 0) {
      searchBody.sort = sortOptions;
    }



    const response = await elasticClient.search({
      index: 'products_index',
      body: searchBody
    });

    console.log(`[Search Controller] Total products found: ${response.hits.total.value}`);
    console.log(`[Search Controller] Query with filters applied successfully`);
    
    // Flatten the response to match the dynamic search format
    const results = response.hits.hits.map(hit => ({ ...hit._source, _score: hit._score }));
    let uniqueResults = Array.from(new Map(results.map(item => [item.productId || item.pid || item.id, item])).values());

    // Optional personalization reranking
    if ((req.query.personalize || '').toString().toLowerCase() === 'true' && uniqueResults.length > 0) {
      try {
        const features_list = uniqueResults.map(p => {
          const price = toNumberSafe(p.price ?? p.discounted_price ?? p.retail_price);
          const retail = toNumberSafe(p.retail_price ?? p.price ?? price);
          const discounted = toNumberSafe(p.discounted_price ?? p.price ?? price);
          const discount_pct = retail > 0 ? Math.max(0, Math.min(1, (retail - discounted) / retail)) : 0;
          const rating = toNumberSafe(p.rating ?? p.product_rating);
          const popularity = toNumberSafe(p.popularity);
          const brand_freq_proxy = 1; // simple proxy; real-time frequency requires aggregation
          return [
            safeLog(discounted),
            safeLog(retail),
            discount_pct,
            rating,
            safeLog(popularity),
            safeLog(brand_freq_proxy),
          ];
        });

        const { data } = await axios.post(`${ML_SERVICE_URL}/personalize/score-batch`, { features_list });
        const scores = Array.isArray(data?.scores) ? data.scores : [];
        const alpha = 0.7; // ES relevance weight
        const beta = 0.3;  // personalization weight
        uniqueResults = uniqueResults.map((p, idx) => {
          const pScore = typeof scores[idx] === 'number' ? scores[idx] : 0;
          const base = typeof p._score === 'number' ? p._score : 0;
          const finalScore = alpha * base + beta * pScore;
          return { ...p, _score: finalScore, _score_es: base, _score_personal: pScore };
        }).sort((a, b) => (b._score || 0) - (a._score || 0));
      } catch (e) {
        console.error('[Search Controller] Personalization rerank error:', e?.response?.data || e?.message || e);
      }
    }

    res.json({ products: uniqueResults, total: response.hits.total.value });

  } catch (error) {
    console.error('[Search] Error fetching search results:', error);
    res.status(500).json({ message: 'Could not fetch search results. Please try again later.' });
  }
};

function toNumberSafe(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : 0;
}

function safeLog(x) {
  x = toNumberSafe(x);
  if (x < 0) x = 0;
  return Math.log1p(x);
}
