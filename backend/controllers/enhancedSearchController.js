const elasticClient = require('../elasticClient');
const { 
  getCorrectedSpelling, 
  getConceptualSearchKeywords, 
  extractQueryIntent,
  generateProductRecommendations,
  enhanceProductDescriptions 
} = require('../services/geminiService');

exports.enhancedSearch = async (req, res) => {
  const { q, page = 1, limit = 20, sortBy } = req.query;
  
  if (!q) {
    return res.status(400).json({ message: 'Search query is required.' });
  }

  try {
    // Step 1: Use Gemini to enhance the query
    const [correctedQuery, queryIntent, relatedKeywords] = await Promise.all([
      getCorrectedSpelling(q),
      extractQueryIntent(q),
      getConceptualSearchKeywords(q)
    ]);

    console.log(`🔍 Original query: "${q}"`);
    console.log(`✅ Corrected query: "${correctedQuery}"`);
    console.log(`🎯 Query intent:`, queryIntent);
    console.log(`🔗 Related keywords:`, relatedKeywords);

    // Step 2: Build enhanced search query
    const searchTerms = [correctedQuery, ...relatedKeywords.slice(0, 3)];
    const shouldClauses = searchTerms.map(term => ({
      multi_match: {
        query: term,
        fields: ['name^4', 'brand^3', 'category^2', 'description'],
        fuzziness: 'AUTO'
      }
    }));

    const searchBody = {
      from: (page - 1) * limit,
      size: limit,
      query: {
        bool: {
          should: shouldClauses,
          minimum_should_match: 1
        }
      },
      sort: sortBy === 'price_low' ? [{ price: { order: 'asc' } }] :
            sortBy === 'price_high' ? [{ price: { order: 'desc' } }] :
            sortBy === 'rating' ? [{ rating: { order: 'desc' } }] :
            [{ _score: { order: 'desc' } }]
    };

    // Step 3: Execute search
    const response = await elasticClient.search({
      index: 'products_index',
      body: searchBody
    });

    const products = response.hits.hits.map(hit => ({ ...hit._source, _score: hit._score }));
    const total = response.hits.total.value;

    // Step 4: Use Gemini to enhance product descriptions and generate recommendations
    const [enhancedProducts, recommendations] = await Promise.all([
      enhanceProductDescriptions(products.slice(0, 5), correctedQuery),
      generateProductRecommendations(correctedQuery, products)
    ]);

    // Step 5: Merge enhanced data with products
    const enhancedResults = products.map(product => {
      const enhanced = enhancedProducts.find(ep => ep.product_id === product.productId);
      return {
        ...product,
        highlights: enhanced?.highlights || [],
        match_reasons: enhanced?.match_reasons || [],
        recommendation_strength: enhanced?.recommendation_strength || 'medium'
      };
    });

    // Step 6: Prepare response
    const responseData = {
      products: enhancedResults,
      total,
      page: parseInt(page),
      limit: parseInt(limit),
      query_analysis: {
        original_query: q,
        corrected_query: correctedQuery,
        query_intent: queryIntent,
        related_keywords: relatedKeywords
      },
      recommendations: recommendations,
      search_metadata: {
        total_results: total,
        search_time: new Date().toISOString(),
        enhanced_with_ai: true
      }
    };

    res.json(responseData);

  } catch (error) {
    console.error('Enhanced search error:', error);
    res.status(500).json({ 
      message: 'Search failed', 
      error: error.message,
      fallback: true 
    });
  }
};

exports.getSmartRecommendations = async (req, res) => {
  const { q, productId } = req.query;
  
  if (!q && !productId) {
    return res.status(400).json({ message: 'Query or product ID is required.' });
  }

  try {
    let baseQuery = q;
    
    // If productId is provided, get the product first
    if (productId) {
      const productResponse = await elasticClient.search({
        index: 'products_index',
        body: {
          query: { term: { productId: productId } },
          size: 1
        }
      });
      
      if (productResponse.hits.hits.length > 0) {
        const product = productResponse.hits.hits[0]._source;
        baseQuery = product.name;
      }
    }

    // Get similar products
    const similarResponse = await elasticClient.search({
      index: 'products_index',
      body: {
        query: {
          multi_match: {
            query: baseQuery,
            fields: ['name^3', 'brand^2', 'category'],
            fuzziness: 'AUTO'
          }
        },
        size: 10
      }
    });

    const similarProducts = similarResponse.hits.hits.map(hit => hit._source);

    // Generate AI-powered recommendations
    const recommendations = await generateProductRecommendations(baseQuery, similarProducts);

    res.json({
      query: baseQuery,
      recommendations: recommendations,
      similar_products: similarProducts.slice(0, 5)
    });

  } catch (error) {
    console.error('Smart recommendations error:', error);
    res.status(500).json({ message: 'Failed to generate recommendations' });
  }
};

exports.getQueryInsights = async (req, res) => {
  const { q } = req.query;
  
  if (!q) {
    return res.status(400).json({ message: 'Query is required.' });
  }

  try {
    // Extract query intent and insights
    const queryIntent = await extractQueryIntent(q);
    const relatedKeywords = await getConceptualSearchKeywords(q);
    const correctedQuery = await getCorrectedSpelling(q);

    // Get search statistics
    const searchResponse = await elasticClient.search({
      index: 'products_index',
      body: {
        query: {
          multi_match: {
            query: q,
            fields: ['name', 'brand', 'category']
          }
        },
        size: 0,
        aggs: {
          categories: {
            terms: { field: 'category.keyword', size: 10 }
          },
          brands: {
            terms: { field: 'brand.keyword', size: 10 }
          },
          price_stats: {
            stats: { field: 'price' }
          }
        }
      }
    });

    const insights = {
      query_analysis: {
        original: q,
        corrected: correctedQuery,
        intent: queryIntent,
        related_keywords: relatedKeywords
      },
      search_statistics: {
        total_results: searchResponse.hits.total.value,
        categories: searchResponse.aggregations?.categories?.buckets || [],
        brands: searchResponse.aggregations?.brands?.buckets || [],
        price_range: searchResponse.aggregations?.price_stats || {}
      },
      suggestions: {
        spelling_correction: correctedQuery !== q ? correctedQuery : null,
        related_searches: relatedKeywords,
        intent_based_suggestions: queryIntent ? [
          `Search for ${queryIntent.category} products`,
          `Filter by ${queryIntent.brand || 'brand'}`,
          `Sort by ${queryIntent.priceRange || 'price'}`
        ] : []
      }
    };

    res.json(insights);

  } catch (error) {
    console.error('Query insights error:', error);
    res.status(500).json({ message: 'Failed to generate insights' });
  }
}; 