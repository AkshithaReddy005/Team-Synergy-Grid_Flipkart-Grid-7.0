const elasticClient = require('../elasticClient');

// Simple test search controller to isolate Elasticsearch issues
exports.testSearch = async (req, res) => {
  const { q } = req.query;
  
  if (!q) {
    return res.status(400).json({ error: "Query is required" });
  }

  try {
    console.log(`Testing search for: "${q}"`);
    
    // Test 1: Ultra-simple match query
    const response = await elasticClient.search({
      index: 'products_index',
      body: {
        query: {
          match: {
            name: q
          }
        },
        size: 5
      }
    });

    console.log('Search successful!');
    console.log('Total hits:', response.hits.total.value);
    
    const results = response.hits.hits.map(hit => ({
      id: hit._id,
      name: hit._source.name,
      brand: hit._source.brand,
      category: hit._source.category,
      score: hit._score
    }));

    res.json({
      success: true,
      query: q,
      total: response.hits.total.value,
      results: results
    });

  } catch (error) {
    console.error('Test search error:', error);
    res.status(500).json({ 
      success: false,
      error: error.message,
      details: error.meta?.body || error
    });
  }
};
