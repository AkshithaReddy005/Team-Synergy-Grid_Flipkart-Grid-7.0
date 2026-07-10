require('dotenv').config();
const { Client } = require('@elastic/elasticsearch');
const { searchFallback } = require('./services/fallbackSearchService');

const elasticClient = new Client({
  node: process.env.ELASTIC_NODE || 'http://127.0.0.1:9200'
});

// Monkey patch the search function for the interview fallback
const originalSearch = elasticClient.search.bind(elasticClient);

elasticClient.search = async (params, options) => {
  try {
    return await originalSearch(params, options);
  } catch (error) {
    if (error.name === 'ConnectionError' || error.message.includes('ConnectionError')) {
      console.log('⚠️ [Elasticsearch] Connection failed. Using zero-dependency CSV fallback search!');
      // Very basic extraction of search query from the Elasticsearch DSL body
      let queryStr = '';
      try {
        const body = params.body || params;
        if (body.query && body.query.bool && body.query.bool.must) {
          const multiMatch = body.query.bool.must.find(m => m.multi_match);
          if (multiMatch) queryStr = multiMatch.multi_match.query;
        }
      } catch(e) {}
      
      const fallbackResult = await searchFallback(queryStr, {
        size: params.body?.size || 100,
        sortBy: 'popularity'
      });
      return fallbackResult;
    }
    throw error;
  }
};

module.exports = elasticClient;
