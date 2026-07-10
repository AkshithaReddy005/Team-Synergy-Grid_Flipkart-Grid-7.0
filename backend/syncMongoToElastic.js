const mongoose = require('mongoose');
const elasticClient = require('./elasticClient');
const Product = require('./models/Product.js');
require('dotenv').config();

const indexName = 'products_index';

const deleteIndexIfExists = async () => {
  console.log(`Checking for existing index '${indexName}'...`);
  try {
    const exists = await elasticClient.indices.exists({ index: indexName });
    if (exists) {
      console.log(`Deleting previous index '${indexName}'...`);
      await elasticClient.indices.delete({ index: indexName });
      console.log(`✅ Previous index deleted.`);
    }
  } catch (error) {
    console.error(`Error deleting index: ${error.message}`);
    // Don't exit, as we might just need to create it
  }
};

const createIndexWithMapping = async () => {
  console.log(`🔧 Creating new index '${indexName}' with mapping...`);
  try {
    await elasticClient.indices.create({
      index: indexName,
      body: {
        settings: {
          analysis: {
            analyzer: {
              english_stemmer: {
                tokenizer: 'standard',
                filter: ['lowercase', 'english_stop', 'english_stemmer_filter']
              }
            },
            filter: {
              english_stop: {
                type: 'stop',
                stopwords: '_english_'
              },
              english_stemmer_filter: {
                type: 'stemmer',
                language: 'english'
              }
            }
          }
        },
        mappings: {
          properties: {
            name: { type: 'text', analyzer: 'english_stemmer' },
            description: { type: 'text', analyzer: 'english_stemmer' },
            brand: { 
              type: 'text',
              analyzer: 'english_stemmer',
              fields: { keyword: { type: 'keyword' } }
            },
            category: { 
              type: 'text',
              analyzer: 'english_stemmer',
              fields: { keyword: { type: 'keyword' } }
            },
            price: { type: 'float' },
            originalPrice: { type: 'float' },
            rating: { type: 'float' },
            popularity: { type: 'integer' },
            numReviews: { type: 'integer' },
            thumbnail: { type: 'text' },
            images: { type: 'text' },
            productId: { type: 'keyword' },
            productUrl: { type: 'text' },
            name_suggest: {
              type: 'completion',
              analyzer: 'standard',
              search_analyzer: 'standard'
            }
          }
        }
      }
    });
    console.log(`✅ Index '${indexName}' created successfully.`);
  } catch (error) {
    console.error(`❌ Error creating index: ${error.message}`);
    throw error; // Stop execution if mapping fails
  }
};

const syncMongoToElastic = async () => {
  console.log('🚀 Starting MongoDB to Elasticsearch sync...');

  try {
    // Connect to MongoDB
    console.log('🔌 Connecting to MongoDB...');
    await mongoose.connect(process.env.MONGO_URI);
    console.log('✅ Connected to MongoDB successfully!');

    // Re-create Elasticsearch index
    await deleteIndexIfExists();
    await createIndexWithMapping();

    // Fetch all products from MongoDB
    console.log('🔍 Fetching all products from MongoDB...');
    const allProducts = await Product.find({}).lean();
    console.log(`✅ Found ${allProducts.length} products to index.`);

    if (allProducts.length === 0) {
      console.log('⚠️ No products found in MongoDB. Nothing to index.');
      return;
    }

    // Prepare bulk indexing payload
    const body = allProducts.flatMap(doc => [
      { index: { _index: indexName, _id: doc.productId } },
      {
        ...doc,
        name_suggest: {
            input: [doc.name, doc.brand, doc.category].filter(Boolean),
            weight: 10
        }
      }
    ]);

    // Perform bulk indexing
    console.log(`📦 Bulk indexing ${allProducts.length} documents...`);
    const { body: bulkResponse } = await elasticClient.bulk({ refresh: true, body });

    if (bulkResponse.errors) {
      console.error('❌ Bulk indexing completed with errors.');
      const erroredDocuments = [];
      bulkResponse.items.forEach((action, i) => {
        const operation = Object.keys(action)[0];
        if (action[operation].error) {
          erroredDocuments.push({
            status: action[operation].status,
            error: action[operation].error,
            operation: body[i * 2],
            document: body[i * 2 + 1]
          });
        }
      });
      console.log(JSON.stringify(erroredDocuments, null, 2));
    } else {
      console.log('✅ Bulk indexing completed successfully.');
    }

  } catch (error) {
    console.error('An error occurred during the sync process:', error);
  } finally {
    await mongoose.connection.close();
    console.log('🔌 MongoDB connection closed.');
  }
};

syncMongoToElastic();
