const mongoose = require('mongoose');
const elasticClient = require('./elasticClient');
const Product = require('./models/Product');
require('dotenv').config();

const indexName = 'products_index';

if (require.main === module) {
  mongoose.connect(process.env.MONGO_URI)
    .then(() => {
      console.log('[indexData.js]: Connected to MongoDB');
      indexData();
    })
    .catch(err => {
      console.error('[indexData.js]: MongoDB connection error:', err);
      process.exit(1);
    });
}

const deleteIndexIfExists = async () => {
  const exists = await elasticClient.indices.exists({ index: indexName });
  if (exists) { // v8 client returns a boolean
    await elasticClient.indices.delete({ index: indexName });
    console.log(`[indexData.js]: Previous index '${indexName}' deleted.`);
  }
};

const createIndexWithMapping = async () => {
  console.log(`[indexData.js]: Creating new index '${indexName}'...`);
  await elasticClient.indices.create({
    index: indexName,
    mappings: { // 'body' wrapper is deprecated in v8
      properties: {
        name: { type: 'text' },
        name_suggest: { type: 'completion' },
        description: { type: 'text' },
        category: { type: 'keyword' },
        brand: { type: 'keyword' },
        price: { type: 'float' },
        rating: { type: 'float' },
        numReviews: { type: 'integer' },
        imageUrl: { type: 'text' }
      }
    }
  });
  console.log(`[indexData.js]: Index '${indexName}' created successfully.`);
};

const indexData = async () => {
  try {
    await deleteIndexIfExists();
    await createIndexWithMapping();

    const products = await Product.find({});
    console.log(`[indexData.js]: Found ${products.length} products to index.`);

    if (!products.length) {
      console.log('[indexData.js]: No products found to index.');
      return;
    }

    const chunkSize = 100;
    for (let i = 0; i < products.length; i += chunkSize) {
      const chunk = products.slice(i, i + chunkSize);

      const body = chunk.flatMap(doc => {
        const product = doc.toObject();
        return [
          { index: { _index: indexName, _id: product._id.toString() } },
          {
            name: product.name,
            name_suggest: { input: [product.name] },
            description: product.description,
            category: product.category,
            brand: product.brand,
            price: product.price,
            rating: product.rating,
            numReviews: product.numReviews || 0,
            imageUrl: product.thumbnail || ''
          }
        ];
      });

      const bulkResponse = await elasticClient.bulk({ refresh: true, body });

      if (bulkResponse.errors) {
        console.error(`[indexData.js]: Failed indexing chunk ${Math.floor(i / chunkSize) + 1}`);
      } else {
        console.log(`[indexData.js]: Successfully indexed chunk ${Math.floor(i / chunkSize) + 1}`);
      }
    }

    console.log('[indexData.js]: Finished indexing all products.');
  } catch (error) {
    console.error('[indexData.js]: Indexing failed:', error);
  } finally {
    if (require.main === module) {
      await mongoose.disconnect();
      console.log('[indexData.js]: MongoDB disconnected after indexing.');
    }
  }
};

module.exports = indexData;
