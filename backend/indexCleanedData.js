const fs = require('fs');
const csv = require('csv-parser');
const elasticClient = require('./elasticClient');
require('dotenv').config();

const indexName = 'products_index';

/**
 * Index cleaned CSV data directly to Elasticsearch
 * This bypasses MongoDB and uses the cleaned dataset
 */

const deleteIndexIfExists = async () => {
  try {
    const exists = await elasticClient.indices.exists({ index: indexName });
    if (exists) {
      await elasticClient.indices.delete({ index: indexName });
      console.log(`✅ Previous index '${indexName}' deleted.`);
    }
  } catch (error) {
    console.log(`ℹ️  Index '${indexName}' doesn't exist, creating new one.`);
  }
};

const createIndexWithMapping = async () => {
  console.log(`🔧 Creating new index '${indexName}'...`);
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
          },
          suggest: {
            type: 'completion',
            analyzer: 'standard',
            search_analyzer: 'standard'
          }
        }
      }
    }
  });
  console.log(`✅ Index '${indexName}' created successfully.`);
};

const processProduct = (rawProduct) => {
  // --- Data Parsing ---
  let images = [];
  let thumbnail = '';
  try {
    if (rawProduct.image) {
      images = JSON.parse(rawProduct.image);
      thumbnail = Array.isArray(images) && images.length > 0 ? images[0] : '';
    }
  } catch (error) {
    thumbnail = rawProduct.image || '';
    images = [thumbnail];
  }

  const originalPrice = parseFloat(rawProduct.retail_price) || 0;
  const price = parseFloat(rawProduct.discounted_price) || originalPrice;
  const rating = rawProduct.product_rating && rawProduct.product_rating !== 'No rating available' 
    ? parseFloat(rawProduct.product_rating) || 0 
    : 0;
  const popularity = parseInt(rawProduct.popularity, 10) || 0;

  let category = 'General';
  let primaryCategory = 'Products'; // A top-level category
  try {
    if (rawProduct.product_category_tree) {
      const categoryTree = JSON.parse(rawProduct.product_category_tree);
      if (Array.isArray(categoryTree) && categoryTree.length > 0) {
        const fullPath = categoryTree[0];
        const parts = fullPath.split(' >> ');
        primaryCategory = parts[0] || 'Products';
        category = parts[parts.length - 1] || 'General';
      }
    }
  } catch (error) {
    category = 'General';
    primaryCategory = 'Products';
  }

  const name = rawProduct.product_name?.trim() || 'Unknown Product';
  const brand = rawProduct.brand?.trim() || 'Unknown Brand';

  // --- Suggestion Generation with Weights ---
  const suggestions = [];
  // Using a Set to ensure inputs are unique before adding with weights
  const uniqueInputs = new Set();

  const addSuggestion = (input, weight) => {
    if (input && !uniqueInputs.has(input)) {
      suggestions.push({ input, weight });
      uniqueInputs.add(input);
    }
  };

  // Add suggestions with priority
  addSuggestion(name, 20); // Priority 1: Exact Product Name
  if (brand !== 'Unknown Brand') {
    addSuggestion(`${brand} in ${primaryCategory}`, 15); // Priority 2: Brand in Category
    addSuggestion(brand, 10); // Priority 3: Brand
  }
  if (price > 0) {
    const priceCeiling = Math.ceil(price / 500) * 500;
    addSuggestion(`${primaryCategory.toLowerCase()} under ${priceCeiling}`, 5); // Priority 4: Category under Price
  }
  
  return {
    name: name,
    name_suggest: suggestions, // Pass the array of weighted suggestion objects
    description: rawProduct.description || `${name} by ${brand}`,
    category: category,
    brand: brand,
    price: price,
    originalPrice: originalPrice,
    rating: rating,
    popularity: popularity,
    numReviews: 0,
    thumbnail: thumbnail,
    images: images,
    productId: rawProduct.pid || rawProduct.uniq_id || '',
    productUrl: rawProduct.product_url || ''
  };
};

const indexCleanedData = async (csvPath = './data/flipkart_expanded_clean.csv') => {
  try {
    console.log('🚀 Starting indexing of cleaned data...');
    console.log(`📂 Reading from: ${csvPath}`);

    await deleteIndexIfExists();
    await createIndexWithMapping();

    const products = [];
    let totalCount = 0;
    let errorCount = 0;

    // Read and process CSV
    await new Promise((resolve, reject) => {
      fs.createReadStream(csvPath)
        .pipe(csv())
        .on('data', (data) => {
          try {
            totalCount++;
            const processedProduct = processProduct(data);
            products.push(processedProduct);
            
            if (totalCount % 1000 === 0) {
              console.log(`📊 Processed ${totalCount} products...`);
            }
          } catch (error) {
            errorCount++;
            console.error(`❌ Error processing product ${totalCount}:`, error.message);
          }
        })
        .on('end', resolve)
        .on('error', reject);
    });

    console.log(`✅ Finished reading CSV: ${products.length} products ready for indexing`);
    console.log(`⚠️  Processing errors: ${errorCount}`);

    if (products.length === 0) {
      console.log('❌ No products to index!');
      return;
    }

    // Index in chunks
    const chunkSize = 100;
    let indexedCount = 0;

    for (let i = 0; i < products.length; i += chunkSize) {
      const chunk = products.slice(i, i + chunkSize);
      
      const body = chunk.flatMap(product => [
        { index: { _index: indexName, _id: product.productId || `product_${i + chunk.indexOf(product)}` } },
        product
      ]);

      try {
        const bulkResponse = await elasticClient.bulk({ refresh: true, body });
        
        if (bulkResponse.errors) {
          console.error(`❌ Errors in chunk ${Math.floor(i / chunkSize) + 1}:`, 
            bulkResponse.items.filter(item => item.index.error).map(item => item.index.error));
        } else {
          indexedCount += chunk.length;
          console.log(`✅ Indexed chunk ${Math.floor(i / chunkSize) + 1}/${Math.ceil(products.length / chunkSize)} (${indexedCount}/${products.length} products)`);
        }
      } catch (error) {
        console.error(`❌ Failed to index chunk ${Math.floor(i / chunkSize) + 1}:`, error.message);
      }
    }

    // Verify indexing
    const indexStats = await elasticClient.count({ index: indexName });
    console.log(`🎉 Indexing complete!`);
    console.log(`📊 Total products in index: ${indexStats.count}`);
    console.log(`📊 Success rate: ${((indexedCount / products.length) * 100).toFixed(1)}%`);

  } catch (error) {
    console.error('❌ Indexing failed:', error);
    throw error;
  }
};

// CLI usage
if (require.main === module) {
  const csvPath = process.argv[2] || '../flipkart_expanded_clean.csv';
  
  indexCleanedData(csvPath)
    .then(() => {
      console.log('🎉 Indexing completed successfully!');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Indexing failed:', error);
      process.exit(1);
    });
}

module.exports = indexCleanedData;
