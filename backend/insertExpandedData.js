const fs = require('fs');
const csv = require('csv-parser');
const { MongoClient } = require('mongodb');
require('dotenv').config();

/**
 * Insert expanded dataset into MongoDB
 * This feeds the expanded data into MongoDB first, then we can index to Elasticsearch
 */

const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/flipkart-grid';
const DATABASE_NAME = 'flipkart-grid';
const COLLECTION_NAME = 'products';

class MongoDataInsertion {
  constructor() {
    this.client = null;
    this.db = null;
    this.collection = null;
    this.totalProcessed = 0;
    this.totalInserted = 0;
    this.errors = 0;
  }

  async connect() {
    try {
      console.log('🔌 Connecting to MongoDB...');
      this.client = new MongoClient(MONGO_URI);
      await this.client.connect();
      this.db = this.client.db(DATABASE_NAME);
      this.collection = this.db.collection(COLLECTION_NAME);
      console.log('✅ Connected to MongoDB successfully!');
    } catch (error) {
      console.error('❌ Failed to connect to MongoDB:', error);
      throw error;
    }
  }

  async clearExistingData() {
    try {
      console.log('🧹 Clearing existing product data...');
      const result = await this.collection.deleteMany({});
      console.log(`✅ Removed ${result.deletedCount} existing products`);
    } catch (error) {
      console.error('❌ Failed to clear existing data:', error);
      throw error;
    }
  }

  processProduct(rawProduct) {
    try {
      // Parse images array
      let images = [];
      let thumbnail = '';
      
      try {
        if (rawProduct.image) {
          images = JSON.parse(rawProduct.image);
          thumbnail = Array.isArray(images) && images.length > 0 ? images[0] : '';
        }
      } catch (error) {
        // Fallback for malformed image data
        thumbnail = rawProduct.image || '';
        images = [thumbnail];
      }

      // Parse prices
      const retailPrice = parseFloat(rawProduct.retail_price) || 0;
      const discountedPrice = parseFloat(rawProduct.discounted_price) || retailPrice;
      
      // Parse rating
      const rating = rawProduct.product_rating && rawProduct.product_rating !== 'No rating available' 
        ? parseFloat(rawProduct.product_rating) || 0 
        : 0;

      // Parse overall rating
      const overallRating = rawProduct.overall_rating && rawProduct.overall_rating !== 'No rating available'
        ? parseFloat(rawProduct.overall_rating) || rating
        : rating;

      // Extract category from category tree
      let categoryTree = [];
      let mainCategory = 'General';
      
      try {
        if (rawProduct.product_category_tree) {
          categoryTree = JSON.parse(rawProduct.product_category_tree);
          if (Array.isArray(categoryTree) && categoryTree.length > 0) {
            const fullPath = categoryTree[0];
            const parts = fullPath.split(' >> ');
            mainCategory = parts[0] || 'General';
          }
        }
      } catch (error) {
        categoryTree = ['General'];
        mainCategory = 'General';
      }

      // Generate numReviews based on rating (more realistic)
      let numReviews = 0;
      if (rating > 0) {
        // Products with higher ratings tend to have more reviews
        const baseReviews = Math.floor(Math.random() * 100) + 1;
        const ratingMultiplier = rating / 5;
        numReviews = Math.floor(baseReviews * ratingMultiplier * (1 + Math.random()));
      }

      // Calculate a popularity score
      const popularity = (rating * 10) + numReviews;

      return {
        productId: rawProduct.uniq_id, // Use uniq_id as the unique productId
        pid: rawProduct.uniq_id, // Ensure pid is also unique
        product_name: rawProduct.product_name?.trim() || 'Unknown Product',
        product_category_tree: categoryTree,
        retail_price: retailPrice,
        discounted_price: discountedPrice,
        image: images,
        is_FK_Advantage_product: rawProduct.is_FK_Advantage_product === 'True',
        description: rawProduct.description || `${rawProduct.product_name} by ${rawProduct.brand}`,
        product_rating: rating,
        overall_rating: overallRating,
        brand: rawProduct.brand?.trim() || 'Unknown Brand',
        product_specifications: rawProduct.product_specifications || '{}',
        numReviews: numReviews,
        popularity: popularity,
        
        // Additional fields for better search
        category: mainCategory,
        thumbnail: thumbnail,
        uniq_id: rawProduct.uniq_id || rawProduct.pid,
        crawl_timestamp: rawProduct.crawl_timestamp || new Date().toISOString(),
        product_url: rawProduct.product_url || '',
        
        // Metadata
        createdAt: new Date(),
        updatedAt: new Date()
      };
    } catch (error) {
      console.error('❌ Error processing product:', error.message);
      return null;
    }
  }

  async insertInBatches(products, batchSize = 1000) {
    console.log(`📦 Inserting ${products.length} products in batches of ${batchSize}...`);
    
    for (let i = 0; i < products.length; i += batchSize) {
      const batch = products.slice(i, i + batchSize);
      
      try {
        const result = await this.collection.insertMany(batch, { ordered: false });
        this.totalInserted += result.insertedCount;
        
        const batchNum = Math.floor(i / batchSize) + 1;
        const totalBatches = Math.ceil(products.length / batchSize);
        console.log(`✅ Inserted batch ${batchNum}/${totalBatches} (${this.totalInserted}/${products.length} products)`);
        
      } catch (error) {
        this.errors++;
        console.error(`❌ Error inserting batch ${Math.floor(i / batchSize) + 1}:`, error.message);
        
        // Try inserting individually for this batch to identify problematic records
        for (const product of batch) {
          try {
            await this.collection.insertOne(product);
            this.totalInserted++;
          } catch (individualError) {
            console.error(`❌ Failed to insert product ${product.pid}:`, individualError.message);
          }
        }
      }
    }
  }

  async processAndInsert(csvPath) {
    try {
      console.log('🚀 Starting MongoDB data insertion...');
      console.log(`📂 Reading from: ${csvPath}`);

      await this.connect();
      await this.clearExistingData();

      const products = [];

      // Read and process CSV
      await new Promise((resolve, reject) => {
        fs.createReadStream(csvPath)
          .pipe(csv())
          .on('data', (data) => {
            this.totalProcessed++;
            
            const processedProduct = this.processProduct(data);
            if (processedProduct && processedProduct.productId) {
              products.push(processedProduct);
            } else {
              this.errors++;
            }
            
            if (this.totalProcessed % 5000 === 0) {
              console.log(`📊 Processed ${this.totalProcessed} products, ${products.length} valid...`);
            }
          })
          .on('end', resolve)
          .on('error', reject);
      });

      console.log(`✅ Finished processing CSV: ${products.length} valid products ready for insertion`);
      console.log(`⚠️  Processing errors: ${this.errors}`);

      if (products.length === 0) {
        console.log('❌ No valid products to insert!');
        return;
      }

      // Insert products in batches
      await this.insertInBatches(products);

      // Verify insertion
      const totalCount = await this.collection.countDocuments();
      console.log(`🎉 MongoDB insertion complete!`);
      console.log(`📊 Total products in database: ${totalCount}`);
      console.log(`📊 Success rate: ${((this.totalInserted / this.totalProcessed) * 100).toFixed(1)}%`);

      // Create indexes for better performance
      await this.createIndexes();

    } catch (error) {
      console.error('❌ MongoDB insertion failed:', error);
      throw error;
    } finally {
      if (this.client) {
        await this.client.close();
        console.log('🔌 MongoDB connection closed');
      }
    }
  }

  async createIndexes() {
    try {
      console.log('🔧 Creating database indexes...');
      
      await this.collection.createIndex({ product_name: 'text', description: 'text', brand: 'text' });
      await this.collection.createIndex({ brand: 1 });
      await this.collection.createIndex({ category: 1 });
      await this.collection.createIndex({ discounted_price: 1 });
      await this.collection.createIndex({ product_rating: -1 });
      await this.collection.createIndex({ pid: 1 }, { unique: true });
      
      console.log('✅ Database indexes created successfully');
    } catch (error) {
      console.error('⚠️  Warning: Failed to create some indexes:', error.message);
    }
  }
}

// CLI usage
if (require.main === module) {
  const csvPath = process.argv[2] || './data/flipkart_expanded_clean.csv';
  
  const inserter = new MongoDataInsertion();
  
  inserter.processAndInsert(csvPath)
    .then(() => {
      console.log('🎉 Data insertion completed successfully!');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Data insertion failed:', error);
      process.exit(1);
    });
}

module.exports = MongoDataInsertion;
