const fs = require('fs');
const csv = require('csv-parser');
const { MongoClient } = require('mongodb');
require('dotenv').config();

/**
 * Insert new dataset into MongoDB
 * This script handles the Dataset_Final_TeamSynergyGrid.csv format
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
      // Parse prices
      const retailPrice = parseFloat(rawProduct.retail_price) || 0;
      const discountedPrice = parseFloat(rawProduct.discounted_price) || retailPrice;
      
      // Parse rating
      const rating = rawProduct.product_rating && rawProduct.product_rating !== 'No rating available' 
        ? parseFloat(rawProduct.product_rating) || 0 
        : 0;

      // Parse popularity
      const popularity = parseInt(rawProduct.popularity) || 0;

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

      // Generate images array (using placeholder images)
      const images = [
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=300&h=300&fit=crop&crop=center",
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=300&h=300&fit=crop&crop=center",
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=300&h=300&fit=crop&crop=center",
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=300&h=300&fit=crop&crop=center"
      ];

      // Generate name suggestions for autocomplete
      const nameSuggest = [
        { input: rawProduct.product_name, weight: 20 },
        { input: `${rawProduct.brand} in ${mainCategory}`, weight: 15 },
        { input: rawProduct.brand, weight: 10 },
        { input: `${mainCategory.toLowerCase()} under ${Math.floor(discountedPrice/1000)*1000}`, weight: 5 }
      ];

      // Create the processed product object
      const processedProduct = {
        productId: rawProduct.pid,
        name: rawProduct.product_name,
        name_suggest: nameSuggest,
        description: rawProduct.description || '',
        category: mainCategory,
        brand: rawProduct.brand || 'Unknown',
        price: discountedPrice,
        originalPrice: retailPrice,
        rating: rating,
        popularity: popularity,
        numReviews: Math.floor(popularity / 10) || 0, // Estimate based on popularity
        thumbnail: images[0],
        images: images,
        productUrl: `http://www.flipkart.com/product/${rawProduct.pid}`,
        product_specifications: rawProduct.product_specifications || '{}',
        category_tree: categoryTree,
        crawl_timestamp: new Date().toISOString(),
        is_FK_Advantage_product: false
      };

      return processedProduct;
    } catch (error) {
      console.error('❌ Error processing product:', error);
      this.errors++;
      return null;
    }
  }

  async insertInBatches(products, batchSize = 1000) {
    try {
      const batches = [];
      for (let i = 0; i < products.length; i += batchSize) {
        batches.push(products.slice(i, i + batchSize));
      }

      for (let i = 0; i < batches.length; i++) {
        const batch = batches[i];
        await this.collection.insertMany(batch);
        this.totalInserted += batch.length;
        console.log(`✅ Inserted batch ${i + 1}/${batches.length} (${this.totalInserted}/${products.length} products)`);
      }
    } catch (error) {
      console.error('❌ Error inserting batch:', error);
      throw error;
    }
  }

  async processAndInsert(csvPath) {
    return new Promise((resolve, reject) => {
      const products = [];
      
      console.log(`📂 Reading from: ${csvPath}`);
      
      fs.createReadStream(csvPath)
        .pipe(csv())
        .on('data', (row) => {
          this.totalProcessed++;
          
          if (this.totalProcessed % 5000 === 0) {
            console.log(`📊 Processed ${this.totalProcessed} products...`);
          }
          
          const processedProduct = this.processProduct(row);
          if (processedProduct) {
            products.push(processedProduct);
          }
        })
        .on('end', async () => {
          console.log(`✅ Finished reading CSV: ${products.length} valid products ready for insertion`);
          console.log(`⚠️  Processing errors: ${this.errors}`);
          
          try {
            console.log(`📦 Inserting ${products.length} products in batches of 1000...`);
            await this.insertInBatches(products);
            resolve();
          } catch (error) {
            reject(error);
          }
        })
        .on('error', (error) => {
          console.error('❌ Error reading CSV:', error);
          reject(error);
        });
    });
  }

  async createIndexes() {
    try {
      console.log('🔧 Creating database indexes...');
      
      // Create indexes for better query performance
      await this.collection.createIndex({ productId: 1 }, { unique: true });
      await this.collection.createIndex({ name: 'text', description: 'text' });
      await this.collection.createIndex({ category: 1 });
      await this.collection.createIndex({ brand: 1 });
      await this.collection.createIndex({ price: 1 });
      await this.collection.createIndex({ rating: -1 });
      await this.collection.createIndex({ popularity: -1 });
      
      console.log('✅ Database indexes created successfully!');
    } catch (error) {
      console.warn('⚠️  Warning: Failed to create some indexes:', error.message);
    }
  }

  async close() {
    if (this.client) {
      await this.client.close();
      console.log('🔌 MongoDB connection closed');
    }
  }
}

// Main execution
const main = async () => {
  const inserter = new MongoDataInsertion();
  
  try {
    console.log('🚀 Starting MongoDB data insertion for new dataset...');
    
    // Connect to MongoDB
    await inserter.connect();
    
    // Clear existing data
    await inserter.clearExistingData();
    
    // Process and insert new data
    const csvPath = process.argv[2] || '../Dataset_Final_TeamSynergyGrid.csv';
    await inserter.processAndInsert(csvPath);
    
    // Create indexes
    await inserter.createIndexes();
    
    // Get final count
    const totalProducts = await inserter.collection.countDocuments();
    const successRate = ((inserter.totalInserted / inserter.totalProcessed) * 100).toFixed(1);
    
    console.log('🎉 MongoDB insertion complete!');
    console.log(`📊 Total products in database: ${totalProducts}`);
    console.log(`📊 Success rate: ${successRate}%`);
    
  } catch (error) {
    console.error('❌ Data insertion failed:', error);
    process.exit(1);
  } finally {
    await inserter.close();
    console.log('🎉 Data insertion completed successfully!');
  }
};

// Run the script
if (require.main === module) {
  main();
}

module.exports = MongoDataInsertion; 