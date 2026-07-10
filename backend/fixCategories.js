const fs = require('fs');
const csv = require('csv-parser');
const { MongoClient } = require('mongodb');
require('dotenv').config();

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/flipkart_grid';

class CategoryFixer {
  constructor() {
    this.client = null;
    this.db = null;
    this.collection = null;
    this.totalProcessed = 0;
    this.totalUpdated = 0;
    this.errors = 0;
  }

  async connect() {
    try {
      this.client = new MongoClient(MONGO_URI, {
        useNewUrlParser: true,
        useUnifiedTopology: true,
      });
      await this.client.connect();
      this.db = this.client.db();
      this.collection = this.db.collection('products');
      console.log('✅ Connected to MongoDB');
    } catch (error) {
      console.error('❌ Failed to connect to MongoDB:', error);
      throw error;
    }
  }

  async close() {
    if (this.client) {
      await this.client.close();
      console.log('✅ Disconnected from MongoDB');
    }
  }

  extractCategoryFromTree(categoryTreeStr) {
    try {
      if (!categoryTreeStr) return 'General';
      
      const categoryTree = JSON.parse(categoryTreeStr);
      if (Array.isArray(categoryTree) && categoryTree.length > 0) {
        const fullPath = categoryTree[0];
        const parts = fullPath.split(' >> ');
        return parts[0] || 'General';
      }
      return 'General';
    } catch (error) {
      console.error('Error parsing category tree:', error);
      return 'General';
    }
  }

  async updateProductCategory(productId, correctCategory) {
    try {
      const result = await this.collection.updateOne(
        { productId: productId },
        { $set: { category: correctCategory } }
      );
      
      if (result.modifiedCount > 0) {
        this.totalUpdated++;
        return true;
      }
      return false;
    } catch (error) {
      console.error(`Error updating product ${productId}:`, error);
      this.errors++;
      return false;
    }
  }

  async processAndUpdate(csvPath) {
    return new Promise((resolve, reject) => {
      console.log(`📂 Reading from: ${csvPath}`);
      
      fs.createReadStream(csvPath)
        .pipe(csv())
        .on('data', async (row) => {
          this.totalProcessed++;
          
          if (this.totalProcessed % 1000 === 0) {
            console.log(`📊 Processed ${this.totalProcessed} products...`);
          }
          
          try {
            const productId = row.pid;
            const categoryTree = row.product_category_tree;
            
            if (productId && categoryTree) {
              const correctCategory = this.extractCategoryFromTree(categoryTree);
              await this.updateProductCategory(productId, correctCategory);
            }
          } catch (error) {
            console.error('Error processing row:', error);
            this.errors++;
          }
        })
        .on('end', () => {
          console.log(`✅ Finished processing ${this.totalProcessed} products`);
          console.log(`✅ Updated ${this.totalUpdated} products`);
          console.log(`❌ Errors: ${this.errors}`);
          resolve();
        })
        .on('error', (error) => {
          console.error('Error reading CSV:', error);
          reject(error);
        });
    });
  }
}

const main = async () => {
  const fixer = new CategoryFixer();
  
  try {
    console.log('🔧 Starting category fix process...');
    
    await fixer.connect();
    
    const csvPath = './data/Dataset_Final_TeamSynergyGrid.csv';
    
    if (!fs.existsSync(csvPath)) {
      console.error(`❌ CSV file not found: ${csvPath}`);
      return;
    }
    
    await fixer.processAndUpdate(csvPath);
    
    console.log('🎉 Category fix completed successfully!');
    
  } catch (error) {
    console.error('❌ Error during category fix:', error);
  } finally {
    await fixer.close();
  }
};

// Run the script
main().catch(console.error); 