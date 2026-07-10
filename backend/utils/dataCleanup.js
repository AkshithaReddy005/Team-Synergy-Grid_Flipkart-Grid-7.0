const fs = require('fs');
const csv = require('csv-parser');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

/**
 * Data cleanup utility for Flipkart dataset
 * Filters out products with missing essential data and handles broken images
 */

class DataCleanup {
  constructor() {
    this.cleanedData = [];
    this.skippedCount = 0;
    this.totalCount = 0;
    
    // Realistic fallback images using Unsplash for different categories and products
    this.placeholderImages = {
      'clothing': 'https://images.unsplash.com/photo-1445205170230-053b83016050?w=300&h=300&fit=crop&crop=center',
      'footwear': 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=300&h=300&fit=crop&crop=center',
      'electronics': 'https://images.unsplash.com/photo-1498049794561-7780e7231661?w=300&h=300&fit=crop&crop=center',
      'furniture': 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=300&h=300&fit=crop&crop=center',
      'beauty': 'https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=300&h=300&fit=crop&crop=center',
      'sports': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=300&h=300&fit=crop&crop=center',
      'books': 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300&h=300&fit=crop&crop=center',
      'default': 'https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=300&h=300&fit=crop&crop=center'
    };
    
    // Product-specific fallback images
    this.productSpecificImages = {
      'bag': 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=300&fit=crop&crop=center',
      'purse': 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=300&fit=crop&crop=center',
      'handbag': 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=300&fit=crop&crop=center',
      'sling': 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=300&fit=crop&crop=center',
      'shoe': 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=300&h=300&fit=crop&crop=center',
      'sandal': 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=300&h=300&fit=crop&crop=center',
      'shirt': 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=300&h=300&fit=crop&crop=center',
      'dress': 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=300&h=300&fit=crop&crop=center',
      'watch': 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=300&h=300&fit=crop&crop=center',
      'phone': 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=300&h=300&fit=crop&crop=center',
      'laptop': 'https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=300&h=300&fit=crop&crop=center',
      'headphone': 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=300&h=300&fit=crop&crop=center'
    };
  }

  /**
   * Determines category from product category tree
   */
  getCategoryFromTree(categoryTree) {
    if (!categoryTree) return 'default';
    
    const tree = categoryTree.toLowerCase();
    if (tree.includes('clothing') || tree.includes('apparel')) return 'clothing';
    if (tree.includes('footwear') || tree.includes('shoes')) return 'footwear';
    if (tree.includes('electronics') || tree.includes('mobile')) return 'electronics';
    if (tree.includes('furniture') || tree.includes('home')) return 'furniture';
    if (tree.includes('beauty') || tree.includes('cosmetics')) return 'beauty';
    if (tree.includes('sports') || tree.includes('fitness')) return 'sports';
    if (tree.includes('books') || tree.includes('media')) return 'books';
    
    return 'default';
  }

  /**
   * Validates if a product has essential fields
   */
  isValidProduct(product) {
    // Essential fields that must be present and non-empty
    const essentialFields = [
      'product_name',
      'retail_price',
      'discounted_price',
      'brand'
    ];

    for (const field of essentialFields) {
      if (!product[field] || 
          product[field].trim() === '' || 
          product[field] === 'null' ||
          product[field] === 'undefined') {
        return false;
      }
    }

    // Validate prices are numeric
    const retailPrice = parseFloat(product.retail_price);
    const discountedPrice = parseFloat(product.discounted_price);
    
    if (isNaN(retailPrice) || isNaN(discountedPrice) || 
        retailPrice <= 0 || discountedPrice <= 0) {
      return false;
    }

    return true;
  }

  /**
   * Get realistic fallback image based on product name and category
   */
  getRealisticFallback(productName, categoryTree) {
    const name = (productName || '').toLowerCase();
    const category = this.getCategoryFromTree(categoryTree);
    
    // Check for specific product types in the name
    for (const [productType, imageUrl] of Object.entries(this.productSpecificImages)) {
      if (name.includes(productType)) {
        return imageUrl;
      }
    }
    
    // Fall back to category-based image
    return this.placeholderImages[category] || this.placeholderImages.default;
  }

  /**
   * Cleans and processes image URLs
   */
  processImages(imageString, categoryTree, productName = '') {
    if (!imageString || imageString.trim() === '') {
      return JSON.stringify([this.getRealisticFallback(productName, categoryTree)]);
    }

    try {
      // Parse the image array string
      let images = JSON.parse(imageString);
      if (!Array.isArray(images)) {
        images = [imageString];
      }

      // Filter out empty or invalid URLs and replace with placeholders
      const validImages = images.filter(img => 
        img && img.trim() !== '' && img.startsWith('http')
      );

      if (validImages.length === 0) {
        return JSON.stringify([this.getRealisticFallback(productName, categoryTree)]);
      }

      // Replace old Flipkart URLs with realistic fallbacks
      const processedImages = validImages.map(img => {
        if (img.includes('flixcart.com') || img.includes('flipkart.com')) {
          return this.getRealisticFallback(productName, categoryTree);
        }
        return img;
      });

      return JSON.stringify(processedImages);
    } catch (error) {
      return JSON.stringify([this.getRealisticFallback(productName, categoryTree)]);
    }
  }

  /**
   * Processes rating fields
   */
  processRating(rating) {
    if (!rating || rating === 'No rating available' || rating.trim() === '') {
      return '0';
    }
    
    const numRating = parseFloat(rating);
    return isNaN(numRating) ? '0' : numRating.toString();
  }

  /**
   * Cleans a single product record
   */
  cleanProduct(product) {
    if (!this.isValidProduct(product)) {
      this.skippedCount++;
      return null;
    }

    return {
      ...product,
      image: this.processImages(product.image, product.product_category_tree, product.product_name),
      product_rating: this.processRating(product.product_rating),
      overall_rating: this.processRating(product.overall_rating),
      // Ensure description exists
      description: product.description || `${product.product_name} by ${product.brand}`,
      // Clean product name
      product_name: product.product_name.trim(),
      // Ensure category exists
      product_category_tree: product.product_category_tree || '["General"]'
    };
  }

  /**
   * Processes the entire CSV file
   */
  async processCSV(inputPath, outputPath) {
    console.log('🧹 Starting data cleanup process...');
    console.log(`📂 Input: ${inputPath}`);
    console.log(`📂 Output: ${outputPath}`);

    return new Promise((resolve, reject) => {
      const results = [];
      
      fs.createReadStream(inputPath)
        .pipe(csv())
        .on('data', (data) => {
          this.totalCount++;
          const cleanedProduct = this.cleanProduct(data);
          if (cleanedProduct) {
            results.push(cleanedProduct);
          }
          
          // Progress indicator
          if (this.totalCount % 1000 === 0) {
            console.log(`📊 Processed ${this.totalCount} products, kept ${results.length}, skipped ${this.skippedCount}`);
          }
        })
        .on('end', async () => {
          console.log(`✅ Processing complete!`);
          console.log(`📊 Total products: ${this.totalCount}`);
          console.log(`✅ Valid products: ${results.length}`);
          console.log(`❌ Skipped products: ${this.skippedCount}`);
          console.log(`📈 Data quality: ${((results.length / this.totalCount) * 100).toFixed(1)}%`);

          // Write cleaned data to new CSV
          if (results.length > 0) {
            const csvWriter = createCsvWriter({
              path: outputPath,
              header: Object.keys(results[0]).map(key => ({ id: key, title: key }))
            });

            try {
              await csvWriter.writeRecords(results);
              console.log(`💾 Cleaned data saved to: ${outputPath}`);
              resolve({
                totalCount: this.totalCount,
                validCount: results.length,
                skippedCount: this.skippedCount,
                outputPath: outputPath
              });
            } catch (error) {
              reject(error);
            }
          } else {
            reject(new Error('No valid products found after cleanup'));
          }
        })
        .on('error', reject);
    });
  }
}

module.exports = DataCleanup;

// CLI usage
if (require.main === module) {
  const cleanup = new DataCleanup();
  const inputPath = process.argv[2] || './data/flipkart.csv';
  const outputPath = process.argv[3] || './data/flipkart_cleaned.csv';
  
  cleanup.processCSV(inputPath, outputPath)
    .then(result => {
      console.log('🎉 Data cleanup completed successfully!');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Error during cleanup:', error);
      process.exit(1);
    });
}
