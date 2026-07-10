const fs = require('fs');
const csv = require('csv-parser');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

/**
 * Dataset Expansion Utility
 * Expands the product dataset by creating realistic variations and processing the full dataset
 */

class DatasetExpansion {
  constructor() {
    this.expandedData = [];
    this.totalProcessed = 0;
    this.totalGenerated = 0;
    
    // Brand variations for different categories
    this.brandVariations = {
      'clothing': ['Nike', 'Adidas', 'Puma', 'H&M', 'Zara', 'Uniqlo', 'Forever21', 'Levis', 'Tommy Hilfiger', 'Calvin Klein'],
      'footwear': ['Nike', 'Adidas', 'Puma', 'Reebok', 'Converse', 'Vans', 'New Balance', 'Sketchers', 'Crocs', 'Bata'],
      'electronics': ['Samsung', 'Apple', 'OnePlus', 'Xiaomi', 'Sony', 'LG', 'HP', 'Dell', 'Lenovo', 'Asus'],
      'beauty': ['Lakme', 'Maybelline', 'LOreal', 'MAC', 'Nykaa', 'Revlon', 'Colorbar', 'Faces', 'Lotus', 'Biotique'],
      'furniture': ['IKEA', 'Godrej', 'Durian', 'Urban Ladder', 'Pepperfry', 'HomeTown', 'Nilkamal', 'Zuari', 'Spacewood', 'Perfect Homes'],
      'books': ['Penguin', 'Harper Collins', 'Scholastic', 'Oxford', 'Cambridge', 'McGraw Hill', 'Pearson', 'Wiley', 'Springer', 'Elsevier']
    };
    
    // Color variations
    this.colors = ['Black', 'White', 'Red', 'Blue', 'Green', 'Yellow', 'Pink', 'Purple', 'Orange', 'Brown', 'Grey', 'Navy', 'Maroon', 'Beige', 'Gold', 'Silver'];
    
    // Size variations
    this.sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', '28', '30', '32', '34', '36', '38', '40', '42', '6', '7', '8', '9', '10', '11', '12'];
    
    // Product type variations
    this.productTypes = {
      'bag': ['Handbag', 'Shoulder Bag', 'Tote Bag', 'Clutch', 'Sling Bag', 'Backpack', 'Messenger Bag', 'Laptop Bag', 'Travel Bag', 'Gym Bag'],
      'shirt': ['T-Shirt', 'Polo Shirt', 'Formal Shirt', 'Casual Shirt', 'Tank Top', 'Henley', 'V-Neck', 'Round Neck', 'Button Down', 'Oxford Shirt'],
      'shoe': ['Sneakers', 'Running Shoes', 'Casual Shoes', 'Formal Shoes', 'Boots', 'Sandals', 'Flip Flops', 'Loafers', 'High Heels', 'Flats'],
      'phone': ['Smartphone', 'Feature Phone', 'Gaming Phone', 'Camera Phone', 'Business Phone', '5G Phone', 'Dual SIM Phone', 'Rugged Phone'],
      'watch': ['Smart Watch', 'Analog Watch', 'Digital Watch', 'Sports Watch', 'Luxury Watch', 'Casual Watch', 'Formal Watch', 'Fitness Tracker']
    };
  }

  /**
   * Get category from product category tree
   */
  getCategoryFromTree(categoryTree) {
    if (!categoryTree) return 'default';
    
    const tree = categoryTree.toLowerCase();
    if (tree.includes('clothing') || tree.includes('apparel')) return 'clothing';
    if (tree.includes('footwear') || tree.includes('shoes')) return 'footwear';
    if (tree.includes('electronics') || tree.includes('mobile')) return 'electronics';
    if (tree.includes('furniture') || tree.includes('home')) return 'furniture';
    if (tree.includes('beauty') || tree.includes('cosmetics')) return 'beauty';
    if (tree.includes('books') || tree.includes('media')) return 'books';
    
    return 'default';
  }

  /**
   * Generate realistic price variations
   */
  generatePriceVariations(basePrice) {
    const variations = [];
    const base = parseFloat(basePrice) || 500;
    
    // Generate 3-5 price variations (different sizes, colors, etc.)
    for (let i = 0; i < Math.floor(Math.random() * 3) + 3; i++) {
      const variation = base + (Math.random() - 0.5) * base * 0.3; // ±30% variation
      variations.push(Math.max(50, Math.round(variation))); // Minimum ₹50
    }
    
    return variations;
  }

  /**
   * Generate product variations
   */
  generateProductVariations(baseProduct) {
    const variations = [];
    const category = this.getCategoryFromTree(baseProduct.product_category_tree);
    const productName = baseProduct.product_name.toLowerCase();
    
    // Determine product type
    let productType = 'default';
    for (const [type, _] of Object.entries(this.productTypes)) {
      if (productName.includes(type)) {
        productType = type;
        break;
      }
    }
    
    // Generate 3-8 variations per product
    const numVariations = Math.floor(Math.random() * 6) + 3;
    const priceVariations = this.generatePriceVariations(baseProduct.discounted_price);
    
    for (let i = 0; i < numVariations; i++) {
      const variation = { ...baseProduct };
      
      // Vary the brand
      const brands = this.brandVariations[category] || ['Generic', 'Brand', 'Premium', 'Quality', 'Style'];
      variation.brand = brands[Math.floor(Math.random() * brands.length)];
      
      // Vary colors and sizes in product name
      const color = this.colors[Math.floor(Math.random() * this.colors.length)];
      const size = this.sizes[Math.floor(Math.random() * this.sizes.length)];
      
      // Vary product type if applicable
      if (this.productTypes[productType]) {
        const types = this.productTypes[productType];
        const newType = types[Math.floor(Math.random() * types.length)];
        variation.product_name = `${variation.brand} ${color} ${newType} ${size}`;
      } else {
        variation.product_name = `${variation.brand} ${color} ${baseProduct.product_name} ${size}`;
      }
      
      // Vary prices
      const priceIndex = i % priceVariations.length;
      variation.discounted_price = priceVariations[priceIndex];
      variation.retail_price = Math.round(priceVariations[priceIndex] * (1 + Math.random() * 0.5)); // 0-50% markup
      
      // Generate unique product ID
      variation.pid = `${baseProduct.pid}_VAR_${i + 1}`;
      variation.uniq_id = `${baseProduct.uniq_id}_${i + 1}`;
      
      // Vary ratings slightly
      const baseRating = parseFloat(baseProduct.product_rating) || 0;
      if (baseRating > 0) {
        variation.product_rating = Math.max(1, Math.min(5, baseRating + (Math.random() - 0.5) * 1)).toFixed(1);
        variation.overall_rating = variation.product_rating;
      }
      
      // Update description
      variation.description = `${variation.product_name} by ${variation.brand}. Available in ${color} color, size ${size}. ${baseProduct.description || 'High quality product with excellent features.'}`;
      
      variations.push(variation);
    }
    
    return variations;
  }

  /**
   * Process and expand the dataset
   */
  async processAndExpand(inputPath, outputPath, expansionFactor = 5) {
    console.log('🚀 Starting dataset expansion...');
    console.log(`📂 Input: ${inputPath}`);
    console.log(`📂 Output: ${outputPath}`);
    console.log(`📈 Expansion Factor: ${expansionFactor}x`);

    return new Promise((resolve, reject) => {
      const results = [];
      
      fs.createReadStream(inputPath)
        .pipe(csv())
        .on('data', (data) => {
          this.totalProcessed++;
          
          // Validate essential fields
          if (!data.product_name || !data.brand || !data.discounted_price || 
              data.product_name.trim() === '' || data.brand.trim() === '') {
            return; // Skip invalid products
          }
          
          // Add original product
          results.push(data);
          
          // Generate variations
          const variations = this.generateProductVariations(data);
          const selectedVariations = variations.slice(0, expansionFactor - 1); // Keep total at expansionFactor
          
          results.push(...selectedVariations);
          this.totalGenerated += selectedVariations.length;
          
          // Progress indicator
          if (this.totalProcessed % 500 === 0) {
            console.log(`📊 Processed ${this.totalProcessed} base products, generated ${results.length} total products`);
          }
        })
        .on('end', async () => {
          console.log(`✅ Processing complete!`);
          console.log(`📊 Base products processed: ${this.totalProcessed}`);
          console.log(`📊 Total products generated: ${results.length}`);
          console.log(`📊 Expansion ratio: ${(results.length / this.totalProcessed).toFixed(1)}x`);

          // Write expanded data to CSV
          if (results.length > 0) {
            const csvWriter = createCsvWriter({
              path: outputPath,
              header: Object.keys(results[0]).map(key => ({ id: key, title: key }))
            });

            try {
              await csvWriter.writeRecords(results);
              console.log(`💾 Expanded dataset saved to: ${outputPath}`);
              resolve({
                totalProcessed: this.totalProcessed,
                totalGenerated: results.length,
                expansionRatio: results.length / this.totalProcessed,
                outputPath: outputPath
              });
            } catch (error) {
              reject(error);
            }
          } else {
            reject(new Error('No valid products found after processing'));
          }
        })
        .on('error', reject);
    });
  }
}

module.exports = DatasetExpansion;

// CLI usage
if (require.main === module) {
  const expansion = new DatasetExpansion();
  const inputPath = process.argv[2] || './data/flipkart.csv';
  const outputPath = process.argv[3] || './data/flipkart_expanded.csv';
  const expansionFactor = parseInt(process.argv[4]) || 5;
  
  expansion.processAndExpand(inputPath, outputPath, expansionFactor)
    .then(result => {
      console.log('🎉 Dataset expansion completed successfully!');
      console.log(`📈 Final dataset size: ${result.totalGenerated} products`);
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Error during expansion:', error);
      process.exit(1);
    });
}
