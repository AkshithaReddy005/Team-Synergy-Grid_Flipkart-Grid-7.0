const mongoose = require('mongoose');
const Product = require('./models/Product');
require('dotenv').config();

const fixInvalidPrices = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log('[fixPrices]: Connected to MongoDB');

    // Find products with invalid prices (0 or null)
    const invalidPriceProducts = await Product.find({
      $or: [
        { price: { $lte: 0 } },
        { price: null },
        { price: { $exists: false } }
      ]
    });

    console.log(`[fixPrices]: Found ${invalidPriceProducts.length} products with invalid prices`);

    if (invalidPriceProducts.length === 0) {
      console.log('[fixPrices]: No products need price fixing!');
      return;
    }

    // Generate reasonable random prices for products with invalid prices
    // Based on category or just random prices between 100-5000
    const priceRanges = {
      'Electronics': { min: 500, max: 50000 },
      'Clothing': { min: 200, max: 3000 },
      'Home': { min: 300, max: 10000 },
      'Books': { min: 100, max: 1000 },
      'Sports': { min: 250, max: 5000 },
      'default': { min: 200, max: 2000 }
    };

    let fixedCount = 0;

    for (const product of invalidPriceProducts) {
      // Determine price range based on category or use default
      const range = priceRanges[product.category] || priceRanges.default;
      
      // Generate a random price within the range
      const newPrice = Math.floor(Math.random() * (range.max - range.min) + range.min);
      
      // Update the product
      await Product.updateOne(
        { _id: product._id },
        { $set: { price: newPrice } }
      );
      
      console.log(`[fixPrices]: Fixed "${product.name}" - Set price to ₹${newPrice}`);
      fixedCount++;
    }

    console.log(`[fixPrices]: Successfully fixed ${fixedCount} products with new prices`);

    // Verify the fix
    const remainingInvalid = await Product.countDocuments({
      $or: [
        { price: { $lte: 0 } },
        { price: null },
        { price: { $exists: false } }
      ]
    });

    console.log(`[fixPrices]: Remaining products with invalid prices: ${remainingInvalid}`);

  } catch (error) {
    console.error('[fixPrices]: Error:', error);
  } finally {
    await mongoose.disconnect();
    console.log('[fixPrices]: MongoDB disconnected');
  }
};

if (require.main === module) {
  fixInvalidPrices();
}

module.exports = fixInvalidPrices;
