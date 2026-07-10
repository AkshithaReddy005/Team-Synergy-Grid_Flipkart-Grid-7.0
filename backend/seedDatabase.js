require('dotenv').config();
const mongoose = require('mongoose');
const Papa = require('papaparse');
const fs = require('fs');
const path = require('path');
const Product = require('./models/Product');

const MONGO_URI = process.env.MONGO_URI;
const BATCH_SIZE = 500; // Process 500 records at a time

const seedDatabase = async () => {
  if (!MONGO_URI) {
    console.error('Error: MONGO_URI is not defined in .env file');
    process.exit(1);
  }

  try {
    await mongoose.connect(MONGO_URI);
    console.log('MongoDB connected successfully.');

    console.log('Clearing existing products from the database...');
    await Product.deleteMany({});
    console.log('Existing products cleared.');

    const csvFilePath = path.join(__dirname, 'data', 'flipkart.csv');
    if (!fs.existsSync(csvFilePath)) {
        console.error(`Error: CSV file not found at ${csvFilePath}`);
        process.exit(1);
    }

    let productsBatch = [];
    let recordCount = 0;

        const fileStream = fs.createReadStream(csvFilePath);

    console.log('Starting to read and process the CSV file with Papaparse...');

    await new Promise((resolve, reject) => {
      Papa.parse(fileStream, {
        header: true,
        worker: true, // Use a background worker for parsing
        step: async (result, parser) => {
          parser.pause(); // Pause streaming to process the batch
          const row = result.data;

      // Skip rows with essential data missing
      if (!row.pid || !row.product_name || !row.discounted_price) {
        parser.resume();
        return;
      }

      // Safely parse image URL
      let imageUrl = '';
      try {
        const images = JSON.parse(row.image.replace(/""/g, '"'));
        if (Array.isArray(images) && images.length > 0) {
          imageUrl = images[0];
        }
      } catch (e) {
        // Ignore parsing errors, imageUrl will remain empty
      }

      // Safely parse category
      let category = '';
      try {
          const categoryTree = JSON.parse(row.product_category_tree);
          if (Array.isArray(categoryTree) && categoryTree.length > 0) {
              category = categoryTree[0].split(' >> ')[1] || categoryTree[0];
          }
      } catch(e) {
          // Ignore parsing errors
      }

      const productData = {
        productId: row.pid,
        name: row.product_name,
        description: row.description || 'No description available.',
        price: parseFloat(row.discounted_price) || 0,
        category: category || 'Uncategorized',
        brand: row.brand || 'Unbranded',
        imageUrl: imageUrl,
        rating: row.product_rating === 'No rating available' ? 0 : parseFloat(row.product_rating) || 0,
        numReviews: 0, // Default value as it's not in the CSV
      };

      productsBatch.push(productData);
      recordCount++;

      if (productsBatch.length >= BATCH_SIZE) {
        try {
          await Product.insertMany(productsBatch);
          console.log(`Inserted batch of ${productsBatch.length} products. Total processed: ${recordCount}`);
          productsBatch = [];
        } catch (e) {
          console.error('Error inserting batch:', e);
          parser.abort();
          return reject(e);
        }
      }
      parser.resume();
        },
        complete: async () => {
          // Insert any remaining products in the last batch
          if (productsBatch.length > 0) {
            try {
              await Product.insertMany(productsBatch);
              console.log(`Inserted final batch of ${productsBatch.length} products.`);
            } catch (e) {
              console.error('Error inserting final batch:', e);
              return reject(e);
            }
          }
          resolve();
        },
        error: (error) => {
          console.error('Papaparse error:', error.message);
          reject(error);
        }
      });
    });

    // Insert any remaining products in the last batch
    if (productsBatch.length > 0) {
      await Product.insertMany(productsBatch);
      console.log(`Inserted final batch of ${productsBatch.length} products.`);
    }

    console.log(`Database seeding completed. Total products inserted: ${recordCount}`);

  } catch (error) {
    console.error('An error occurred during database seeding:', error);
  } finally {
    await mongoose.disconnect();
    console.log('MongoDB connection closed.');
  }
};

seedDatabase();
