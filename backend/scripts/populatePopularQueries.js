const mongoose = require('mongoose');
const UserQuery = require('../models/UserQuery');
const elasticClient = require('../elasticClient');

// Popular queries for gift suggestions and common searches
const POPULAR_QUERIES = [
  // Gift queries
  'gift for sister', 'gift for best friend', 'gift for wife', 'gift for husband',
  'gift for mom', 'gift for dad', 'gift for girlfriend', 'gift for boyfriend',
  'birthday gift', 'anniversary gift', 'wedding gift', 'housewarming gift',
  'diwali gift', 'rakhi gift', 'christmas gift', 'valentine gift',
  'gift for brother', 'gift for parents', 'gift for grandparents',
  'gift for teacher', 'gift for colleague', 'gift for boss',
  
  // Common product categories
  'mobile phones', 'laptops', 'smartphones', 'headphones', 'watches',
  'shoes', 'clothing', 'jewellery', 'home decor', 'kitchen appliances',
  'books', 'toys', 'sports equipment', 'fitness equipment', 'beauty products',
  'gaming console', 'camera', 'television', 'refrigerator', 'washing machine',
  'air conditioner', 'laptop bag', 'backpack', 'wallet', 'sunglasses',
  'perfume', 'makeup', 'skincare', 'hair care', 'body care',
  
  // Specific product searches
  'iphone 15', 'samsung galaxy', 'oneplus nord', 'redmi note',
  'nike shoes', 'adidas sneakers', 'puma sports shoes',
  'titan watch', 'fastrack watch', 'casio watch',
  'ray-ban sunglasses', 'oakley sunglasses',
  'hp laptop', 'dell laptop', 'lenovo laptop', 'macbook',
  'boat headphones', 'jbl speaker', 'sony headphones',
  'canon camera', 'nikon camera', 'gopro camera',
  
  // Seasonal and occasion-based
  'diwali home decor', 'christmas decorations', 'rakhi gifts',
  'wedding guest dress', 'party wear for women', 'office wear men',
  'casual dress for girls', 'summer clothes for kids', 'winter jackets',
  'raincoat for bike', 'mosquito net for bed', 'study lamp for students',
  'yoga accessories', 'gym wear men', 'travel essentials',
  'new born baby gifts', 'housewarming gift ideas',
  
  // Price-based searches
  'mobiles under 15000', 'laptop under 50000', 'shoes under 2000',
  'watch under 1000', 'headphones under 1000', 'camera under 30000',
  'phone with good camera', 'laptop with 16gb ram',
  'noise cancellation earphones', 'water resistant smartwatch',
  'smart tv 55 inch', 'fully automatic washing machine',
  'no frost refrigerator', 'split ac 1.5 ton'
];

async function populatePopularQueries() {
  try {
    console.log('Starting to populate popular queries...');
    
    for (const query of POPULAR_QUERIES) {
      try {
        // Create or update the query in MongoDB
        const updatedQuery = await UserQuery.findOneAndUpdate(
          { queryText: query },
          {
            $inc: { frequency: 10 }, // Give popular queries a higher base frequency
            $set: { lastClicked: new Date() }
          },
          { new: true, upsert: true }
        );

        console.log(`Updated query: "${query}" with frequency: ${updatedQuery.frequency}`);

        // Sync to Elasticsearch
        await elasticClient.index({
          index: 'search_queries',
          id: updatedQuery._id.toString(),
          body: {
            queryText: updatedQuery.queryText,
            suggest: {
              input: [updatedQuery.queryText],
              weight: updatedQuery.frequency
            }
          }
        });

        console.log(`Synced "${query}" to Elasticsearch`);
        
        // Small delay to avoid overwhelming the system
        await new Promise(resolve => setTimeout(resolve, 100));
        
      } catch (error) {
        console.error(`Error processing query "${query}":`, error);
      }
    }
    
    console.log('Successfully populated popular queries!');
    
  } catch (error) {
    console.error('Error populating popular queries:', error);
  } finally {
    // Close the connection
    await mongoose.connection.close();
    console.log('Database connection closed');
  }
}

// Run the script
populatePopularQueries(); 