const mongoose = require('mongoose');
const UserQuery = require('../models/UserQuery');
const elasticClient = require('../elasticClient');

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/flipkart_grid', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

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

// General suggestion patterns - these will also be stored in database
const GENERAL_SUGGESTION_PATTERNS = {
  // Electronics & Tech
  'iph': ['iphone', 'iphone 15', 'iphone 14', 'iphone case', 'iphone charger'],
  'phone': ['phone', 'phone case', 'phone charger', 'phone under 10000', 'phone under 15000', 'phone under 20000'],
  'mobile': ['mobile', 'mobile phone', 'mobile case', 'mobile charger', 'mobile under 10000'],
  'laptop': ['laptop', 'laptop bag', 'laptop stand', 'laptop under 30000', 'laptop under 50000'],
  'headphone': ['headphones', 'headphones wireless', 'headphones bluetooth', 'headphones gaming'],
  'earphone': ['earphones', 'earphones wireless', 'earphones bluetooth', 'earphones wired'],
  'watch': ['watch', 'watch men', 'watch women', 'watch smart', 'watch analog', 'watch digital'],
  'smart': ['smartphone', 'smart tv', 'smartwatch', 'smart home', 'smart speaker'],
  'wireless': ['wireless earphones', 'wireless mouse', 'wireless keyboard', 'wireless charger'],
  'bluetooth': ['bluetooth speaker', 'bluetooth earphones', 'bluetooth mouse', 'bluetooth headphones'],
  'power': ['power bank', 'power adapter', 'power tools', 'power bank 10000mah', 'power bank 20000mah'],
  'gaming': ['gaming laptop', 'gaming chair', 'gaming mouse', 'gaming keyboard', 'gaming headset'],
  'camera': ['camera', 'camera digital', 'camera dslr', 'camera mirrorless', 'camera action'],
  'tv': ['tv', 'tv smart', 'tv led', 'tv 32 inch', 'tv 43 inch', 'tv 55 inch'],
  
  // Fashion & Accessories
  'shoe': ['shoes', 'shoes men', 'shoes women', 'shoes sports', 'shoes formal', 'shoes casual'],
  'bag': ['bag', 'bag women', 'bag men', 'bag laptop', 'bag school', 'bag travel'],
  'dress': ['dress', 'dress women', 'dress party', 'dress casual', 'dress formal'],
  'shirt': ['shirt', 'shirt men', 'shirt women', 'shirt formal', 'shirt casual'],
  'jean': ['jeans', 'jeans men', 'jeans women', 'jeans blue', 'jeans black'],
  
  // Home & Kitchen
  'kitchen': ['kitchen appliances', 'kitchen utensils', 'kitchen storage', 'kitchen tools'],
  'home': ['home decor', 'home appliances', 'home furniture', 'home lighting'],
  'furniture': ['furniture', 'furniture bedroom', 'furniture living room', 'furniture office'],
  'bed': ['bed', 'bed sheet', 'bed cover', 'bedroom furniture'],
  'chair': ['chair', 'chair office', 'chair gaming', 'chair dining'],
  
  // Beauty & Personal Care
  'beauty': ['beauty products', 'beauty makeup', 'beauty skincare', 'beauty tools'],
  'makeup': ['makeup', 'makeup kit', 'makeup brushes', 'makeup remover'],
  'skin': ['skincare', 'skin care products', 'skin moisturizer', 'skin cleanser'],
  'hair': ['hair care', 'hair oil', 'hair shampoo', 'hair conditioner'],
  
  // Sports & Fitness
  'fitness': ['fitness equipment', 'fitness tracker', 'fitness accessories'],
  'gym': ['gym equipment', 'gym accessories', 'gym bag', 'gym clothes'],
  'sports': ['sports equipment', 'sports shoes', 'sports wear', 'sports accessories'],
  
  // Books & Education
  'book': ['books', 'books fiction', 'books non fiction', 'books educational'],
  'study': ['study table', 'study chair', 'study materials', 'study accessories'],
  
  // Brands
  'apple': ['apple iphone', 'apple macbook', 'apple watch', 'apple airpods'],
  'samsung': ['samsung phone', 'samsung tv', 'samsung galaxy', 'samsung tablet'],
  'nike': ['nike shoes', 'nike clothing', 'nike accessories', 'nike sports'],
  'adidas': ['adidas shoes', 'adidas clothing', 'adidas sports', 'adidas accessories']
};

async function populateAllSuggestions() {
  try {
    console.log('🚀 Starting to populate all suggestions in database...');
    
    // 1. Populate Popular Queries
    console.log('\n📝 Populating popular queries...');
    for (const query of POPULAR_QUERIES) {
      try {
        const updatedQuery = await UserQuery.findOneAndUpdate(
          { queryText: query },
          {
            $inc: { frequency: 15 }, // Higher base frequency for popular queries
            $set: { 
              lastClicked: new Date(),
              isPopular: true // Mark as popular query
            }
          },
          { new: true, upsert: true }
        );

        console.log(`✅ Popular query: "${query}" (frequency: ${updatedQuery.frequency})`);

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
        
      } catch (error) {
        console.error(`❌ Error with popular query "${query}":`, error.message);
      }
    }
    
    // 2. Populate General Suggestion Patterns
    console.log('\n🎯 Populating general suggestion patterns...');
    for (const [pattern, suggestions] of Object.entries(GENERAL_SUGGESTION_PATTERNS)) {
      for (let i = 0; i < suggestions.length; i++) {
        const suggestion = suggestions[i];
        try {
          const updatedQuery = await UserQuery.findOneAndUpdate(
            { queryText: suggestion },
            {
              $inc: { frequency: 10 - i }, // Higher frequency for earlier suggestions
              $set: { 
                lastClicked: new Date(),
                isGeneralPattern: true,
                patternKey: pattern, // Store which pattern this belongs to
                patternWeight: 8 - i * 0.5
              }
            },
            { new: true, upsert: true }
          );

          console.log(`✅ Pattern "${pattern}" -> "${suggestion}" (frequency: ${updatedQuery.frequency})`);

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
          
        } catch (error) {
          console.error(`❌ Error with pattern "${pattern}" -> "${suggestion}":`, error.message);
        }
      }
    }
    
    console.log('\n🎉 Successfully populated all suggestions in database!');
    console.log('📊 You can now view all suggestions in MongoDB Compass');
    console.log('🔍 Check the UserQuery collection for all populated data');
    
  } catch (error) {
    console.error('💥 Error populating suggestions:', error);
  } finally {
    await mongoose.connection.close();
    console.log('🔌 Database connection closed');
  }
}

// Run the script
populateAllSuggestions();
