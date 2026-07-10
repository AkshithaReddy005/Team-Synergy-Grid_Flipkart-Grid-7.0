const { filterRelevantProducts } = require('./geminiService');

class RelevanceFilter {
  constructor() {
    console.log('🎯 Strict relevance filter initialized');
  }

  // Multi-layer relevance filtering to eliminate irrelevant products
  async applyStrictRelevanceFilter(query, products, maxResults = 50) {
    console.log(`🎯 Applying strict relevance filtering for "${query}" on ${products.length} products`);
    
    if (!products || products.length === 0) {
      return products;
    }

    try {
      // Layer 1: Basic keyword relevance check
      let filteredProducts = this.basicKeywordFilter(query, products);
      console.log(`🔍 Layer 1 (Keywords): ${filteredProducts.length}/${products.length} products passed`);

      // Layer 2: Category relevance check
      filteredProducts = this.categoryRelevanceFilter(query, filteredProducts);
      console.log(`📂 Layer 2 (Category): ${filteredProducts.length} products passed`);

      // Layer 3: AI-powered semantic relevance (most important)
      if (filteredProducts.length > 0) {
        filteredProducts = await this.aiSemanticFilter(query, filteredProducts, maxResults);
        console.log(`🤖 Layer 3 (AI Semantic): ${filteredProducts.length} products passed`);
      }

      // Layer 4: Final relevance scoring and ranking
      filteredProducts = this.finalRelevanceRanking(query, filteredProducts);
      console.log(`⭐ Layer 4 (Final Ranking): ${filteredProducts.length} products returned`);

      return filteredProducts.slice(0, maxResults);

    } catch (error) {
      console.error('❌ Relevance filtering error:', error);
      // Fallback to basic filtering if AI fails
      return this.basicKeywordFilter(query, products).slice(0, maxResults);
    }
  }

  // Layer 1: Basic keyword relevance filtering
  basicKeywordFilter(query, products) {
    const queryWords = query.toLowerCase().split(/\s+/).filter(word => word.length > 2);
    
    return products.filter(product => {
      const productText = `${product.name} ${product.brand} ${product.category}`.toLowerCase();
      
      // At least 50% of query words should match
      const matchingWords = queryWords.filter(word => 
        productText.includes(word) || 
        this.fuzzyMatch(word, productText)
      );
      
      const relevanceScore = matchingWords.length / queryWords.length;
      return relevanceScore >= 0.5; // Strict threshold
    });
  }

  // Layer 2: Category relevance filtering
  categoryRelevanceFilter(query, products) {
    const categoryKeywords = {
      'mobile': ['mobile', 'phone', 'smartphone', 'iphone', 'android'],
      'laptop': ['laptop', 'computer', 'notebook', 'macbook'],
      'shoes': ['shoes', 'sneakers', 'boots', 'sandals', 'footwear'],
      'watch': ['watch', 'smartwatch', 'timepiece'],
      'headphones': ['headphones', 'earphones', 'earbuds', 'audio'],
      'clothing': ['shirt', 'dress', 'jeans', 'pants', 'clothing', 'apparel']
    };

    const queryLower = query.toLowerCase();
    
    // Detect expected category from query
    let expectedCategory = null;
    for (const [category, keywords] of Object.entries(categoryKeywords)) {
      if (keywords.some(keyword => queryLower.includes(keyword))) {
        expectedCategory = category;
        break;
      }
    }

    if (!expectedCategory) {
      return products; // No specific category detected, keep all
    }

    // Filter products that match the expected category
    return products.filter(product => {
      const productCategory = product.category.toLowerCase();
      const productName = product.name.toLowerCase();
      
      const expectedKeywords = categoryKeywords[expectedCategory];
      return expectedKeywords.some(keyword => 
        productCategory.includes(keyword) || 
        productName.includes(keyword)
      );
    });
  }

  // Layer 3: AI-powered semantic relevance filtering
  async aiSemanticFilter(query, products, maxResults) {
    try {
      console.log(`🤖 Applying AI semantic filtering for "${query}"`);
      
      // Use Gemini AI for deep semantic understanding
      const relevantProducts = await filterRelevantProducts(query, products, maxResults * 2);
      
      if (relevantProducts && relevantProducts.length > 0) {
        return relevantProducts;
      } else {
        console.log('⚠️ AI found no relevant products, using filtered results');
        return products;
      }
    } catch (error) {
      console.error('AI semantic filtering error:', error);
      return products;
    }
  }

  // Layer 4: Final relevance scoring and ranking
  finalRelevanceRanking(query, products) {
    const queryWords = query.toLowerCase().split(/\s+/);
    
    return products.map(product => {
      let relevanceScore = 0;
      const productText = `${product.name} ${product.brand} ${product.category}`.toLowerCase();
      
      // Name match (highest weight)
      queryWords.forEach(word => {
        if (product.name.toLowerCase().includes(word)) {
          relevanceScore += 3;
        }
      });
      
      // Brand match (medium weight)
      queryWords.forEach(word => {
        if (product.brand.toLowerCase().includes(word)) {
          relevanceScore += 2;
        }
      });
      
      // Category match (lower weight)
      queryWords.forEach(word => {
        if (product.category.toLowerCase().includes(word)) {
          relevanceScore += 1;
        }
      });
      
      // Boost for exact phrase matches
      if (productText.includes(query.toLowerCase())) {
        relevanceScore += 5;
      }
      
      return { ...product, relevanceScore };
    })
    .filter(product => product.relevanceScore > 0) // Only products with some relevance
    .sort((a, b) => b.relevanceScore - a.relevanceScore); // Sort by relevance
  }

  // Fuzzy matching for typos
  fuzzyMatch(word, text) {
    // Simple fuzzy matching - check if word is substring with 1 character difference
    if (word.length < 4) return false;
    
    for (let i = 0; i <= text.length - word.length; i++) {
      const substring = text.substr(i, word.length);
      let differences = 0;
      
      for (let j = 0; j < word.length; j++) {
        if (word[j] !== substring[j]) {
          differences++;
        }
      }
      
      if (differences <= 1) { // Allow 1 character difference
        return true;
      }
    }
    
    return false;
  }

  // Quick relevance check for obvious mismatches
  isObviouslyIrrelevant(query, product) {
    const queryLower = query.toLowerCase();
    const productName = product.name.toLowerCase();
    const productCategory = product.category.toLowerCase();
    
    // Define obvious category mismatches
    const categoryMismatches = {
      'mobile': ['shoes', 'clothing', 'watch', 'laptop'],
      'shoes': ['mobile', 'phone', 'laptop', 'electronics'],
      'laptop': ['shoes', 'clothing', 'mobile', 'phone'],
      'watch': ['shoes', 'mobile', 'laptop', 'clothing']
    };
    
    for (const [queryCategory, incompatibleCategories] of Object.entries(categoryMismatches)) {
      if (queryLower.includes(queryCategory)) {
        return incompatibleCategories.some(incompatible => 
          productCategory.includes(incompatible)
        );
      }
    }
    
    return false;
  }
}

// Singleton instance
const relevanceFilter = new RelevanceFilter();

module.exports = relevanceFilter;
