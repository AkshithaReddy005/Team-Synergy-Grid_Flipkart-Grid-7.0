const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

// Cache the dataset in memory
let productsCache = [];
let isLoaded = false;

const loadDataset = () => {
  return new Promise((resolve, reject) => {
    if (isLoaded) return resolve();
    
    const results = [];
    let csvPath = path.join(__dirname, '../data/Dataset_Final_TeamSynergyGrid.csv');
    const alternatePath = "C:\\Users\\akshi\\OneDrive\\Documents\\AKKI\\projects\\Team-Synergy-Grid_Flipkart-Grid-7.0\\SRP-main\\Dataset_Final_TeamSynergyGrid.csv";
    
    if (!fs.existsSync(csvPath)) {
        if (fs.existsSync(alternatePath)) {
            csvPath = alternatePath;
        } else {
            console.log('[FallbackSearch] No CSV dataset found at local or alternate path.');
            return resolve();
        }
    }

    console.log('[FallbackSearch] Loading CSV dataset into memory...');
    fs.createReadStream(csvPath)
      .pipe(csv())
      .on('data', (data) => {
        try {
            const mappedProduct = {
                id: data.uniq_id || data.id || Math.random().toString(),
                name: data.product_name || '',
                brand: data.brand || '',
                description: data.description || '',
                price: parseFloat(data.discounted_price || data.retail_price || 0),
                originalPrice: parseFloat(data.retail_price || 0),
                rating: parseFloat(data.product_rating || 0),
                popularity: parseInt(data.popularity || 0),
                image: (data.image && data.image !== '') ? JSON.parse(data.image.replace(/'/g, '"'))[0] : 'https://via.placeholder.com/300',
                category: data.category_tree ? data.category_tree : 'General'
            };
            results.push(mappedProduct);
        } catch(e) {
            // Ignore badly formatted rows
        }
      })
      .on('end', () => {
        productsCache = results;
        isLoaded = true;
        console.log(`[FallbackSearch] Loaded ${productsCache.length} products into memory.`);
        resolve();
      })
      .on('error', (err) => {
        console.error('[FallbackSearch] Error loading dataset:', err);
        resolve();
      });
  });
};

const searchFallback = async (query, options = {}) => {
  if (!isLoaded) {
    await loadDataset();
  }

  const queryLower = query ? query.toLowerCase() : '';
  
  // Basic text match filtering
  let results = productsCache.filter(p => {
    if (!queryLower) return true;
    return (
      (p.name && p.name.toLowerCase().includes(queryLower)) ||
      (p.brand && p.brand.toLowerCase().includes(queryLower)) ||
      (p.category && p.category.toLowerCase().includes(queryLower))
    );
  });

  // Price constraints
  if (options.minPrice !== undefined) {
      results = results.filter(p => p.price >= options.minPrice);
  }
  if (options.maxPrice !== undefined) {
      results = results.filter(p => p.price <= options.maxPrice);
  }

  // Sorting
  if (options.sortBy === 'price_asc') {
      results.sort((a, b) => a.price - b.price);
  } else if (options.sortBy === 'price_desc') {
      results.sort((a, b) => b.price - a.price);
  } else if (options.sortBy === 'rating') {
      results.sort((a, b) => b.rating - a.rating);
  } else {
      // Default: popularity
      results.sort((a, b) => b.popularity - a.popularity);
  }

  // Pagination limit
  const size = options.size || 100;
  results = results.slice(0, size);

  // Map to format that controllers expect from Elastic
  return {
    hits: {
        total: { value: results.length },
        hits: results.map(r => ({
            _id: r.id,
            _source: r
        }))
    }
  };
};

module.exports = {
  searchFallback,
  loadDataset
};
