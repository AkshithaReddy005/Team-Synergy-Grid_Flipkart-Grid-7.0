import axios from 'axios';
import { extractFilters, isTopRatedQuery, extractRatingFromQuery } from '../utils/queryParser';

// API client configuration
const apiClient = axios.create({
  baseURL: 'http://localhost:5001/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Fallback suggestions when Gemini API is not available
const fallbackSuggestions: { [key: string]: string[] } = {
  'iphone': ['iPhone 15', 'iPhone 14', 'iPhone 13', 'iPhone charger', 'iPhone case'],
  'laptop': ['Laptop bag', 'Laptop charger', 'Gaming laptop', 'MacBook', 'Dell laptop'],
  'headphone': ['Wireless headphones', 'Bluetooth headphones', 'Gaming headset', 'Earbuds'],
  'backpack': ['School backpack', 'Travel backpack', 'Laptop backpack', 'Hiking backpack'],
  'shoes': ['Running shoes', 'Sports shoes', 'Casual shoes', 'Formal shoes'],
  'mobile': ['Smartphone', 'Android phone', 'iPhone', 'Mobile case', 'Mobile charger'],
  'watch': ['Smartwatch', 'Digital watch', 'Analog watch', 'Sports watch'],
  'camera': ['DSLR camera', 'Mirrorless camera', 'Action camera', 'Camera lens'],
  'speaker': ['Bluetooth speaker', 'Portable speaker', 'Home theater', 'Soundbar'],
  'bag': ['Handbag', 'Shoulder bag', 'Crossbody bag', 'Tote bag', 'Laptop bag'],
  'charger': ['Phone charger', 'Laptop charger', 'Wireless charger', 'Fast charger'],
  'case': ['Phone case', 'Laptop case', 'Tablet case', 'Camera case'],
  'wireless': ['Wireless headphones', 'Wireless charger', 'Wireless mouse', 'Wireless keyboard'],
  'bluetooth': ['Bluetooth headphones', 'Bluetooth speaker', 'Bluetooth mouse', 'Bluetooth keyboard'],
  'gaming': ['Gaming laptop', 'Gaming mouse', 'Gaming keyboard', 'Gaming headset'],
  'sports': ['Sports shoes', 'Sports watch', 'Sports bag', 'Sports equipment'],
  'travel': ['Travel bag', 'Travel adapter', 'Travel pillow', 'Travel accessories'],
  'kitchen': ['Kitchen appliances', 'Kitchen tools', 'Kitchen gadgets', 'Kitchen accessories'],
  'home': ['Home decor', 'Home appliances', 'Home furniture', 'Home accessories'],
  'fitness': ['Fitness tracker', 'Fitness equipment', 'Fitness accessories', 'Fitness gear']
};

// Get fallback suggestions based on query
function getFallbackSuggestions(query: string): string[] {
  const lowerQuery = query.toLowerCase();
  
  // Find matching categories
  for (const [category, suggestions] of Object.entries(fallbackSuggestions)) {
    if (lowerQuery.includes(category)) {
      return suggestions;
    }
  }
  
  // Default suggestions
  return ['iPhone 15', 'Samsung Galaxy', 'MacBook Air', 'Wireless Headphones', 'Gaming Laptop'];
}

// Types
interface Product {
  id: string;
  name: string;
  price: number;
  originalPrice?: number;
  image: string;
  rating: number;
  reviews: number;
  discount?: number;
  badges?: string[];
  brand?: string;
  category?: string;
  popularity?: number;
}

interface Suggestion {
  id: string;
  text: string;
  type: 'product' | 'category' | 'brand';
  isCorrection?: boolean;
}

interface BackendSuggestion {
  text: string;
  type: 'popular' | 'general' | 'correction' | 'conceptual';
  originalQuery?: string;
}

interface SearchResponse {
  products: Array<{
    _source?: {
      productId: string;
      id?: string;
      name: string;
      price: number;
      originalPrice?: number;
      original_price?: number;
      thumbnail?: string;
      image?: string;
      rating: number;
      numReviews?: number;
      reviews?: number;
      brand?: string;
      category?: string;
      popularity?: number;
    };
    // For direct product format (from srp endpoint)
    productId?: string;
    id?: string;
    name?: string;
    price?: number;
    originalPrice?: number;
    thumbnail?: string;
    image?: string;
    rating?: number;
    numReviews?: number;
    reviews?: number;
    brand?: string;
    category?: string;
    popularity?: number;
  }>;
  total: number | { value: number };
}

// Convert backend suggestion to frontend format
const convertSuggestion = (backendSuggestion: BackendSuggestion, index: number): Suggestion => {
  return {
    id: `suggestion-${index}`,
    text: backendSuggestion.text,
    type: 'product', // Map all types to product for now
    isCorrection: backendSuggestion.type === 'correction',
  };
};

// Convert backend product to frontend format
const convertProduct = (backendProduct: any): Product => {
  console.log('Converting product:', backendProduct);

  // Handle both _source format and direct format
  const source = backendProduct._source || backendProduct;
  console.log('Source object:', source);
  
  const originalPrice = source.originalPrice || source.original_price || source.price;
  const discount = originalPrice > source.price ? Math.round(((originalPrice - source.price) / originalPrice) * 100) : 0;

  // Handle image field - check multiple possible sources
  let imageUrl = '';
  if (source.thumbnail && source.thumbnail !== '') {
    imageUrl = source.thumbnail;
  } else if (source.image && source.image !== '') {
    imageUrl = source.image;
  } else if (source.images && Array.isArray(source.images) && source.images.length > 0) {
    imageUrl = source.images[0];
  }

  const product: Product = {
    id: source.productId || source.id || `product-${Math.random()}`,
    name: source.name || 'Product Name',
    price: source.price || 0,
    originalPrice: originalPrice,
    image: imageUrl, // Will be handled by ProductCard component for placeholders
    rating: source.rating || 0,
    reviews: source.numReviews || source.reviews || 0,
    discount: discount > 0 ? discount : undefined,
    badges: discount > 0 ? ['Sale'] : [],
    brand: source.brand || 'Unknown Brand',
    category: source.category || 'General',
    popularity: source.popularity, // Ensure popularity is passed from backend
  };

  console.log('Converted product:', product);
  return product;
};

// API functions
export const getAutoSuggestions = async (query: string): Promise<Suggestion[]> => {
  try {
    if (!query.trim()) return [];
    
    const response = await apiClient.get(`/search/autosuggest?q=${encodeURIComponent(query)}`);
    const backendSuggestions: BackendSuggestion[] = response.data || [];
    
    // If we got suggestions from backend, use them
    if (backendSuggestions.length > 0) {
      return backendSuggestions.map(convertSuggestion);
    }
    
    // Fallback to local suggestions if backend returns empty
    const fallbackSuggestions = getFallbackSuggestions(query);
    return fallbackSuggestions.map((text, index) => ({
      id: `fallback-${index}`,
      text,
      type: 'product',
      isCorrection: false,
    }));
    
  } catch (error) {
    console.error('Error fetching autosuggestions:', error);
    // Fallback to local suggestions on error
    const fallbackSuggestions = getFallbackSuggestions(query);
    return fallbackSuggestions.map((text, index) => ({
      id: `fallback-${index}`,
      text,
      type: 'product',
      isCorrection: false,
    }));
  }
};

export const searchProducts = async (
  query: string,
  filters: any = {},
  sortBy: string = 'relevance'
): Promise<Product[]> => {
  try {
    console.log('Searching for:', query, 'with sortBy:', sortBy);
    
    // Parse the query for dynamic filters
    const parsedFilters = extractFilters(query);
    const queryKeywords = parsedFilters.keywords.join(' ');
    console.log('Parsed filters:', parsedFilters);
    
    // Build search parameters
    const params = new URLSearchParams();
    params.append('q', query);
    params.append('sortBy', sortBy); // Add sortBy parameter
    
    // Add dynamic filters from query parsing
    if (parsedFilters.price.lte) {
      params.append('price_lt', parsedFilters.price.lte.toString());
    }
    if (parsedFilters.price.gte) {
      params.append('price_gt', parsedFilters.price.gte.toString());
    }
    if (parsedFilters.rating.gte) {
      params.append('rating_gte', parsedFilters.rating.gte.toString());
    }
    
    // Only add category if it's a valid database category
    const validCategories = [
      'clothing', 'jewellery', 'footwear', 'mobiles & accessories', 
      'mobile phones', 'automotive', 'home decor & festive needs', 
      'home decor', 'home furnishing', 'computers', 'bags, wallets & belts', 
      'sports & fitness', 'toys & school supplies', 'mobiles', 'laptops'
    ];
    if (parsedFilters.category && validCategories.includes(parsedFilters.category.toLowerCase())) {
      params.append('category', parsedFilters.category);
    }
    
    if (parsedFilters.gender) {
      params.append('gender', parsedFilters.gender); // Use gender field for gender
    }
    
    if (parsedFilters.brand) {
      params.append('brand', parsedFilters.brand); // Use brand field for brand
    }
    
    // Add manual filters if they exist
    if (filters.category && filters.category.length > 0) {
      params.append('category', filters.category.join(','));
    }
    
    if (filters.priceRange) {
      params.append('minPrice', filters.priceRange[0].toString());
      params.append('maxPrice', filters.priceRange[1].toString());
    }
    
    if (filters.rating > 0) {
      params.append('minRating', filters.rating.toString());
    }
    
    console.log('Search URL:', `/srp/search?${params.toString()}`);
    console.log('Sort will be handled by backend with sortBy:', sortBy);
    
    // Try the dynamic search endpoint first
    let response;
    let searchResponse: SearchResponse;
    
    try {
      response = await apiClient.get(`/srp/search?${params.toString()}`);
      searchResponse = response.data;
      console.log('Dynamic search response:', searchResponse);
      
      // If dynamic search returns 0 products, fallback to regular search
      if (!searchResponse.products || searchResponse.products.length === 0) {
        console.log('Dynamic search returned 0 products, falling back to regular search');
        response = await apiClient.get(`/search/products?${params.toString()}`);
        searchResponse = response.data;
        console.log('Regular search response:', searchResponse);
      }
    } catch (dynamicError) {
      // Fallback to regular search if dynamic search fails
      console.log('Dynamic search failed, falling back to regular search:', dynamicError);
      response = await apiClient.get(`/search/products?${params.toString()}`);
      searchResponse = response.data;
      console.log('Regular search response:', searchResponse);
    }
    
    const totalProducts = typeof searchResponse.total === 'number' 
      ? searchResponse.total 
      : searchResponse.total.value;
    console.log('Total products found:', totalProducts);
    console.log('Products array length:', searchResponse.products?.length);
    
    if (!searchResponse.products || searchResponse.products.length === 0) {
      console.log('No products in response');
      return [];
    }
    
        console.log('Raw products array:', searchResponse.products);
    const convertedProducts = searchResponse.products.map(convertProduct);
    console.log('Converted products:', convertedProducts.length);
    console.log('First converted product:', convertedProducts[0]);

    return convertedProducts;
  } catch (error) {
    console.error('Error searching products:', error);
    // Fallback to empty array on error
    return [];
  }
};

export const getFeaturedProducts = async (): Promise<Product[]> => {
  try {
    // Use a generic search to get featured products
    const response = await apiClient.get('/search/products?q=featured');
    const searchResponse: SearchResponse = response.data;
    
    return searchResponse.products.slice(0, 8).map(convertProduct);
  } catch (error) {
    console.error('Error fetching featured products:', error);
    return [];
  }
};

// Analytics functions (optional, for tracking search behavior)
export const logSearch = async (query: string) => {
  try {
    await apiClient.post('/search/log', { query });
    console.log('Search logged:', query);
  } catch (error) {
    console.error('Failed to log search:', error);
  }
};

export const logSuggestionClick = async (query: string, prefix: string) => {
  try {
    await apiClient.post('/analytics/autosuggest-click', { query, prefix });
  } catch (error) {
    console.error('Error logging suggestion click:', error);
  }
}; 