// Mock API functions to simulate backend functionality

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
}

interface Suggestion {
  id: string;
  text: string;
  type: 'product' | 'category' | 'brand';
  isCorrection?: boolean;
}

// Mock product database
export const mockProducts: Product[] = [
  {
    id: '1',
    name: 'iPhone 15 Pro Max 256GB Natural Titanium',
    price: 134900,
    originalPrice: 159900,
    image: 'https://images.pexels.com/photos/699122/pexels-photo-699122.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.5,
    reviews: 15420,
    badges: ['Assured', 'Trending'],
    brand: 'Apple',
    category: 'Electronics',
  },
  {
    id: '2',
    name: 'Samsung Galaxy S24 Ultra 5G 512GB Titanium Black',
    price: 129999,
    originalPrice: 149999,
    image: 'https://images.pexels.com/photos/404280/pexels-photo-404280.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.3,
    reviews: 8730,
    badges: ['Sale'],
    brand: 'Samsung',
    category: 'Electronics',
  },
  {
    id: '3',
    name: 'MacBook Air 13" M3 Chip 8GB RAM 256GB SSD',
    price: 114900,
    originalPrice: 134900,
    image: 'https://images.pexels.com/photos/18105/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.7,
    reviews: 3245,
    badges: ['Assured'],
    brand: 'Apple',
    category: 'Electronics',
  },
  {
    id: '4',
    name: 'Sony WH-1000XM5 Wireless Noise Canceling Headphones',
    price: 29990,
    originalPrice: 34990,
    image: 'https://images.pexels.com/photos/3394650/pexels-photo-3394650.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.6,
    reviews: 12450,
    badges: ['Bestseller'],
    brand: 'Sony',
    category: 'Electronics',
  },
  {
    id: '5',
    name: 'Nike Air Max 270 Running Shoes',
    price: 12995,
    originalPrice: 16995,
    image: 'https://images.pexels.com/photos/2529148/pexels-photo-2529148.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.2,
    reviews: 5670,
    badges: ['Sale'],
    brand: 'Nike',
    category: 'Clothing',
  },
  {
    id: '6',
    name: 'LG 55" 4K UHD Smart OLED TV',
    price: 89990,
    originalPrice: 129990,
    image: 'https://images.pexels.com/photos/1444416/pexels-photo-1444416.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.4,
    reviews: 2340,
    badges: ['Assured', 'Sale'],
    brand: 'LG',
    category: 'Electronics',
  },
  {
    id: '7',
    name: 'Dell XPS 13 Laptop Intel i7 16GB RAM 512GB SSD',
    price: 124999,
    originalPrice: 149999,
    image: 'https://images.pexels.com/photos/18105/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.5,
    reviews: 1890,
    badges: ['Trending'],
    brand: 'Dell',
    category: 'Electronics',
  },
  {
    id: '8',
    name: 'Adidas Ultraboost 22 Running Shoes',
    price: 17999,
    originalPrice: 21999,
    image: 'https://images.pexels.com/photos/2529148/pexels-photo-2529148.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.3,
    reviews: 4520,
    badges: ['Bestseller'],
    brand: 'Adidas',
    category: 'Clothing',
  },
];

// Mock suggestions database
const mockSuggestions: Suggestion[] = [
  { id: '1', text: 'iPhone 15', type: 'product' },
  { id: '2', text: 'Samsung Galaxy', type: 'product' },
  { id: '3', text: 'MacBook', type: 'product' },
  { id: '4', text: 'Electronics', type: 'category' },
  { id: '5', text: 'Clothing', type: 'category' },
  { id: '6', text: 'Apple', type: 'brand' },
  { id: '7', text: 'Samsung', type: 'brand' },
  { id: '8', text: 'Nike', type: 'brand' },
  { id: '9', text: 'Sony', type: 'brand' },
  { id: '10', text: 'headphones', type: 'product' },
  { id: '11', text: 'laptop', type: 'product' },
  { id: '12', text: 'shoes', type: 'product' },
];

// Typo correction mapping
const typoCorrections: { [key: string]: string } = {
  'iphone': 'iPhone',
  'iPhne': 'iPhone',
  'mobiile': 'mobile',
  'laptp': 'laptop',
  'headfones': 'headphones',
  'samsng': 'Samsung',
  'macbok': 'MacBook',
};

// Simulate API delay
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const getAutoSuggestions = async (query: string): Promise<Suggestion[]> => {
  await delay(200); // Simulate network delay
  
  if (!query) return [];
  
  const lowerQuery = query.toLowerCase();
  let correctedQuery = lowerQuery;
  
  // Check for typo corrections
  Object.keys(typoCorrections).forEach(typo => {
    if (lowerQuery.includes(typo)) {
      correctedQuery = lowerQuery.replace(typo, typoCorrections[typo].toLowerCase());
    }
  });
  
  // Filter suggestions based on query
  let suggestions = mockSuggestions.filter(suggestion =>
    suggestion.text.toLowerCase().includes(correctedQuery)
  );
  
  // Add corrected suggestions if there was a typo
  if (correctedQuery !== lowerQuery) {
    const correctedSuggestions = mockSuggestions
      .filter(s => s.text.toLowerCase().includes(correctedQuery))
      .map(s => ({ ...s, isCorrection: true }));
    
    suggestions = [...correctedSuggestions, ...suggestions];
  }
  
  // Add trending suggestions if no matches
  if (suggestions.length === 0) {
    suggestions = [
      { id: 'trend1', text: 'Diwali Sale', type: 'category' },
      { id: 'trend2', text: 'Best Offers', type: 'category' },
      { id: 'trend3', text: 'iPhone 15', type: 'product' },
      { id: 'trend4', text: 'Samsung Galaxy', type: 'product' },
    ];
  }
  
  return suggestions.slice(0, 8);
};

export const searchProducts = async (
  query: string,
  filters: any,
  sortBy: string
): Promise<Product[]> => {
  await delay(500); // Simulate network delay
  
  let results = [...mockProducts];
  
  // Filter by search query
  if (query) {
    const lowerQuery = query.toLowerCase();
    results = results.filter(product =>
      product.name.toLowerCase().includes(lowerQuery) ||
      product.brand?.toLowerCase().includes(lowerQuery) ||
      product.category?.toLowerCase().includes(lowerQuery)
    );
  }
  
  // Apply filters
  if (filters.category.length > 0) {
    results = results.filter(product =>
      filters.category.includes(product.category)
    );
  }
  
  if (filters.priceRange) {
    results = results.filter(product =>
      product.price >= filters.priceRange[0] && product.price <= filters.priceRange[1]
    );
  }
  
  if (filters.rating > 0) {
    results = results.filter(product => product.rating >= filters.rating);
  }
  
  if (filters.offers) {
    results = results.filter(product => product.originalPrice && product.originalPrice > product.price);
  }
  
  // Apply sorting
  switch (sortBy) {
    case 'price-low-high':
      results.sort((a, b) => a.price - b.price);
      break;
    case 'price-high-low':
      results.sort((a, b) => b.price - a.price);
      break;
    case 'rating':
      results.sort((a, b) => b.rating - a.rating);
      break;
    case 'discount':
      results.sort((a, b) => {
        const aDiscount = a.originalPrice ? ((a.originalPrice - a.price) / a.originalPrice) * 100 : 0;
        const bDiscount = b.originalPrice ? ((b.originalPrice - b.price) / b.originalPrice) * 100 : 0;
        return bDiscount - aDiscount;
      });
      break;
    case 'newest':
      // Simulate newest first (in real app, this would be based on creation date)
      results.reverse();
      break;
    case 'popularity':
    default:
      results.sort((a, b) => b.reviews - a.reviews);
      break;
  }
  
  return results;
};

export const getFeaturedProducts = (): Product[] => {
  return mockProducts.slice(0, 8);
};