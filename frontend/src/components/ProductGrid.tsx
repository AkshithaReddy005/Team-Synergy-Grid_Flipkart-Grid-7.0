import React from 'react';
import ProductCard from './ProductCard';
import { useSearch } from '../context/SearchContext';

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

interface ProductGridProps {
  products: Product[];
  viewMode?: 'grid' | 'list';
}

const ProductGrid: React.FC<ProductGridProps> = ({ products, viewMode = 'grid' }) => {
  const { filters } = useSearch();

  // Check if a product should be hidden based on filters
  const shouldHideProduct = (product: Product) => {
    // Price range filter
    if (product.price < filters.priceRange[0] || product.price > filters.priceRange[1]) {
      return true;
    }
    
    // Rating filter
    if (filters.rating > 0 && product.rating < filters.rating) {
      return true;
    }
    
    // Offers filter (products with discount)
    if (filters.offers && (!product.discount || product.discount === 0)) {
      return true;
    }
    
    return false;
  };
  if (viewMode === 'list') {
    return (
      <div className="space-y-4">
        {products.map((product) => (
          <div 
            key={product.id} 
            className={`bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow ${
              shouldHideProduct(product) ? 'hidden' : ''
            }`}
          >
            <div className="flex p-4">
              <div className="w-32 h-32 flex-shrink-0 mr-4">
                <img
                  src={product.image}
                  alt={product.name}
                  className="w-full h-full object-cover rounded-lg"
                />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-lg mb-2 line-clamp-2">{product.name}</h3>
                <div className="flex items-center space-x-2 mb-2">
                  <div className="flex items-center">
                    <span className="text-[#FFD700] mr-1">★</span>
                    <span className="font-medium">{product.rating}</span>
                    <span className="text-gray-500 text-sm ml-1">
                      ({product.reviews.toLocaleString()})
                    </span>
                  </div>
                  {product.badges && product.badges.map((badge) => (
                    <span
                      key={badge}
                      className="px-2 py-1 bg-[#2874F0] text-white text-xs rounded-full font-medium"
                    >
                      {badge}
                    </span>
                  ))}
                </div>
                <div className="flex items-center space-x-2 mb-4">
                  <span className="text-2xl font-bold text-gray-900">
                    ₹{product.price.toLocaleString()}
                  </span>
                  {product.originalPrice && (
                    <>
                      <span className="text-gray-500 line-through">
                        ₹{product.originalPrice.toLocaleString()}
                      </span>
                      <span className="text-green-600 font-medium">
                        {Math.round(((product.originalPrice - product.price) / product.originalPrice) * 100)}% off
                      </span>
                    </>
                  )}
                </div>
                <button className="px-6 py-2 bg-[#2874F0] text-white rounded-lg hover:bg-[#1e5cb8] transition-colors font-medium">
                  Add to Cart
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {products.map((product) => (
        <div key={product.id} className={shouldHideProduct(product) ? 'hidden' : ''}>
          <ProductCard product={product} />
        </div>
      ))}
    </div>
  );
};

export default ProductGrid;