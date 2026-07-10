import React, { useState } from 'react';
import { Heart, Star, Check } from 'lucide-react';
import { useCart } from '../context/CartContext';

interface Product {
  id: string | number;
  name?: string;
  title?: string;
  price: number;
  originalPrice?: number;
  image: string;
  rating: number | { rate: number; count: number };
  reviews?: number;
  discount?: number;
  badges?: string[];
  brand?: string;
  category?: string;
  popularity?: number;
}

interface ProductCardProps {
  product: Product;
}

const ProductCard: React.FC<ProductCardProps> = ({ product }) => {
  const [isLiked, setIsLiked] = useState(false);
  const [isAddingToCart, setIsAddingToCart] = useState(false);
  const [isAdded, setIsAdded] = useState(false);
  const { addToCart } = useCart();

  // Function to get placeholder image based on product category and brand
  const getPlaceholderImage = (product: Product): string => {
    const category = product.category?.toLowerCase() || '';
    const brand = product.brand?.toLowerCase() || '';
    const name = product.name?.toLowerCase() || product.title?.toLowerCase() || '';

    // Shoe-related products
    if (category.includes('shoe') || category.includes('footwear') || name.includes('shoe')) {
      if (brand.includes('nike')) return 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300&h=300&fit=crop';
      if (brand.includes('adidas')) return 'https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=300&h=300&fit=crop';
      if (brand.includes('puma')) return 'https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=300&h=300&fit=crop';
      return 'https://images.unsplash.com/photo-1560769629-975ec94e6a86?w=300&h=300&fit=crop';
    }

    // Mobile/Phone products
    if (category.includes('mobile') || category.includes('phone') || name.includes('mobile') || name.includes('phone')) {
      if (brand.includes('iphone') || brand.includes('apple')) return 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=300&h=300&fit=crop';
      if (brand.includes('samsung')) return 'https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=300&h=300&fit=crop';
      return 'https://images.unsplash.com/photo-1592750475338-74b7b21085ab?w=300&h=300&fit=crop';
    }

    // Laptop/Computer products
    if (category.includes('laptop') || category.includes('computer') || name.includes('laptop') || name.includes('computer')) {
      if (brand.includes('macbook') || brand.includes('apple')) return 'https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=300&h=300&fit=crop';
      if (brand.includes('dell')) return 'https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=300&h=300&fit=crop';
      return 'https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=300&h=300&fit=crop';
    }

    // Headphone/Audio products
    if (category.includes('headphone') || category.includes('earphone') || name.includes('headphone') || name.includes('earphone')) {
      return 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=300&h=300&fit=crop';
    }

    // Watch/Timepiece products
    if (category.includes('watch') || name.includes('watch')) {
      return 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=300&h=300&fit=crop';
    }

    // Bag/Backpack products
    if (category.includes('bag') || category.includes('backpack') || name.includes('bag') || name.includes('backpack')) {
      return 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=300&fit=crop';
    }

    // Clothing/Fashion products
    if (category.includes('clothing') || category.includes('fashion') || name.includes('shirt') || name.includes('dress')) {
      return 'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=300&h=300&fit=crop';
    }

    // Camera products
    if (category.includes('camera') || name.includes('camera')) {
      return 'https://images.unsplash.com/photo-1516035069371-29a1b244cc32?w=300&h=300&fit=crop';
    }

    // Speaker/Audio products
    if (category.includes('speaker') || name.includes('speaker')) {
      return 'https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=300&h=300&fit=crop';
    }

    // Default placeholder
    return 'https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=300&h=300&fit=crop';
  };

  // Get the appropriate image URL
  const imageUrl = product.image && product.image !== '' && product.image !== 'https://via.placeholder.com/300x300?text=Product' 
    ? product.image 
    : getPlaceholderImage(product);

  const handleAddToCart = async () => {
    setIsAddingToCart(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Add to cart
    addToCart({
      id: product.id.toString(),
      name: product.name || product.title || 'Product',
      price: product.price,
      originalPrice: product.originalPrice,
      image: imageUrl,
      brand: product.brand,
      category: product.category,
    });
    
    setIsAdded(true);
    setIsAddingToCart(false);
    
    // Reset after 2 seconds
    setTimeout(() => {
      setIsAdded(false);
    }, 2000);
  };

  const discountPercentage = product.originalPrice 
    ? Math.round(((product.originalPrice - product.price) / product.originalPrice) * 100)
    : 0;

  return (
    <div className="bg-white rounded-lg shadow-sm border hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 group h-full flex flex-col">
      <div className="relative overflow-hidden flex-shrink-0">
        <img
          src={imageUrl}
          alt={product.name || product.title}
          className="w-full h-48 object-contain bg-white group-hover:scale-105 transition-transform duration-300 p-4"
          onError={(e) => {
            // Fallback to default placeholder if image fails to load
            e.currentTarget.src = 'https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=300&h=300&fit=crop';
          }}
        />
        
        {/* Badges */}
        {product.badges && product.badges.length > 0 && (
          <div className="absolute top-2 left-2 space-y-1">
            {product.badges.map((badge) => (
              <span
                key={badge}
                className="inline-block px-2 py-1 bg-[#2874F0] text-white text-xs rounded-full font-medium"
              >
                {badge}
              </span>
            ))}
          </div>
        )}
        
        {/* Discount Badge */}
        {discountPercentage > 0 && (
          <div className="absolute top-2 right-2">
            <span className="bg-[#FFD700] text-[#2874F0] px-2 py-1 rounded-full text-xs font-bold">
              {discountPercentage}% OFF
            </span>
          </div>
        )}
        
        {/* Wishlist Button */}
        <button
          onClick={() => setIsLiked(!isLiked)}
          className="absolute bottom-2 right-2 p-2 bg-white rounded-full shadow-md hover:shadow-lg transition-all"
        >
          <Heart
            className={`h-4 w-4 ${
              isLiked ? 'fill-red-500 text-red-500' : 'text-gray-600'
            } transition-colors`}
          />
        </button>
      </div>
      
      <div className="p-4 flex-1 flex flex-col">
        {/* Brand */}
        {product.brand && (
          <p className="text-sm text-gray-500 mb-1">{product.brand}</p>
        )}
        
        {/* Product Name */}
        <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2 flex-1">
          {product.name || product.title}
        </h3>
        
        {/* Rating */}
        <div className="flex items-center space-x-1 mb-3">
          <div className="flex items-center bg-green-600 text-white px-2 py-1 rounded text-sm">
            <span className="font-medium">
              {typeof product.rating === 'number' ? product.rating : product.rating.rate}
            </span>
            <Star className="h-3 w-3 ml-1 fill-current" />
          </div>
          <span className="text-gray-500 text-sm">
            {(() => {
              const reviews = typeof product.rating === 'number' ? (product.reviews || 0) : product.rating.count;
              const hasReviews = reviews > 0;
              const hasPopularity = product.popularity && product.popularity > 0;
              
              if (hasReviews && hasPopularity) {
                return `(${reviews.toLocaleString()} • (${product.popularity.toLocaleString()}))`;
              } else if (hasReviews) {
                return `(${reviews.toLocaleString()})`;
              } else if (hasPopularity) {
                return `(${product.popularity.toLocaleString()})`;
              } else {
                return '(0)';
              }
            })()}
          </span>
        </div>
        
        {/* Price */}
        <div className="flex items-center space-x-2 mb-4">
          <span className="text-xl font-bold text-gray-900">
            ₹{product.price.toLocaleString()}
          </span>
          {product.originalPrice && (
            <span className="text-gray-500 line-through text-sm">
              ₹{product.originalPrice.toLocaleString()}
            </span>
          )}
        </div>
        
        {/* Add to Cart Button */}
        <button
          onClick={handleAddToCart}
          disabled={isAddingToCart}
          className={`w-full py-2 px-4 rounded-lg font-medium transition-colors disabled:opacity-70 disabled:cursor-not-allowed mt-auto ${
            isAdded 
              ? 'bg-green-500 text-white hover:bg-green-600' 
              : 'bg-[#2874F0] text-white hover:bg-[#1e5cb8]'
          }`}
        >
          {isAddingToCart ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
              Adding...
            </div>
          ) : isAdded ? (
            <div className="flex items-center justify-center">
              <Check className="h-4 w-4 mr-2" />
              Added to Cart
            </div>
          ) : (
            'Add to Cart'
          )}
        </button>
      </div>
    </div>
  );
};

export default ProductCard;