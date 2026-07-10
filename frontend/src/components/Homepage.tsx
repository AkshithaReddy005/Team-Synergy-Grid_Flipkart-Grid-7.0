import React from 'react';
import HeroBanner from './HeroBanner';
import ProductGrid from './ProductGrid';
import DealSlider from './DealSlider';

// Top Picks products from famous brands
const topPicksProducts = [
  {
    id: 'top-1',
    name: 'Apple iPhone 15 Pro',
    brand: 'Apple',
    price: 149999,
    originalPrice: 159999,
    rating: 4.8,
    numReviews: 12500,
    image: 'https://images.pexels.com/photos/1647976/pexels-photo-1647976.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Up to ₹10,000 off',
    category: 'Electronics'
  },
  {
    id: 'top-2',
    name: 'Samsung Galaxy S24 Ultra',
    brand: 'Samsung',
    price: 129999,
    originalPrice: 139999,
    rating: 4.7,
    numReviews: 8900,
    image: 'https://images.pexels.com/photos/699122/pexels-photo-699122.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Free Galaxy Buds',
    category: 'Electronics'
  },
  {
    id: 'top-3',
    name: 'Nike Air Max 270',
    brand: 'Nike',
    price: 12995,
    originalPrice: 15995,
    rating: 4.6,
    numReviews: 3400,
    image: 'https://images.pexels.com/photos/2526878/pexels-photo-2526878.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Up to 20% off',
    category: 'Footwear'
  },
  {
    id: 'top-4',
    name: 'Adidas Ultraboost 22',
    brand: 'Adidas',
    price: 18995,
    originalPrice: 22995,
    rating: 4.5,
    numReviews: 2100,
    image: 'https://images.pexels.com/photos/2526878/pexels-photo-2526878.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Limited Time Deal',
    category: 'Footwear'
  },
  {
    id: 'top-5',
    name: 'Sony WH-1000XM5',
    brand: 'Sony',
    price: 29990,
    originalPrice: 34990,
    rating: 4.9,
    numReviews: 5600,
    image: 'https://images.pexels.com/photos/3394650/pexels-photo-3394650.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Premium Audio',
    category: 'Electronics'
  },
  {
    id: 'top-6',
    name: 'Dell XPS 13 Plus',
    brand: 'Dell',
    price: 189990,
    originalPrice: 209990,
    rating: 4.7,
    numReviews: 1200,
    image: 'https://images.pexels.com/photos/18105/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Student Discount',
    category: 'Electronics'
  },
  {
    id: 'top-7',
    name: 'Rolex Submariner',
    brand: 'Rolex',
    price: 899990,
    originalPrice: 999990,
    rating: 4.9,
    numReviews: 450,
    image: 'https://images.pexels.com/photos/3785147/pexels-photo-3785147.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Luxury Timepiece',
    category: 'Accessories'
  },
  {
    id: 'top-8',
    name: 'Canon EOS R5',
    brand: 'Canon',
    price: 399990,
    originalPrice: 449990,
    rating: 4.8,
    numReviews: 890,
    image: 'https://images.pexels.com/photos/1647976/pexels-photo-1647976.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Professional Grade',
    category: 'Electronics'
  },
  {
    id: 'top-9',
    name: 'Levi\'s 501 Original',
    brand: 'Levi\'s',
    price: 3995,
    originalPrice: 4995,
    rating: 4.4,
    numReviews: 7800,
    image: 'https://images.pexels.com/photos/7679720/pexels-photo-7679720.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Classic Denim',
    category: 'Clothing'
  },
  {
    id: 'top-10',
    name: 'Bose QuietComfort 45',
    brand: 'Bose',
    price: 24990,
    originalPrice: 29990,
    rating: 4.6,
    numReviews: 3200,
    image: 'https://images.pexels.com/photos/3394650/pexels-photo-3394650.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Noise Cancelling',
    category: 'Electronics'
  },
  {
    id: 'top-11',
    name: 'MacBook Pro M3',
    brand: 'Apple',
    price: 249990,
    originalPrice: 269990,
    rating: 4.9,
    numReviews: 2100,
    image: 'https://images.pexels.com/photos/18105/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Latest M3 Chip',
    category: 'Electronics'
  },
  {
    id: 'top-12',
    name: 'Ray-Ban Aviator',
    brand: 'Ray-Ban',
    price: 15995,
    originalPrice: 18995,
    rating: 4.5,
    numReviews: 4200,
    image: 'https://images.pexels.com/photos/3785147/pexels-photo-3785147.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Classic Style',
    category: 'Accessories'
  },
  {
    id: 'top-13',
    name: 'Sony PlayStation 5',
    brand: 'Sony',
    price: 49990,
    originalPrice: 54990,
    rating: 4.8,
    numReviews: 8900,
    image: 'https://images.pexels.com/photos/1647976/pexels-photo-1647976.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Gaming Console',
    category: 'Electronics'
  },
  {
    id: 'top-14',
    name: 'Puma RS-X',
    brand: 'Puma',
    price: 8995,
    originalPrice: 11995,
    rating: 4.3,
    numReviews: 1800,
    image: 'https://images.pexels.com/photos/2526878/pexels-photo-2526878.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    offers: 'Retro Style',
    category: 'Footwear'
  }
];

const Homepage: React.FC = () => {

  return (
    <div className="space-y-8">
      {/* Hero Banner Slideshow */}
      <HeroBanner />
      
      {/* Flash Deals Slider */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <DealSlider />
      </div>
      
      {/* Top Picks from Synergy Kart */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Top Picks from Synergy Kart</h2>
          <p className="text-gray-600">Curated selection of premium products from world-renowned brands</p>
        </div>
        
        <ProductGrid products={topPicksProducts} />
      </div>
    </div>
  );
};

export default Homepage;