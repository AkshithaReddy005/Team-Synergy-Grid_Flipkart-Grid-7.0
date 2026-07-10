import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Clock, Zap, ShoppingBag, Star } from 'lucide-react';
import { useCart } from '../context/CartContext';

// Helper function to convert time string to seconds
const timeStringToSeconds = (timeString: string): number => {
  const [hours, minutes, seconds] = timeString.split(':').map(Number);
  return hours * 3600 + minutes * 60 + seconds;
};

// Helper function to convert seconds to time string
const secondsToTimeString = (totalSeconds: number): string => {
  if (totalSeconds <= 0) return '00:00:00';
  
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
};

const initialDeals = [
  {
    id: 1,
    title: 'Intel Core Processor',
    originalPrice: 29999,
    discountedPrice: 24999,
    discount: 17,
    image: 'https://images.pexels.com/photos/18105/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    rating: 4.7,
    reviews: 1200,
    initialTimeLeft: '02:45:30',
    sold: 89,
    total: 100,
    badge: 'Flash Sale'
  },
  {
    id: 2,
    title: 'Radius Wireless Earbuds',
    originalPrice: 25999,
    discountedPrice: 22630,
    discount: 13,
    image: 'https://images.pexels.com/photos/3394650/pexels-photo-3394650.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    rating: 4.6,
    reviews: 980,
    initialTimeLeft: '01:23:45',
    sold: 156,
    total: 200,
    badge: 'Limited Time'
  },
  {
    id: 3,
    title: 'Asus WiFi Router',
    originalPrice: 24999,
    discountedPrice: 21290,
    discount: 15,
    image: 'https://images.pexels.com/photos/3780681/pexels-photo-3780681.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    rating: 4.5,
    reviews: 800,
    initialTimeLeft: '03:12:18',
    sold: 234,
    total: 300,
    badge: 'Hot Deal'
  },
  {
    id: 4,
    title: 'Maserati Chronograph Watch',
    originalPrice: 27999,
    discountedPrice: 24400,
    discount: 13,
    image: 'https://images.pexels.com/photos/3785147/pexels-photo-3785147.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    rating: 4.8,
    reviews: 650,
    initialTimeLeft: '00:45:22',
    sold: 45,
    total: 80,
    badge: 'Premium Deal'
  },
  {
    id: 5,
    title: 'HP Gaming Laptop',
    originalPrice: 42999,
    discountedPrice: 38890,
    discount: 10,
    image: 'https://images.pexels.com/photos/1647976/pexels-photo-1647976.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    rating: 4.7,
    reviews: 1100,
    initialTimeLeft: '04:30:15',
    sold: 178,
    total: 250,
    badge: 'Best Seller'
  },
  {
    id: 6,
    title: 'Cobra Paris Watch',
    originalPrice: 16999,
    discountedPrice: 15195,
    discount: 11,
    image: 'https://images.pexels.com/photos/2526878/pexels-photo-2526878.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    rating: 4.4,
    reviews: 500,
    initialTimeLeft: '02:15:40',
    sold: 312,
    total: 400,
    badge: 'Trending'
  }
];

const DealSlider: React.FC = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isHovered, setIsHovered] = useState(false);
  const [addedItems, setAddedItems] = useState<Set<number>>(new Set());
  const { addToCart } = useCart();
  const [timers, setTimers] = useState(() => 
    initialDeals.map(deal => ({
      id: deal.id,
      secondsLeft: timeStringToSeconds(deal.initialTimeLeft)
    }))
  );

  const nextSlide = () => {
    setCurrentIndex((prev) => (prev + 1) % Math.ceil(initialDeals.length / 3));
  };

  const prevSlide = () => {
    setCurrentIndex((prev) => (prev - 1 + Math.ceil(initialDeals.length / 3)) % Math.ceil(initialDeals.length / 3));
  };

  // Timer countdown effect
  useEffect(() => {
    const timer = setInterval(() => {
      setTimers(prevTimers => 
        prevTimers.map(timer => ({
          ...timer,
          secondsLeft: Math.max(0, timer.secondsLeft - 1)
        }))
      );
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Auto-slide effect
  useEffect(() => {
    if (isHovered) return;

    const timer = setInterval(() => {
      nextSlide();
    }, 5000);

    return () => clearInterval(timer);
  }, [isHovered]);

  // Create deals with current timer values
  const deals = initialDeals.map(deal => {
    const timer = timers.find(t => t.id === deal.id);
    return {
      ...deal,
      timeLeft: secondsToTimeString(timer?.secondsLeft || 0)
    };
  });

  const visibleDeals = deals.slice(currentIndex * 3, currentIndex * 3 + 3);

  const handleAddToCart = (deal: any) => {
    const cartItem = {
      id: `deal-${deal.id}`,
      name: deal.title,
      price: deal.discountedPrice,
      originalPrice: deal.originalPrice,
      image: deal.image,
      brand: deal.badge,
      category: 'Flash Deal'
    };
    
    addToCart(cartItem);
    setAddedItems(prev => new Set(prev).add(deal.id));
    
    // Reset the added state after 2 seconds
    setTimeout(() => {
      setAddedItems(prev => {
        const newSet = new Set(prev);
        newSet.delete(deal.id);
        return newSet;
      });
    }, 2000);
  };

  return (
    <div 
      className="py-8"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Zap className="h-6 w-6" style={{ color: '#F8E831' }} />
            <h2 className="text-2xl font-bold" style={{ color: '#0C73EB' }}>Flash Deals</h2>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1 rounded-full" style={{ backgroundColor: '#0C73EB' }}>
            <Clock className="h-4 w-4" style={{ color: '#F8E831' }} />
            <span className="text-sm font-semibold" style={{ color: '#F8E831' }}>Limited Time</span>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={prevSlide}
            className="p-2 rounded-full transition-colors hover:opacity-80"
            style={{ backgroundColor: '#0C73EB' }}
          >
            <ChevronLeft className="h-5 w-5" style={{ color: '#F8E831' }} />
          </button>
          <button
            onClick={nextSlide}
            className="p-2 rounded-full transition-colors hover:opacity-80"
            style={{ backgroundColor: '#0C73EB' }}
          >
            <ChevronRight className="h-5 w-5" style={{ color: '#F8E831' }} />
          </button>
        </div>
      </div>

      <div className="relative overflow-hidden">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {visibleDeals.map((deal) => (
            <div
              key={deal.id}
              className="group bg-white rounded-xl shadow-sm hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 cursor-pointer overflow-hidden border border-gray-100"
            >
              {/* Image Container */}
              <div className="relative h-48 overflow-hidden">
                <img
                  src={deal.image}
                  alt={deal.title}
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                />
                
                {/* Badge */}
                <div className="absolute top-3 left-3 text-white px-2 py-1 rounded-full text-xs font-semibold" style={{ backgroundColor: '#F8E831', color: '#0C73EB' }}>
                  {deal.badge}
                </div>
                
                {/* Discount Badge */}
                <div className="absolute top-3 right-3 text-white px-2 py-1 rounded-full text-xs font-semibold" style={{ backgroundColor: '#0C73EB' }}>
                  {deal.discount}% OFF
                </div>
                
                {/* Progress Bar */}
                <div className="absolute bottom-0 left-0 right-0 text-white p-2" style={{ backgroundColor: 'rgba(248, 232, 49, 0.9)' }}>
                  <div className="flex items-center justify-between text-xs mb-1" style={{ color: '#0C73EB' }}>
                    <span>Sold: {deal.sold}/{deal.total}</span>
                    <span>{Math.round((deal.sold / deal.total) * 100)}%</span>
                  </div>
                  <div className="w-full rounded-full h-1" style={{ backgroundColor: 'rgba(12, 115, 235, 0.3)' }}>
                    <div 
                      className="h-1 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${(deal.sold / deal.total) * 100}%`,
                        backgroundColor: '#0C73EB'
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="p-4">
                {/* Title */}
                <h3 className="font-semibold mb-2 line-clamp-2 transition-colors" style={{ color: '#0C73EB' }}>
                  {deal.title}
                </h3>

                {/* Rating */}
                <div className="flex items-center space-x-2 mb-3">
                  <div className="flex items-center">
                    <Star className="h-4 w-4 fill-current" style={{ color: '#F8E831' }} />
                    <span className="text-sm font-medium ml-1" style={{ color: '#0C73EB' }}>{deal.rating}</span>
                  </div>
                  <span className="text-sm" style={{ color: '#666' }}>({deal.reviews.toLocaleString()})</span>
                </div>

                {/* Price */}
                <div className="flex items-center space-x-3 mb-3">
                  <span className="text-xl font-bold" style={{ color: '#0C73EB' }}>
                    ₹{deal.discountedPrice.toLocaleString()}
                  </span>
                  <span className="line-through" style={{ color: '#999' }}>
                    ₹{deal.originalPrice.toLocaleString()}
                  </span>
                </div>

                {/* Timer */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2 text-sm" style={{ color: '#666' }}>
                    <Clock className="h-4 w-4" style={{ color: '#F8E831' }} />
                    <span>Ends in: {deal.timeLeft}</span>
                  </div>
                </div>

                {/* CTA Button */}
                <button 
                  onClick={() => handleAddToCart(deal)}
                  className="w-full text-white py-2 px-4 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 flex items-center justify-center space-x-2" 
                  style={{ 
                    backgroundColor: addedItems.has(deal.id) ? '#F8E831' : '#0C73EB',
                    color: addedItems.has(deal.id) ? '#0C73EB' : 'white'
                  }}
                >
                  {addedItems.has(deal.id) ? (
                    <>
                      <span>✓ Added to Cart</span>
                    </>
                  ) : (
                    <>
                      <ShoppingBag className="h-4 w-4" />
                      <span>Buy Now</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Dots Indicator */}
      <div className="flex justify-center mt-6 space-x-2">
        {Array.from({ length: Math.ceil(initialDeals.length / 3) }).map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentIndex(index)}
            className={`w-2 h-2 rounded-full transition-all ${
              index === currentIndex ? 'w-6' : ''
            }`}
            style={{ 
              backgroundColor: index === currentIndex ? '#F8E831' : '#0C73EB',
              opacity: index === currentIndex ? 1 : 0.4
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default DealSlider; 