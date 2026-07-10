import React, { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Play, Pause, Star, ShoppingBag, Zap, Gift } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const banners = [
  {
    id: 1,
    title: 'Clothing Bonanza',
    subtitle: 'Starting from ₹199',
    description: 'Trending fashion for men, women & kids. Latest styles with amazing discounts.',
    image: 'https://images.pexels.com/photos/7679720/pexels-photo-7679720.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    gradient: 'from-[#0C73EB] via-[#1a7ff0] to-[#2a8bf5]',
    bgColor: 'bg-gradient-to-br from-[#0C73EB] via-[#1a7ff0] to-[#2a8bf5]',
    cta: 'Shop Clothing',
    badge: 'Up to 80% Off',
    icon: '👕',
    features: ['Free Shipping', 'Easy Returns', 'Size Guide'],
    accentColor: 'from-yellow-400 to-orange-500',
    category: 'Clothing'
  },
  {
    id: 2,
    title: 'Mobile Phones',
    subtitle: 'Latest Smartphones',
    description: 'iPhone, Samsung, OnePlus & more. EMI available with instant discounts.',
    image: 'https://images.pexels.com/photos/699122/pexels-photo-699122.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    gradient: 'from-[#152336] via-[#1a2a4a] to-[#2a3f6b]',
    bgColor: 'bg-gradient-to-br from-[#152336] via-[#1a2a4a] to-[#2a3f6b]',
    cta: 'Explore Mobiles',
    badge: 'EMI from ₹999',
    icon: '📱',
    features: ['EMI Available', 'Free Returns', 'Genuine Products'],
    accentColor: 'from-[#0C73EB] to-[#1a7ff0]',
    category: 'Mobile Phones'
  },
  {
    id: 3,
    title: 'Computers & Laptops',
    subtitle: 'Gaming & Business',
    description: 'Dell, HP, Lenovo laptops with latest processors. Student discounts available.',
    image: 'https://images.pexels.com/photos/18105/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    gradient: 'from-[#0C73EB] via-[#1a7ff0] to-[#2a8bf5]',
    bgColor: 'bg-gradient-to-br from-[#0C73EB] via-[#1a7ff0] to-[#2a8bf5]',
    cta: 'Shop Computers',
    badge: 'Student Discount',
    icon: '💻',
    features: ['Student Discount', 'Free Setup', 'Extended Warranty'],
    accentColor: 'from-yellow-400 to-orange-500',
    category: 'Computers'
  },
  {
    id: 4,
    title: 'Home Decor',
    subtitle: 'Transform Your Space',
    description: 'Beautiful furniture, decor items & lifestyle products for your dream home.',
    image: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    gradient: 'from-[#152336] via-[#1a2a4a] to-[#2a3f6b]',
    bgColor: 'bg-gradient-to-br from-[#152336] via-[#1a2a4a] to-[#2a3f6b]',
    cta: 'Explore Home',
    badge: 'Free Installation',
    icon: '🏠',
    features: ['Free Installation', 'Premium Quality', 'Easy Assembly'],
    accentColor: 'from-[#0C73EB] to-[#1a7ff0]',
    category: 'Home Decor'
  }
];

const HeroBanner: React.FC = () => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isHovered, setIsHovered] = useState(false);
  const navigate = useNavigate();

  const handleCategoryClick = (category: string) => {
    navigate(`/search?q=${encodeURIComponent(category)}`);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const nextSlide = useCallback(() => {
    setCurrentSlide((prev) => (prev + 1) % banners.length);
  }, []);

  const prevSlide = useCallback(() => {
    setCurrentSlide((prev) => (prev - 1 + banners.length) % banners.length);
  }, []);

  const goToSlide = useCallback((index: number) => {
    setCurrentSlide(index);
  }, []);

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  useEffect(() => {
    if (!isPlaying || isHovered) return;

    const timer = setInterval(() => {
      nextSlide();
    }, 5000);

    return () => clearInterval(timer);
  }, [isPlaying, isHovered, nextSlide]);

  return (
    <>
      <style>
        {`
          .firework {
            position: relative;
            width: 0;
            height: 0;
          }
          
          .firework-spark {
            position: absolute;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: radial-gradient(circle, var(--spark-color, #ff6b6b) 0%, transparent 60%);
            box-shadow: 0 0 10px var(--spark-color, #ff6b6b);
            animation: firework-explode 3s infinite;
            animation-delay: var(--delay);
            opacity: 0;
          }
          
          .firework-spark:nth-child(1) { --spark-color: #ff6b6b; transform: translate(0, 0) rotate(0deg); }
          .firework-spark:nth-child(2) { --spark-color: #4ecdc4; transform: translate(20px, -20px) rotate(45deg); }
          .firework-spark:nth-child(3) { --spark-color: #45b7d1; transform: translate(0, -40px) rotate(90deg); }
          .firework-spark:nth-child(4) { --spark-color: #96ceb4; transform: translate(-20px, -20px) rotate(135deg); }
          .firework-spark:nth-child(5) { --spark-color: #feca57; transform: translate(-40px, 0) rotate(180deg); }
          .firework-spark:nth-child(6) { --spark-color: #ff9ff3; transform: translate(-20px, 20px) rotate(225deg); }
          .firework-spark:nth-child(7) { --spark-color: #54a0ff; transform: translate(0, 40px) rotate(270deg); }
          .firework-spark:nth-child(8) { --spark-color: #5f27cd; transform: translate(20px, 20px) rotate(315deg); }
          
          @keyframes firework-explode {
            0% {
              opacity: 0;
              transform: scale(0) rotate(0deg);
            }
            10% {
              opacity: 1;
              transform: scale(1) rotate(0deg);
            }
            50% {
              opacity: 1;
              transform: scale(2) rotate(180deg);
            }
            100% {
              opacity: 0;
              transform: scale(0) rotate(360deg);
            }
          }
        `}
      </style>
      <div 
        className="relative h-80 md:h-96 lg:h-[500px] overflow-hidden bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800 shadow-2xl"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{
          '--firework-colors': 'hsl(0, 100%, 50%), hsl(60, 100%, 50%), hsl(120, 100%, 50%), hsl(180, 100%, 50%), hsl(240, 100%, 50%), hsl(300, 100%, 50%)'
        } as any}
      >
      {/* Animated Background Pattern */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%23ffffff%22%20fill-opacity%3D%220.1%22%3E%3Ccircle%20cx%3D%2230%22%20cy%3D%2230%22%20r%3D%222%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] animate-pulse"></div>
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-8 left-8 w-3 h-3 bg-white rounded-full opacity-60 animate-bounce" style={{ animationDelay: '0s' }}></div>
        <div className="absolute top-16 right-16 w-2 h-2 bg-yellow-300 rounded-full opacity-80 animate-bounce" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-16 left-16 w-2 h-2 bg-pink-300 rounded-full opacity-70 animate-bounce" style={{ animationDelay: '2s' }}></div>
        <div className="absolute bottom-8 right-8 w-2 h-2 bg-cyan-300 rounded-full opacity-60 animate-bounce" style={{ animationDelay: '0.5s' }}></div>
      </div>

      {/* Fireworks Animation */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-10">
        {/* Firework 1 */}
        <div className="absolute top-20 left-20">
          <div className="firework">
            <div className="firework-spark" style={{ '--delay': '0s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.1s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.2s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.3s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.4s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.5s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.6s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.7s' } as any}></div>
          </div>
        </div>

        {/* Firework 2 */}
        <div className="absolute top-32 right-32">
          <div className="firework">
            <div className="firework-spark" style={{ '--delay': '0.5s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.6s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.7s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.8s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '0.9s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.1s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.2s' } as any}></div>
          </div>
        </div>

        {/* Firework 3 */}
        <div className="absolute bottom-32 left-32">
          <div className="firework">
            <div className="firework-spark" style={{ '--delay': '1s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.1s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.2s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.3s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.4s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.5s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.6s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.7s' } as any}></div>
          </div>
        </div>

        {/* Firework 4 */}
        <div className="absolute bottom-20 right-20">
          <div className="firework">
            <div className="firework-spark" style={{ '--delay': '1.5s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.6s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.7s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.8s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '1.9s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.1s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.2s' } as any}></div>
          </div>
        </div>

        {/* Firework 5 */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <div className="firework">
            <div className="firework-spark" style={{ '--delay': '2s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.1s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.2s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.3s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.4s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.5s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.6s' } as any}></div>
            <div className="firework-spark" style={{ '--delay': '2.7s' } as any}></div>
          </div>
        </div>
      </div>

      {/* Slides Container */}
      <div
        className="flex transition-transform duration-1000 ease-in-out h-full"
        style={{ transform: `translateX(-${currentSlide * 100}%)` }}
      >
        {banners.map((banner, index) => (
          <div
            key={banner.id}
            className="min-w-full h-full relative"
          >
            {/* Background Image with Overlay */}
            <div 
              className="absolute inset-0 bg-cover bg-center bg-no-repeat"
              style={{ backgroundImage: `url(${banner.image})` }}
            >
              <div className={`absolute inset-0 ${banner.bgColor} bg-opacity-80`} />
            </div>

            {/* Content */}
            <div className="absolute inset-0 flex items-center">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                  {/* Text Content */}
                  <div className="text-white space-y-4">
                    {/* Badge */}
                    <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-full px-4 py-2 w-fit shadow-lg transform hover:scale-105 transition-all duration-300 animate-bounce">
                      <Zap className="h-4 w-4 animate-pulse" />
                      <span className="text-xs font-bold">{banner.badge}</span>
                    </div>

                    {/* Title */}
                    <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold leading-tight bg-gradient-to-r from-white to-gray-200 bg-clip-text text-transparent animate-pulse">
                      {banner.title}
                    </h1>

                    {/* Subtitle */}
                    <p className="text-lg md:text-xl lg:text-2xl font-medium text-gray-200 animate-fade-in delay-200">
                      {banner.subtitle}
                    </p>

                    {/* Description */}
                    <p className="text-base text-gray-300 max-w-md animate-fade-in delay-300">
                      {banner.description}
                    </p>

                    {/* Features */}
                    <div className="flex flex-wrap gap-2 animate-fade-in delay-400">
                      {banner.features.map((feature, idx) => (
                        <div key={idx} className="flex items-center space-x-2 bg-white bg-opacity-20 backdrop-blur-sm rounded-lg px-3 py-2 border border-white border-opacity-30 hover:bg-opacity-30 transition-all duration-300 transform hover:scale-105">
                          <Star className="h-3 w-3 text-yellow-300 animate-spin" style={{ animationDuration: '3s' }} />
                          <span className="text-xs font-medium">{feature}</span>
                        </div>
                      ))}
                    </div>

                    {/* CTA Button */}
                    <button 
                      onClick={() => handleCategoryClick(banner.category)}
                      className="group bg-gradient-to-r from-white to-gray-100 text-gray-900 px-8 py-4 rounded-full font-bold text-base hover:from-gray-100 hover:to-white transition-all duration-300 transform hover:scale-110 shadow-2xl hover:shadow-3xl animate-fade-in delay-500 flex items-center space-x-2"
                    >
                      <ShoppingBag className="h-5 w-5 group-hover:animate-bounce" />
                      <span>{banner.cta}</span>
                      <ChevronRight className="h-5 w-5 group-hover:translate-x-2 transition-transform duration-300" />
                    </button>
                  </div>

                  {/* Visual Element */}
                  <div className="hidden lg:flex justify-center items-center">
                    <div className="relative">
                      {/* Main Icon Container */}
                      <div className="w-48 h-48 bg-gradient-to-br from-white bg-opacity-30 to-transparent backdrop-blur-md rounded-full flex items-center justify-center border-4 border-white border-opacity-50 animate-pulse shadow-2xl">
                        <span className="text-7xl animate-bounce" style={{ animationDuration: '2s' }}>{banner.icon}</span>
                      </div>
                      
                      {/* Floating Elements */}
                      <div className="absolute -top-4 -right-4 w-16 h-16 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center animate-bounce shadow-lg">
                        <Gift className="h-8 w-8 text-white animate-pulse" />
                      </div>
                      <div className="absolute -bottom-4 -left-4 w-12 h-12 bg-gradient-to-r from-red-400 to-pink-500 rounded-full flex items-center justify-center animate-bounce shadow-lg" style={{ animationDelay: '0.5s' }}>
                        <span className="text-white font-bold text-lg animate-pulse">%</span>
                      </div>
                      
                      {/* Additional Floating Elements */}
                      <div className="absolute top-1/2 -right-8 w-6 h-6 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '1s' }}></div>
                      <div className="absolute top-1/2 -left-8 w-5 h-5 bg-pink-400 rounded-full animate-bounce" style={{ animationDelay: '1.5s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Navigation Controls */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex items-center space-x-6">
        {/* Play/Pause Button */}
        <button
          onClick={togglePlayPause}
          className="bg-white bg-opacity-30 hover:bg-opacity-50 text-white p-4 rounded-full transition-all backdrop-blur-sm shadow-lg hover:scale-110 transform"
        >
          {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6" />}
        </button>

        {/* Dots Indicator */}
        <div className="flex space-x-3">
          {banners.map((_, index) => (
            <button
              key={index}
              onClick={() => goToSlide(index)}
              className={`w-4 h-4 rounded-full transition-all duration-300 shadow-lg ${
                index === currentSlide 
                  ? 'bg-white scale-150 shadow-xl' 
                  : 'bg-white bg-opacity-50 hover:bg-opacity-75 hover:scale-125'
              }`}
            />
          ))}
        </div>

        {/* Slide Counter */}
        <div className="bg-white bg-opacity-30 backdrop-blur-sm rounded-full px-4 py-2 text-white text-sm font-bold shadow-lg">
          {currentSlide + 1} / {banners.length}
        </div>
      </div>

      {/* Navigation Arrows */}
      <button
        onClick={prevSlide}
        className="absolute left-4 top-1/2 transform -translate-y-1/2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white p-3 rounded-full transition-all backdrop-blur-sm hover:scale-110"
      >
        <ChevronLeft className="h-6 w-6" />
      </button>
      <button
        onClick={nextSlide}
        className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-white bg-opacity-20 hover:bg-opacity-30 text-white p-3 rounded-full transition-all backdrop-blur-sm hover:scale-110"
      >
        <ChevronRight className="h-6 w-6" />
      </button>

      {/* Progress Bar */}
      <div className="absolute bottom-0 left-0 w-full h-1 bg-white bg-opacity-20">
        <div 
          className="h-full bg-white transition-all duration-300 ease-out"
          style={{ width: `${((currentSlide + 1) / banners.length) * 100}%` }}
        />
      </div>
    </div>
    </>
  );
};

export default HeroBanner;