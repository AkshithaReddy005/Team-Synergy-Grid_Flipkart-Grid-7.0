import React from 'react';
import { useNavigate } from 'react-router-dom';

const Footer: React.FC = () => {
  const navigate = useNavigate();

  const handleCategoryClick = (categoryName: string) => {
    navigate(`/search?q=${encodeURIComponent(categoryName)}`);
    // Scroll to top of the page
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="bg-[#2874F0] text-white mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-1">
          {/* Brand */}
          <div>
            <div className="flex items-center space-x-3 mb-4">
              <div className="flex items-center">
                <img 
                  src="/logo.jpg" 
                  alt="Synergy Kart Logo" 
                  className="h-10 w-auto object-contain"
                />
              </div>
              <div>
                <div className="font-bold text-xl">SYNERGY KART</div>
                <div className="text-sm text-[#FFEA00]">Explore Plus</div>
              </div>
            </div>
            <p className="text-blue-200 mb-4 max-w-md">
              A modern e-commerce platform with advanced search capabilities, responsive design, and intuitive user experience.
            </p>
          </div>

          {/* Categories Column 1 */}
          <div className="text-right">
            <h3 className="font-semibold text-lg mb-4 text-right">Categories</h3>
            <ul className="space-y-2">
              <li>
                <button 
                  onClick={() => handleCategoryClick('Clothing')}
                  className="text-blue-200 hover:text-white transition-colors"
                >
                  Clothing
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleCategoryClick('Jewellery')}
                  className="text-blue-200 hover:text-white transition-colors"
                >
                  Jewellery
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleCategoryClick('Footwear')}
                  className="text-blue-200 hover:text-white transition-colors"
                >
                  Footwear
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleCategoryClick('Mobile Phones')}
                  className="text-blue-200 hover:text-white transition-colors"
                >
                  Mobile Phones
                </button>
              </li>
            </ul>
          </div>

          {/* Categories Column 2 */}
          <div className="text-right">
            <h3 className="font-semibold text-lg mb-4 opacity-0">Categories</h3>
            <ul className="space-y-2">
              <li>
                <button 
                  onClick={() => handleCategoryClick('Computers')}
                  className="text-blue-200 hover:text-white transition-colors"
                >
                  Computers
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleCategoryClick('Automotive')}
                  className="text-blue-200 hover:text-white transition-colors"
                >
                  Automotive
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleCategoryClick('Home Decor')}
                  className="text-blue-200 hover:text-white transition-colors"
                >
                  Home Decor
                </button>
              </li>
              <li>
                <button 
                  onClick={() => handleCategoryClick('Home Furnishing')}
                  className="text-blue-200 hover:text-white transition-colors"
                >
                  Home Furnishing
                </button>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-blue-400 mt-8 pt-8 text-center">
          <p className="text-blue-200 text-sm">
            © 2025 Synergy Kart.
          </p>
          <p className="text-blue-300 text-xs mt-2">
            This is a prototype showcasing modern e-commerce functionality and UI/UX design.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;