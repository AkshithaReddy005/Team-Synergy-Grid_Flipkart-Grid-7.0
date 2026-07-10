import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronDown } from 'lucide-react';

const CategoryNav: React.FC = () => {
  const navigate = useNavigate();

  const handleCategoryClick = (categoryName: string) => {
    navigate(`/search?q=${encodeURIComponent(categoryName)}`);
  };
    const categories = [
    { 
      name: 'Clothing', 
      hasDropdown: false,
      image: 'https://images.pexels.com/photos/7679720/pexels-photo-7679720.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
      count: '5,317'
    },
    { 
      name: 'Jewellery', 
      hasDropdown: false,
      image: 'https://images.pexels.com/photos/3785147/pexels-photo-3785147.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
      count: '3,514'
    },
    { 
      name: 'Footwear', 
      hasDropdown: false,
      image: 'https://images.pexels.com/photos/2526878/pexels-photo-2526878.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
      count: '1,093'
    },
    { 
      name: 'Mobiles', 
      hasDropdown: false,
      image: 'https://images.pexels.com/photos/699122/pexels-photo-699122.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
      count: '1,084'
    },
    { 
      name: 'Computers', 
      hasDropdown: false,
      image: 'https://images.pexels.com/photos/18105/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
      count: '561'
    },
    { 
      name: 'Automotive', 
      hasDropdown: false,
      image: 'https://images.pexels.com/photos/3802510/pexels-photo-3802510.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
      count: '971'
    },
    { 
      name: 'Home', 
      hasDropdown: false,
      image: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
      count: '896'
    },
    { 
      name: 'Furniture',
      hasDropdown: false,
      image: 'https://images.pexels.com/photos/1571468/pexels-photo-1571468.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
      count: '696'
    },
  ];

  return (
    <div className="shadow-md border-b py-6" style={{ backgroundColor: '#F8E831' }}>
      <div className="w-full mx-auto px-4 sm:px-6 lg:px-8">
        {/* Heading and Subheading */}
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Shop by Category</h2>
          <p className="text-gray-600">Explore our wide range of products</p>
        </div>
        
        <div className="flex items-center justify-center h-24">
          <div className="grid grid-cols-4 md:grid-cols-8 gap-4 md:gap-6 lg:gap-8 w-full max-w-7xl">
            {categories.map((category, index) => {
              return (
                <div 
                  key={index} 
                  className="flex flex-col items-center space-y-2 cursor-pointer hover:text-[#2874F0] transition-colors group"
                  onClick={() => handleCategoryClick(category.name)}
                >
                  <div className="relative w-16 h-16 md:w-20 md:h-20">
                    <img 
                      src={category.image} 
                      alt={category.name}
                      className="w-16 h-16 md:w-20 md:h-20 object-cover rounded-lg shadow-md group-hover:shadow-lg transition-shadow"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-10 rounded-lg group-hover:bg-opacity-5 transition-all"></div>
                  </div>
                  <div className="flex flex-col items-center space-y-1 w-full">
                    <span className="text-xs md:text-sm font-medium text-center leading-tight">{category.name}</span>
                    {category.hasDropdown && (
                      <ChevronDown className="h-3 w-3 opacity-60 group-hover:opacity-100 flex-shrink-0" />
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CategoryNav; 