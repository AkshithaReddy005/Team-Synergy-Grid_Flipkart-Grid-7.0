import React from 'react';
import { X } from 'lucide-react';
import { useSearch } from '../context/SearchContext';

const ProductFilters: React.FC = () => {
  const { filters, setFilters } = useSearch();

  const handlePriceRangeChange = (min: number, max: number) => {
    setFilters({ ...filters, priceRange: [min, max] });
  };

  const handleRatingChange = (rating: number) => {
    setFilters({ ...filters, rating });
  };

  const handleOffersToggle = () => {
    setFilters({ ...filters, offers: !filters.offers });
  };

  const clearAllFilters = () => {
    setFilters({
      category: [],
      priceRange: [0, 100000],
      rating: 0,
      offers: false,
    });
  };

  const hasActiveFilters = 
    filters.rating > 0 || 
    filters.offers ||
    filters.priceRange[0] > 0 || 
    filters.priceRange[1] < 100000;

  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-lg">Filters</h3>
        {hasActiveFilters && (
          <button
            onClick={clearAllFilters}
            className="text-[#2874F0] text-sm hover:underline flex items-center bg-blue-50 px-2 py-1 rounded"
          >
            <X className="h-4 w-4 mr-1" />
            Clear All
          </button>
        )}
      </div>

      {/* Clear Filters Button - Always visible when there are active filters */}
      {hasActiveFilters && (
        <div className="mb-4 p-3 bg-blue-50 rounded-lg">
          <button
            onClick={clearAllFilters}
            className="w-full text-[#2874F0] text-sm font-medium hover:bg-blue-100 py-2 px-3 rounded border border-blue-200"
          >
            Clear All Filters
          </button>
        </div>
      )}

      {/* Price Range */}
      <div className="mb-6">
        <h4 className="font-medium mb-3">Price Range</h4>
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <input
              type="number"
              placeholder="Min"
              value={filters.priceRange[0]}
              onChange={(e) => handlePriceRangeChange(parseInt(e.target.value) || 0, filters.priceRange[1])}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
            />
            <span className="text-gray-500">to</span>
            <input
              type="number"
              placeholder="Max"
              value={filters.priceRange[1]}
              onChange={(e) => handlePriceRangeChange(filters.priceRange[0], parseInt(e.target.value) || 100000)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>₹{filters.priceRange[0].toLocaleString()}</span>
            <span>₹{filters.priceRange[1].toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* Rating */}
      <div className="mb-6">
        <h4 className="font-medium mb-3">Customer Rating</h4>
        <div className="space-y-2">
          {[0, 4, 3, 2, 1].map((rating) => (
            <label key={rating} className="flex items-center">
              <input
                type="radio"
                name="rating"
                checked={filters.rating === rating}
                onChange={() => handleRatingChange(rating)}
                className="text-[#2874F0] focus:ring-[#2874F0] mr-2"
              />
              <div className="flex items-center">
                {rating === 0 ? (
                  <span className="text-sm text-gray-500">All Ratings</span>
                ) : (
                  <>
                    <span className="text-sm mr-1">{rating}</span>
                    <span className="text-[#FFD700]">★</span>
                    <span className="text-sm text-gray-500 ml-1">& above</span>
                  </>
                )}
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Offers */}
      <div>
        <h4 className="font-medium mb-3">Offers</h4>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={filters.offers}
            onChange={handleOffersToggle}
            className="rounded border-gray-300 text-[#2874F0] focus:ring-[#2874F0] mr-2"
          />
          <span className="text-sm">Show only items with offers</span>
        </label>
      </div>
    </div>
  );
};

export default ProductFilters;