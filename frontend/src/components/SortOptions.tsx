import React from 'react';
import { ChevronDown } from 'lucide-react';
import { useSearch } from '../context/SearchContext';

const sortOptions = [
  { value: 'relevance', label: 'Relevance' },
  { value: 'popularity', label: 'Popularity' },
  { value: 'price-high-low', label: 'Price: High to Low' },
  { value: 'price-low-high', label: 'Price: Low to High' },
  { value: 'rating', label: 'Customer Rating' },
];

const SortOptions: React.FC = () => {
  const { sortBy, setSortBy } = useSearch();

  const handleSortChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newSortBy = e.target.value;
    console.log('Sort changed from', sortBy, 'to', newSortBy);
    setSortBy(newSortBy);
  };

  return (
    <div className="relative">
      <select
        value={sortBy}
        onChange={handleSortChange}
        className="appearance-none bg-white border border-gray-300 rounded-lg px-4 py-2 pr-8 focus:outline-none focus:ring-2 focus:ring-[#2874F0] focus:border-transparent"
      >
        {sortOptions.map((option) => (
          <option key={option.value} value={option.value}>
            Sort by: {option.label}
          </option>
        ))}
      </select>
      <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
    </div>
  );
};

export default SortOptions;