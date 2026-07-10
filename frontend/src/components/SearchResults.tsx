import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Filter, Grid, List, ChevronLeft, ChevronRight } from 'lucide-react';
import { useSearch } from '../context/SearchContext';
import ProductGrid from './ProductGrid';
import ProductFilters from './ProductFilters';
import SortOptions from './SortOptions';
import { searchProducts as searchRealProducts } from '../api/realApi';

const SearchResults: React.FC = () => {
  const [searchParams] = useSearchParams();
  const query = searchParams.get('q') || '';
  const [products, setProducts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showFilters, setShowFilters] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalProducts, setTotalProducts] = useState(0);
  const { sortBy } = useSearch();

  const productsPerPage = 20;

  useEffect(() => {
    const fetchProducts = async () => {
      setLoading(true);
      setCurrentPage(1); // Reset to first page on new search
      console.log('Fetching products for query:', query);
      try {
        const results = await searchRealProducts(query, {}, sortBy); // Always fetch with default sort
        console.log('Search results received:', results.length, 'products');
        console.log('First few products:', results.slice(0, 3).map(p => ({ name: p.name, price: p.price, rating: p.rating })));
        setProducts(results);
        setTotalProducts(results.length);
        setTotalPages(Math.ceil(results.length / productsPerPage));
      } catch (error) {
        console.error('Search failed:', error);
        setProducts([]);
        setTotalProducts(0);
        setTotalPages(1);
      }
      setLoading(false);
    };

    fetchProducts();
  }, [query, sortBy]); // Remove sortBy from dependencies

  const toggleFilters = () => {
    setShowFilters(!showFilters);
  };

  // Sort products based on sortBy
  const sortProducts = (productsToSort: any[]) => {
    const sortedProducts = [...productsToSort];
    
    switch (sortBy) {
      case 'relevance':
        // Keep original order (already sorted by relevance from backend)
        return sortedProducts;
      case 'price-high-low':
        return sortedProducts.sort((a, b) => b.price - a.price);
      case 'price-low-high':
        return sortedProducts.sort((a, b) => a.price - b.price);
      case 'rating':
        return sortedProducts.sort((a, b) => b.rating - a.rating);
      case 'popularity':
        // Sort by popularity (higher popularity first)
        return sortedProducts.sort((a, b) => {
          const popularityA = a.popularity || 0;
          const popularityB = b.popularity || 0;
          return popularityB - popularityA;
        });
      default:
        // Keep original order (already sorted by relevance from backend)
        return sortedProducts;
    }
  };

  const sortedProducts = sortProducts(products);
  
  // Debug: Log first few sorted products
  if (sortedProducts.length > 0) {
    console.log('Sort applied:', sortBy);
    console.log('First 3 sorted products:', sortedProducts.slice(0, 3).map(p => ({ name: p.name, price: p.price, rating: p.rating })));
  }
  
  // Calculate pagination
  const startIndex = (currentPage - 1) * productsPerPage;
  const endIndex = startIndex + productsPerPage;
  const currentProducts = sortedProducts.slice(startIndex, endIndex);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    const pages = [];
    const maxVisiblePages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    // Previous button
    if (currentPage > 1) {
      pages.push(
        <button
          key="prev"
          onClick={() => handlePageChange(currentPage - 1)}
          className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-l-md hover:bg-gray-50"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
      );
    }

    // Page numbers
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          onClick={() => handlePageChange(i)}
          className={`px-3 py-2 text-sm font-medium border ${
            i === currentPage
              ? 'bg-[#2874F0] text-white border-[#2874F0]'
              : 'bg-white text-gray-500 border-gray-300 hover:bg-gray-50'
          }`}
        >
          {i}
        </button>
      );
    }

    // Next button
    if (currentPage < totalPages) {
      pages.push(
        <button
          key="next"
          onClick={() => handlePageChange(currentPage + 1)}
          className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-r-md hover:bg-gray-50"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      );
    }

    return (
      <div className="flex items-center justify-center mt-8">
        <div className="flex items-center space-x-0">
          {pages}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-[#2874F0] border-t-transparent mx-auto mb-4"></div>
            <p className="text-gray-600">Searching for products...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Search Results Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          {query.charAt(0).toUpperCase() + query.slice(1)} Products
        </h1>
        <p className="text-gray-600">
          Showing {startIndex + 1}-{Math.min(endIndex, totalProducts)} of {totalProducts} {totalProducts === 1 ? 'product' : 'products'}
          {totalPages > 1 && ` (Page ${currentPage} of ${totalPages})`}
        </p>
      </div>

      {/* Controls Bar */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 space-y-4 md:space-y-0">
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleFilters}
            className="flex items-center space-x-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <Filter className="h-4 w-4" />
            <span>Filters</span>
          </button>
          
          <div className="hidden md:flex items-center space-x-2">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded ${
                viewMode === 'grid' ? 'bg-[#2874F0] text-white' : 'bg-gray-100 text-gray-600'
              }`}
            >
              <Grid className="h-4 w-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded ${
                viewMode === 'list' ? 'bg-[#2874F0] text-white' : 'bg-gray-100 text-gray-600'
              }`}
            >
              <List className="h-4 w-4" />
            </button>
          </div>
        </div>

        <SortOptions />
      </div>

      <div className="flex gap-6">
        {/* Filters Sidebar */}
        <div className={`${showFilters ? 'block' : 'hidden'} md:block w-full md:w-64 flex-shrink-0`}>
          <ProductFilters />
        </div>

                            {/* Products Grid */}
          <div className="flex-1">
            {currentProducts.length > 0 ? (
              <>
                <ProductGrid products={currentProducts} viewMode={viewMode} />
                {renderPagination()}
              </>
            ) : (
              <div className="text-center py-12">
                <div className="mb-4">
                  <img
                    src="https://images.pexels.com/photos/3683107/pexels-photo-3683107.jpeg?auto=compress&cs=tinysrgb&w=300&h=200&dpr=1"
                    alt="No results"
                    className="mx-auto w-64 h-48 object-cover rounded-lg opacity-50"
                  />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  No products found
                </h3>
                <p className="text-gray-600 mb-4">
                  Try adjusting your search criteria or browse our popular categories
                </p>
              <div className="flex flex-wrap justify-center gap-2">
                {['iPhone', 'Samsung', 'Laptop', 'Headphones'].map((suggestion) => (
                  <button
                    key={suggestion}
                    className="px-4 py-2 bg-[#2874F0] text-white rounded-lg hover:bg-[#1e5cb8] transition-colors"
                    onClick={() => window.location.href = `/search?q=${suggestion}`}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchResults;