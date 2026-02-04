import React, { useState, useRef, useEffect } from 'react';
import { Search, X, Loader2 } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSearch } from '../context/SearchContext';
import { getAutosuggestions } from '../services/api';

interface Suggestion {
  id: string;
  text: string;
  type: 'product' | 'category' | 'brand';
  isCorrection?: boolean;
}

const SearchBar: React.FC = () => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [initialLoadComplete, setInitialLoadComplete] = useState(false);
  const { setSearchQuery, addToSearchHistory, searchHistory } = useSearch();
  const navigate = useNavigate();
  const location = useLocation();
  const searchRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Effect 1: Sync the search bar's text with the URL when the page location changes.
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const urlQuery = params.get('q') || '';
    setQuery(urlQuery);
    setInitialLoadComplete(true); // Mark that the initial URL has been processed
  }, [location.search]);

  // Effect 2: Handle clicks outside the search bar to close suggestions.
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Effect 3: Fetch suggestions when the user types.
  useEffect(() => {
    // Don't fetch suggestions until the initial URL query has been set.
    if (!initialLoadComplete) {
      return;
    }

    // Only fetch suggestions if the user is actively typing (not from URL navigation)
    // We can detect this by checking if the input is focused or if it's a user-initiated change
    const isUserTyping = document.activeElement === inputRef.current;
    
    // Condition to fetch: query must not be empty AND user must be actively typing
    if (query && isUserTyping) {
      setIsLoading(true);
      const timer = setTimeout(async () => {
        try {
          const results = await getAutosuggestions(query);
          const mapped: Suggestion[] = results.map((text, idx) => ({
            id: `suggestion-${idx}`,
            text,
            type: 'product' as const,
          }));
          setSuggestions(mapped);
          setError(null);
        } catch (error) {
          setError('Could not fetch suggestions');
          setSuggestions([]);
        }
        setIsLoading(false);
        setShowSuggestions(true);
      }, 300); // Debounce API calls

      return () => clearTimeout(timer);
    } else if (!isUserTyping) {
      // Clear suggestions if not user typing (e.g., from URL navigation)
      setSuggestions([]);
      setShowSuggestions(false);
      setIsLoading(false);
    }
  }, [query, initialLoadComplete]);

  const handleSearch = (searchQuery: string) => {
    const trimmedTerm = searchQuery.trim();
    if (!trimmedTerm) return;

    // Log the direct search event for analytics.
    try {
      console.log('[Analytics] Search:', trimmedTerm);
    } catch (err) {
      console.error('Failed to log search event:', err);
    }

    // Update local history and navigate.
    setSearchQuery(trimmedTerm);
    addToSearchHistory(trimmedTerm);
    setShowSuggestions(false);
    setQuery(trimmedTerm);
    navigate(`/search?q=${encodeURIComponent(trimmedTerm)}`);
  };

  const handleSuggestionClick = async (suggestionText: string) => {
    // First, log the click event for analytics. Await ensures this completes before navigation.
    try {
      console.log('[Analytics] Suggestion click:', suggestionText, 'from:', query);
    } catch (err) {
      // Log error without disturbing the user. The navigation will still proceed.
      console.error('Failed to log suggestion click:', err);
    }

    // Then, navigate the user to the search results page.
    handleSearch(suggestionText);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      handleSearch(query.trim());
    }
  };

  const clearQuery = () => {
    setQuery('');
    setSuggestions([]);
    setShowSuggestions(false);
    setError(null);
    inputRef.current?.focus();
  };

  const handleFocus = () => {
    setShowSuggestions(true);
    // If the user focuses on the bar and there's already text but no suggestions, fetch them.
    if (query && suggestions.length === 0) {
      setIsLoading(true);
      getAutosuggestions(query)
        .then((response: string[]) => {
          const mapped: Suggestion[] = response.map((text, idx) => ({
            id: `suggestion-${idx}`,
            text,
            type: 'product' as const,
          }));
          setSuggestions(mapped);
        })
        .catch(() => setError('Could not fetch suggestions'))
        .finally(() => setIsLoading(false));
    }
  };

  const groupSuggestions = (suggestions: Suggestion[]) => {
    const grouped = {
      products: suggestions.filter(s => s.type === 'product'),
      categories: suggestions.filter(s => s.type === 'category'),
      brands: suggestions.filter(s => s.type === 'brand'),
    };
    return grouped;
  };

  const grouped = groupSuggestions(suggestions);
  const hasResults = suggestions.length > 0;
  const showHistory = showSuggestions && query.length === 0 && searchHistory.length > 0;

  return (
    <div ref={searchRef} className="relative w-full">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={handleFocus}
            placeholder="Search for products, brands and more"
            className="w-full px-8 py-5 pr-28 bg-white border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-[#2874F0] focus:border-transparent text-lg"
          />
          <div className="absolute right-4 flex items-center space-x-3">
            {query && (
              <button
                type="button"
                onClick={clearQuery}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            )}
            <button
              type="submit"
              className="bg-[#2874F0] text-white p-4 rounded-md hover:bg-[#1e5cb8] transition-colors"
            >
              {isLoading ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                <Search className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Suggestions Dropdown */}
      {showSuggestions && (
        <div className="absolute top-full left-0 right-0 bg-white border border-gray-200 rounded-lg shadow-lg mt-1 max-h-96 overflow-y-auto z-50">
          {error && <div className="p-4 text-center text-red-500">{error}</div>}
          {isLoading ? (
            <div className="p-4 text-center">
              <Loader2 className="h-5 w-5 animate-spin mx-auto text-[#2874F0]" />
              <p className="text-sm text-gray-500 mt-2">Searching...</p>
            </div>
          ) : showHistory ? (
            <div className="py-2">
              <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase border-b">
                Recent Searches
              </div>
              {searchHistory.slice(0, 5).map((historyItem, index) => (
                <button
                  key={index}
                  onClick={() => handleSearch(historyItem)}
                  className="w-full text-left px-4 py-2 hover:bg-gray-50 transition-colors flex items-center justify-between"
                >
                  <span className="text-gray-600">{historyItem}</span>
                  <Search className="h-4 w-4 text-gray-400" />
                </button>
              ))}
            </div>
          ) : hasResults ? (
            <div className="py-2">
              {/* Products */}
              {grouped.products.length > 0 && (
                <div>
                  <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase border-b">
                    Products
                  </div>
                  {grouped.products.map((suggestion) => (
                    <button
                      key={suggestion.id}
                      onClick={() => handleSuggestionClick(suggestion.text)}
                      className="w-full text-left px-4 py-2 hover:bg-gray-50 transition-colors flex items-center justify-between"
                    >
                      <div className="flex items-center">
                        <span className={suggestion.isCorrection ? 'text-[#2874F0] font-medium' : 'text-gray-800'}>
                          {suggestion.text}
                          {suggestion.isCorrection && (
                            <span className="text-xs text-gray-500 ml-2">(corrected)</span>
                          )}
                        </span>
                        {suggestion.id.startsWith('fallback') && (
                          <span className="ml-2 px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full">
                            AI
                          </span>
                        )}
                      </div>
                      <Search className="h-4 w-4 text-gray-400" />
                    </button>
                  ))}
                </div>
              )}

              {/* Categories */}
              {grouped.categories.length > 0 && (
                <div>
                  <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase border-b">
                    Categories
                  </div>
                  {grouped.categories.map((suggestion) => (
                    <button
                      key={suggestion.id}
                      onClick={() => handleSuggestionClick(suggestion.text)}
                      className="w-full text-left px-4 py-2 hover:bg-gray-50 transition-colors flex items-center justify-between"
                    >
                      <span>{suggestion.text}</span>
                      <Search className="h-4 w-4 text-gray-400" />
                    </button>
                  ))}
                </div>
              )}

              {/* Brands */}
              {grouped.brands.length > 0 && (
                <div>
                  <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase border-b">
                    Brands
                  </div>
                  {grouped.brands.map((suggestion) => (
                    <button
                      key={suggestion.id}
                      onClick={() => handleSuggestionClick(suggestion.text)}
                      className="w-full text-left px-4 py-2 hover:bg-gray-50 transition-colors flex items-center justify-between"
                    >
                      <span>{suggestion.text}</span>
                      <Search className="h-4 w-4 text-gray-400" />
                    </button>
                  ))}
                </div>
              )}

              {/* Search History */}
              {searchHistory.length > 0 && (
                <div>
                  <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase border-b">
                    Recent Searches
                  </div>
                  {searchHistory.slice(0, 3).map((historyItem, index) => (
                    <button
                      key={index}
                      onClick={() => handleSearch(historyItem)}
                      className="w-full text-left px-4 py-2 hover:bg-gray-50 transition-colors flex items-center justify-between"
                    >
                      <span className="text-gray-600">{historyItem}</span>
                      <Search className="h-4 w-4 text-gray-400" />
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="p-4">
              <p className="text-sm text-gray-500 mb-3">Popular right now:</p>
              <div className="space-y-1">
                {['iPhone 15', 'Samsung Galaxy', 'Diwali Sale', 'Best Offers'].map((item) => (
                  <button
                    key={item}
                    onClick={() => handleSearch(item)}
                    className="block w-full text-left px-2 py-1 text-sm text-[#2874F0] hover:bg-gray-50 rounded"
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchBar;