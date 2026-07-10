import React, { useState } from 'react';
import { searchProducts } from '../api/realApi';

const TestSearch: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      console.log('Testing search for:', query);
      const products = await searchProducts(query);
      console.log('Search results:', products);
      setResults(products);
    } catch (err) {
      console.error('Search error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Search Test</h2>
      
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter search query..."
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg"
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
          Error: {error}
        </div>
      )}

      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">Results ({results.length})</h3>
        {results.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {results.map((product, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <h4 className="font-semibold">{product.name}</h4>
                <p className="text-gray-600">₹{product.price}</p>
                <p className="text-sm text-gray-500">{product.brand}</p>
                <p className="text-sm text-gray-500">Rating: {product.rating}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No results found</p>
        )}
      </div>

      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-2">Quick Test Queries</h3>
        <div className="flex flex-wrap gap-2">
          {['iphone', 'backpack under 5000', 'laptop', 'headphones'].map((testQuery) => (
            <button
              key={testQuery}
              onClick={() => {
                setQuery(testQuery);
                setTimeout(() => handleSearch(), 100);
              }}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
            >
              {testQuery}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TestSearch; 