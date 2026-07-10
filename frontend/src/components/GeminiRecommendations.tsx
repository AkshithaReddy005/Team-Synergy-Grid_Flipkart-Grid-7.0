import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Recommendation {
  suggestion: string;
  reason: string;
  type: 'alternative' | 'complementary' | 'upgrade' | 'budget_option';
}

interface QueryInsights {
  query_analysis: {
    original: string;
    corrected: string;
    intent: any;
    related_keywords: string[];
  };
  search_statistics: {
    total_results: number;
    categories: any[];
    brands: any[];
    price_range: any;
  };
  suggestions: {
    spelling_correction: string | null;
    related_searches: string[];
    intent_based_suggestions: string[];
  };
}

interface GeminiRecommendationsProps {
  searchQuery: string;
  onRecommendationClick: (query: string) => void;
}

const GeminiRecommendations: React.FC<GeminiRecommendationsProps> = ({ 
  searchQuery, 
  onRecommendationClick 
}) => {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [insights, setInsights] = useState<QueryInsights | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (searchQuery && searchQuery.trim().length > 2) {
      fetchRecommendations();
      fetchInsights();
    }
  }, [searchQuery]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`http://localhost:5001/api/search/recommendations`, {
        params: { q: searchQuery }
      });
      setRecommendations(response.data.recommendations || []);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError('Failed to load AI recommendations');
    } finally {
      setLoading(false);
    }
  };

  const fetchInsights = async () => {
    try {
      const response = await axios.get(`http://localhost:5001/api/search/insights`, {
        params: { q: searchQuery }
      });
      setInsights(response.data);
    } catch (err) {
      console.error('Error fetching insights:', err);
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'alternative': return 'bg-blue-100 text-blue-800';
      case 'complementary': return 'bg-green-100 text-green-800';
      case 'upgrade': return 'bg-purple-100 text-purple-800';
      case 'budget_option': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'alternative': return '🔄';
      case 'complementary': return '➕';
      case 'upgrade': return '⬆️';
      case 'budget_option': return '💰';
      default: return '💡';
    }
  };

  if (!searchQuery || searchQuery.trim().length <= 2) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <div className="flex items-center mb-4">
        <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mr-3">
          <span className="text-white text-sm font-bold">AI</span>
        </div>
        <h3 className="text-lg font-semibold text-gray-800">
          AI-Powered Recommendations
        </h3>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
          <span className="ml-2 text-gray-600">Analyzing your search...</span>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {insights && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-800 mb-2">Search Insights</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Results:</span>
              <span className="ml-2 font-medium">{insights.search_statistics.total_results}</span>
            </div>
            {insights.query_analysis.corrected !== insights.query_analysis.original && (
              <div>
                <span className="text-gray-600">Did you mean:</span>
                <button 
                  onClick={() => onRecommendationClick(insights.query_analysis.corrected)}
                  className="ml-2 text-blue-600 hover:underline font-medium"
                >
                  {insights.query_analysis.corrected}
                </button>
              </div>
            )}
            <div>
              <span className="text-gray-600">Category:</span>
              <span className="ml-2 font-medium">
                {insights.query_analysis.intent?.category || 'General'}
              </span>
            </div>
          </div>
        </div>
      )}

      {recommendations.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-800 mb-3">Smart Suggestions</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {recommendations.map((rec, index) => (
              <div 
                key={index}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => onRecommendationClick(rec.suggestion)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center">
                    <span className="text-lg mr-2">{getTypeIcon(rec.type)}</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(rec.type)}`}>
                      {rec.type.replace('_', ' ')}
                    </span>
                  </div>
                </div>
                <h5 className="font-medium text-gray-800 mb-1 hover:text-purple-600">
                  {rec.suggestion}
                </h5>
                <p className="text-sm text-gray-600">{rec.reason}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {insights?.suggestions.related_searches.length > 0 && (
        <div className="mt-6">
          <h4 className="font-medium text-gray-800 mb-3">Related Searches</h4>
          <div className="flex flex-wrap gap-2">
            {insights.suggestions.related_searches.slice(0, 6).map((search, index) => (
              <button
                key={index}
                onClick={() => onRecommendationClick(search)}
                className="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full text-sm transition-colors"
              >
                {search}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GeminiRecommendations; 