export interface Product {
  productId?: string;
  pid?: string;
  id?: string;
  name: string;
  brand?: string;
  category?: string;
  price?: number;
  discounted_price?: number;
  retail_price?: number;
  rating?: number;
  product_rating?: number;
  popularity?: number;
  description?: string;
  thumbnail?: string;
  image?: string;
  stock?: number;
  offers?: string[];
  numReviews?: number;
  _score?: number;
  _score_es?: number;
  _score_personal?: number;
  _score_gbert?: number;
}

export interface SearchResponse {
  products: Product[];
  total: number;
}

export interface UserHistoryEvent {
  user_id: string;
  session_id: string;
  product_id: string;
  action: 'view' | 'click' | 'purchase';
  metadata?: {
    query?: string;
    price?: number;
    category?: string;
  };
}

export interface GBertRecommendation {
  product_id: string;
  score: number;
}

export interface GBertResponse {
  user_id: string;
  recommendations: GBertRecommendation[];
}

export interface GBertRerankRequest {
  user_id: string;
  history: Array<{
    product_id?: string;
    title?: string;
    action?: string;
  }>;
  candidate_pids: string[];
}

export interface GBertRerankResponse {
  user_id: string;
  scores: GBertRecommendation[];
}

export interface PersonalizeScoreRequest {
  features: number[];
}

export interface PersonalizeScoreResponse {
  score: number;
}

export interface PersonalizeBatchRequest {
  features_list: number[][];
}

export interface PersonalizeBatchResponse {
  scores: number[];
}

export interface SpellCorrectRequest {
  text: string;
}

export interface SpellCorrectResponse {
  corrected: string;
}
