import axios from 'axios';
import type {
  SearchResponse,
  UserHistoryEvent,
  GBertResponse,
  GBertRerankRequest,
  GBertRerankResponse,
  PersonalizeScoreRequest,
  PersonalizeScoreResponse,
  PersonalizeBatchRequest,
  PersonalizeBatchResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5001/api';

export const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Search endpoints
export const searchProducts = async (params: {
  q?: string;
  page?: number;
  category?: string;
  brand?: string;
  sortBy?: string;
  price_lt?: number;
  price_gt?: number;
  rating_gte?: number;
  user_id?: string;
  personalize?: boolean;
}): Promise<SearchResponse> => {
  const { data } = await api.get('/search', { params });
  return data;
};

export const getAutosuggestions = async (q: string): Promise<string[]> => {
  const { data } = await api.get('/search/suggestions', { params: { q } });
  return data;
};

// User history endpoints
export const recordUserEvent = async (event: UserHistoryEvent): Promise<{ success: boolean }> => {
  const { data } = await api.post('/userHistory/event', event);
  return data;
};

export const getUserHistory = async (
  userId: string,
  limit = 20
): Promise<UserHistoryEvent[]> => {
  const { data } = await api.get(`/userHistory/${userId}`, { params: { limit } });
  return data;
};

// ML endpoints
export const gbertRecommend = async (req: {
  user_id: string;
  history?: Array<{ product_id?: string; title?: string; action?: string }>;
  k?: number;
}): Promise<GBertResponse> => {
  const { data } = await api.post('/ml/gbert', req);
  return data;
};

export const gbertRerank = async (req: GBertRerankRequest): Promise<GBertRerankResponse> => {
  const { data } = await api.post('/ml/gbert/rerank', req);
  return data;
};

export const personalizeScore = async (req: PersonalizeScoreRequest): Promise<PersonalizeScoreResponse> => {
  const { data } = await api.post('/ml/personalize', req);
  return data;
};

export const personalizeScoreBatch = async (req: PersonalizeBatchRequest): Promise<PersonalizeBatchResponse> => {
  const { data } = await api.post('/ml/personalize/score-batch', req);
  return data;
};

export const spellCorrect = async (text: string): Promise<string> => {
  const { data } = await api.post('/ml/spell/correct', { text });
  return data.corrected;
};
