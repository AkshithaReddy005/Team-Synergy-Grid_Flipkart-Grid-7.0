const UserHistory = require('../models/UserHistory');

class UserHistoryService {
  static async recordEvent({ user_id, session_id, product_id, action, metadata = {} }) {
    try {
      await UserHistory.create({ user_id, session_id, product_id, action, metadata });
    } catch (error) {
      console.error('[UserHistoryService] Failed to record event:', error);
    }
  }

  static async fetchHistory(user_id, limit = 20) {
    try {
      const history = await UserHistory.find({ user_id })
        .sort({ timestamp: -1 })
        .limit(limit)
        .lean();
      return history;
    } catch (error) {
      console.error('[UserHistoryService] Failed to fetch history:', error);
      return [];
    }
  }

  static async enrichWithProductDetails(history) {
    const Product = require('../models/Product');
    const productIds = [...new Set(history.map(h => h.product_id))];
    const products = await Product.find({ productId: { $in: productIds } })
      .select('productId name brand category price')
      .lean();
    const productMap = Object.fromEntries(products.map(p => [p.productId, p]));
    return history.map(h => ({
      ...h,
      product: productMap[h.product_id] || null,
    }));
  }
}

module.exports = UserHistoryService;
