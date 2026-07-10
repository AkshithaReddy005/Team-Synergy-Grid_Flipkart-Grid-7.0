const mongoose = require('mongoose');

const userHistorySchema = new mongoose.Schema({
  user_id: { type: String, required: true, index: true },
  session_id: { type: String, required: true },
  product_id: { type: String, required: true },
  action: { type: String, enum: ['view', 'click', 'purchase'], required: true },
  timestamp: { type: Date, default: Date.now },
  metadata: {
    query: String,
    price: Number,
    category: String,
  },
}, { timestamps: true });

userHistorySchema.index({ user_id: 1, timestamp: -1 });

module.exports = mongoose.model('UserHistory', userHistorySchema);
