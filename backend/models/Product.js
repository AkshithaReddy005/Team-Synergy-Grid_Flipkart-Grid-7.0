const mongoose = require('mongoose');

const productSchema = new mongoose.Schema({
    productId: { type: String, required: true, unique: true },
    name: { type: String, required: true, index: true },
    description: { type: String, required: true },
    category: { type: String, required: true, index: true },
    brand: { type: String, required: true, index: true },
    price: { type: Number, required: true },
    rating: { type: Number, default: 0 }, // Renamed from 'ratings'
    stock: { type: Number, default: 100 }, // Added 'stock' field
    thumbnail: { type: String }, // Renamed from 'imageUrl'
    offers: { type: String },
    numReviews: { type: Number, default: 0 },
});

const Product = mongoose.model('Product', productSchema);

module.exports = Product;
