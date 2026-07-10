require('dotenv').config();
const mongoose = require('mongoose');
const axios = require('axios');
const Product = require('./models/Product');

const MONGO_URI = process.env.MONGO_URI;

const manualProducts = [
    {
        id: 1001,
        title: 'Prestige Pressure Cooker',
        description: 'A durable and safe pressure cooker for fast and healthy cooking.',
        price: 50,
        rating: 4.7,
        brand: 'Prestige',
        category: 'kitchen-appliances',
        thumbnail: 'https://rukminim1.flixcart.com/image/832/832/xif0q/pressure-cooker/y/g/i/-original-imagk7s2yvrg5zhy.jpeg',
        images: ['https://rukminim1.flixcart.com/image/832/832/xif0q/pressure-cooker/y/g/i/-original-imagk7s2yvrg5zhy.jpeg']
    },
    {
        id: 1002,
        title: 'Hawkins Contura Pressure Cooker',
        description: 'Hard anodised body with a unique body shape for easy stirring.',
        price: 65,
        rating: 4.8,
        brand: 'Hawkins',
        category: 'kitchen-appliances',
        thumbnail: 'https://m.media-amazon.com/images/I/61x055uS-yL._AC_SL1500_.jpg',
        images: ['https://m.media-amazon.com/images/I/61x055uS-yL._AC_SL1500_.jpg']
    },
    {
        id: 1003,
        title: 'Butterfly Stainless Steel Pressure Cooker',
        description: 'Made from high-quality stainless steel, this cooker is a reliable kitchen companion.',
        price: 45,
        rating: 4.5,
        brand: 'Butterfly',
        category: 'kitchen-appliances',
        thumbnail: 'https://butterflyindia.com/wp-content/uploads/2022/12/Standard-plus-5L-Cooker.jpg',
        images: ['https://butterflyindia.com/wp-content/uploads/2022/12/Standard-plus-5L-Cooker.jpg']
    }
];

const run = async () => {
    let isConnected = false;
    try {
        if (mongoose.connection.readyState !== 1) {
            await mongoose.connect(MONGO_URI, {
                useNewUrlParser: true,
                useUnifiedTopology: true,
            });
            isConnected = true;
            console.log('MongoDB connected for seeding...');
        }

        console.log('Clearing existing product data...');
        await Product.deleteMany({});

        console.log('Fetching real product data from DummyJSON API...');
        const response = await axios.get('https://dummyjson.com/products?limit=0');
        const totalProducts = response.data.total;
        console.log(`Found ${totalProducts} products. Fetching all...`);

        const allProductsResponse = await axios.get(`https://dummyjson.com/products?limit=${totalProducts}`);
        console.log(`Successfully fetched ${allProductsResponse.data.products.length} products.`);

        let products = allProductsResponse.data.products.map(p => ({
            productId: p.id,
            name: p.title,
            description: p.description,
            price: p.price,
            category: p.category,
            brand: p.brand || 'Generic',
            thumbnail: p.thumbnail,
            rating: p.rating,
            stock: p.stock,
            offers: `${p.discountPercentage}% off`
        }));

        const mappedManualProducts = manualProducts.map(p => ({
            productId: p.id,
            name: p.title,
            description: p.description,
            price: p.price,
            brand: p.brand,
            category: p.category,
            thumbnail: p.thumbnail,
            rating: p.rating,
            stock: 50,
            offers: '15% off'
        }));

        const allProducts = [...products, ...mappedManualProducts];

        await Product.insertMany(allProducts);
        console.log(`${allProducts.length} products have been successfully seeded into the database.`);

    } catch (error) {
        console.error('Error during data fetching and seeding process:', error);
        throw error; // Re-throw the error to be caught by the caller
    } finally {
        if (isConnected && mongoose.connection.readyState === 1) {
            await mongoose.connection.close();
            console.log('MongoDB connection closed after seeding.');
        }
    }
};

// If the script is run directly, execute the run function.
// This allows it to be both importable and runnable.
if (require.main === module) {
    run().catch(err => console.error(err));
}

module.exports = { run };
