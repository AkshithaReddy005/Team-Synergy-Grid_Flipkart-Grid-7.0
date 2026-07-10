const mongoose = require('mongoose');

const UserQuerySchema = new mongoose.Schema({
    queryText: {
        type: String,
        required: true,
        unique: true, // Each query text should be unique
        index: true
    },
    frequency: {
        type: Number,
        default: 1 // Start with a base frequency of 1
    },
    lastClicked: {
        type: Date
    },
    // You could add other metrics here later, e.g., quality_score
});

module.exports = mongoose.model('UserQuery', UserQuerySchema);
