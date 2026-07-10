const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '../.env') });
const { Client } = require('@elastic/elasticsearch');
const mongoose = require('mongoose');
const connectDB = require('../config/db');
const UserQuery = require('../models/UserQuery');

const esClient = new Client({ node: process.env.ELASTICSEARCH_HOST || 'http://localhost:9200' });
const QUERIES_INDEX = 'search_queries';

const createIndexWithMapping = async () => {
        const indexExists = await esClient.indices.exists({ index: QUERIES_INDEX });
    if (indexExists) {
        console.log(`Deleting existing index '${QUERIES_INDEX}' to apply new mapping...`);
        await esClient.indices.delete({ index: QUERIES_INDEX });
    }

    console.log(`Creating index '${QUERIES_INDEX}' with mapping...`);
        console.log(`Creating index '${QUERIES_INDEX}' with mapping...`);
        await esClient.indices.create({
            index: QUERIES_INDEX,
            body: {
                mappings: {
                    properties: {
                        queryText: { type: 'text' },
                        frequency: { type: 'integer' },
                        suggest: {
                            type: 'completion',
                            analyzer: 'standard',
                            search_analyzer: 'standard',
                        }
                    }
                }
            }
        });
        console.log('Index created successfully.');

};

const getQueriesFromDB = async () => {
    console.log('Fetching queries from MongoDB...');
    const queries = await UserQuery.find({}).lean();
    console.log(`Found ${queries.length} queries in the database.`);

        return queries.map(doc => ({
        _id: doc._id, // CRITICAL FIX: Carry over the original document ID
        queryText: doc.queryText,
        frequency: doc.frequency,
                suggest: {
            input: doc.queryText.toLowerCase(), // Use the full query text as a single input string
            weight: doc.frequency || 1, // Use frequency as weight, default to 1
        }
    }));
};

const syncQueriesToEs = async () => {
    await connectDB();
    try {
        await createIndexWithMapping();
        const queriesToIndex = await getQueriesFromDB();

        if (queriesToIndex.length === 0) {
            console.log('No queries to index.');
            return;
        }

        console.log(`Preparing to bulk index ${queriesToIndex.length} queries...`);

                const body = queriesToIndex.flatMap(doc => {
            const { _id, ...docBody } = doc; // Destructure to separate _id from the rest of the document
            return [
                { index: { _index: QUERIES_INDEX, _id: _id.toString() } },
                docBody // Send only the actual document body for indexing
            ];
        });

        const bulkResponse = await esClient.bulk({ refresh: true, body });

        if (bulkResponse.errors) {
            console.error('Failed to index some documents.');
            const erroredDocuments = [];
            bulkResponse.items.forEach((action, i) => {
                const operation = Object.keys(action)[0];
                if (action[operation].error) {
                    erroredDocuments.push({
                        status: action[operation].status,
                        error: action[operation].error,
                        document: body[i * 2 + 1] // The document that failed
                    });
                }
            });
            console.log('Detailed errors:', JSON.stringify(erroredDocuments, null, 2));
        } else {
            console.log(`Successfully indexed ${bulkResponse.items.length} documents.`);
        }

    } catch (error) {
        console.error('Error syncing queries to Elasticsearch:', error);
    } finally {
        await mongoose.disconnect();
        console.log('MongoDB connection closed after sync.');
    }
};

if (require.main === module) {
    syncQueriesToEs();
}

module.exports = { syncQueriesToEs };
