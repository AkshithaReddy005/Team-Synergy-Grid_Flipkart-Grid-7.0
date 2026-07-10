const UserQuery = require('../models/UserQuery');
const elasticClient = require('../elasticClient');

const QUERIES_INDEX = 'search_queries';

/**
 * Updates the popularity of a single query in MongoDB and syncs it to Elasticsearch in real-time.
 * @param {string} queryText The search query to update.
 */
const updateAndSyncQuery = async (queryText) => {
    if (!queryText) return;

    try {
        // 1. Update the query's frequency in MongoDB, or create it if it doesn't exist.
        const updatedQuery = await UserQuery.findOneAndUpdate(
            { queryText: queryText },
            {
                $inc: { frequency: 1 },
                $set: { lastClicked: new Date() }
            },
            { new: true, upsert: true } // `new: true` returns the updated doc, `upsert: true` creates it if not found
        ).lean();

        console.log(`[Popularity Service] Updated "${queryText}", new frequency: ${updatedQuery.frequency}`);

        // 2. Sync the updated query to Elasticsearch.
        await elasticClient.index({
            index: QUERIES_INDEX,
            id: updatedQuery._id.toString(), // Use the MongoDB document ID for consistency
            body: {
                queryText: updatedQuery.queryText,
                suggest: {
                    input: [updatedQuery.queryText], // Suggester input needs to be an array
                    weight: updatedQuery.frequency // The popularity score
                }
            }
        });

        console.log(`[Popularity Service] Synced "${queryText}" to Elasticsearch.`);

    } catch (error) {
        console.error(`[Popularity Service] Error processing query "${queryText}":`, error);
    }
};

module.exports = { updateAndSyncQuery };
