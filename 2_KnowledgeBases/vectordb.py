import os
from pymongo import UpdateOne
from pymongo.operations import SearchIndexModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

USER = os.environ.get("MDB_USER", None)
PASS = os.environ.get("MDB_PASS", None)

class MongoDB():
    def __init__(self, database_name, collection_name) -> None:
        # connect to the mongo db cluster
        self.uri = f"mongodb+srv://{USER}:{PASS}@chatpdf.uzbwl.mongodb.net/?retryWrites=true&w=majority&appName=ChatPDF"
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        self.database = self.client.get_database(database_name)
        self.collection = self.database[collection_name]
        # Send a ping to confirm a successful connection
        try:
            # Create a new client and connect to the server
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    def load_chunks(self, chunks):
        # using bulk operations to add chunks all together
        bulk_operations = []
        for chunk in chunks:
            bulk_operations.append(
                # update adds the item if its not already there, or updates it if it is there in the database
                UpdateOne(
                    {"_id": chunk["_id"]},  # chunk id must be unique. uses this to check if the entry is already present in the database
                    {"$set": chunk},        # updates the chunk information
                    upsert = True           # paramter to allow mongo db to add an item if it isn't there in the database already
                )
            )
        result = self.collection.bulk_write(bulk_operations)
        return result

    def create_index(self, index_name, dimensions, similarity="cosine", embedding_field='embeddings'):
        indexes = []
        # list all the current search indexes created for the database/ collection
        for index in self.collection.list_search_indexes():
            indexes.append(index)

        # if there are no indexes associated with the collection, create one
        if indexes == []:
            search_index_model = SearchIndexModel(
                definition={
                    "fields":[{
                        "type":"vector",
                        "path":embedding_field,
                        "numDimensions": dimensions,
                        "similarity":similarity
                    }]
                },
                name=index_name,
                type="vectorSearch"
            )
            result = self.collection.create_search_index(model=search_index_model)
            print(result)
        # else, index has already been created so we just need to update the index
        else:
            definition = {
                "fields":[{
                    "type":"vector",
                    "path":embedding_field,
                    "numDimensions": dimensions,
                    "similarity":similarity
                }]
            }
            self.collection.update_search_index(index_name, definition)

    # do a similarity search between the query embedding and the embeddings in the database and return the 3 most relevant items/ chunks
    def retrieve(self, index_name, query_embedding, embedding_field='embedding', num_neighbors=100, limit=5):
        pipeline = [
            {'$vectorSearch':{
                'index':index_name,
                'path':embedding_field,
                'queryVector':query_embedding,
                'numCandidates':num_neighbors,
                'limit':limit
            }},
            {'$project':{
                '_id':0,
                'text':1,
                'metadata':1,
                'score':{
                    '$meta': 'vectorSearchScore'
                }
            }}
        ]
        result = self.collection.aggregate(pipeline)
        return result
