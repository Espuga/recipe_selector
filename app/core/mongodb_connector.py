import os
from pymongo import MongoClient

class MongoDBConnector():
  def __init__(self):
    mongo_db_name = os.getenv("MONGO_DB_NAME", "recipes")
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    self.client = MongoClient(mongo_uri)
    self.db = self.client[mongo_db_name]
