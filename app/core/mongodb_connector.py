from pymongo import MongoClient

class MongoDBConnector():
  def __init__(
      self,
      db_name: str = "recipes",
      mongo_uri: str = "mongodb://localhost:27017"
    ):
    self.client = MongoClient(mongo_uri)
    self.db = self.client[db_name]
