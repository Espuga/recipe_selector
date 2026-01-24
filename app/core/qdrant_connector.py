import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

class QdrantConnector():
  def __init__(
      self,
    ):
      QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
      QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
      self.COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
      self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
