import os
import json
from typing import List

class FeedbackStore:
  def __init__(self, path: str):
    self.path = path

  def append(self, record: dict) -> None:
    with open(self.path, "a", encoding="utf-8") as f:
      f.write(json.dumps(record, ensure_ascii=False) + "\n")

  def read_all(self) -> List[dict]:
    if not os.path.exists(self.path):
      return []

    records = []
    with open(self.path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          records.append(json.loads(line))
        except json.JSONDecodeError:
          continue

    return records
