from dataclasses import dataclass
from typing import Optional, List, Any

@dataclass(frozen=True)
class Recipe:
  id: str
  name: str
  description: str
  app_id: Optional[str] = None
  steps: Optional[List[dict]] = None

  def base_text(self) -> str:
    return f"{self.name}. {self.description}".strip()

  @classmethod
  def from_mongo(cls, doc: dict) -> "Recipe":
    return cls(
      id=str(doc["_id"]),
      name=doc["name"],
      description=doc.get("description", ""),
      app_id=str(doc["app_id"]) if doc.get("app_id") is not None else None,
      steps=doc.get("steps")
    )