from dataclasses import dataclass
from typing import Optional, List

@dataclass(frozen=True)
class Recipe:
  id: str
  name: str
  description: str
  app_id: Optional[str] = None
  steps: Optional[List[dict]] = None
  def base_text(self) -> str:
    return f"{self.name}. {self.description}".strip()