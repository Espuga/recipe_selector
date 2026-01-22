from dataclasses import dataclass

@dataclass(frozen=True)
class Recipe:
  id: str
  name: str
  description: str
  def base_text(self) -> str:
    return f"{self.name}. {self.description}".strip()