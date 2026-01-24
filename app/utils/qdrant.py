from qdrant_client.http import models as qm

def build_filter_app_or_generic(app_id: str) -> qm.Filter:
  # (app_id == X) OR (app_id is empty/absent)
  return qm.Filter(
    should=[
      qm.FieldCondition(
        key="app_id",
        match=qm.MatchValue(value=app_id)
      ),
      qm.IsEmptyCondition(is_empty=qm.PayloadField(key="app_id"))
    ]
  )
