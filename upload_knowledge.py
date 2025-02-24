from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.table_learner import TableLearner

app = FastAPI()
learner = TableLearner()

class KnowledgeData(BaseModel):
    knowledge: Dict[str, Any]

@app.post("/upload_knowledge")
async def upload_knowledge(data: KnowledgeData):
    """
    Upload knowledge data to be learned by the system
    """
    try:
        result = learner.learn_from_knowledge(data.knowledge)
        return {"status": "success", "message": "Knowledge successfully learned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query_knowledge")
async def query_knowledge(query: str):
    """
    Query the learned knowledge
    """
    try:
        result = learner.get_relevant_knowledge(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
