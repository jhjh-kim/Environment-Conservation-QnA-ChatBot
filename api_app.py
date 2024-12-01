import chatbot
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

print(f"System: Env Variables are loaded: {load_dotenv()}")

app = FastAPI()
docs_dir_path = "./document_set"
db_dir = './database'
chatbot = chatbot.QnAChatBot(docs_dir_path, db_dir)

class MissionRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    quiz: str
    question: str

@app.post("/mission")
async def ask_question(request: MissionRequest):
    # The query method returns a Python dictionary
    response = chatbot.ask(request.question)
    # FastAPI automatically converts this dictionary to JSON
    return response

@app.post("/quiz")
async def ask_question(request: QuizRequest):
    # The query method returns a Python dictionary
    response = chatbot.ask(request.question)
    # FastAPI automatically converts this dictionary to JSON
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
