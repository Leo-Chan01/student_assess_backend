from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
from sentence_transformers import SentenceTransformer, util # type: ignore

app = FastAPI()

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class CompareRequest(BaseModel):
    pdfText: str
    userInput: str

@app.post("/compare")
async def compare_texts(request: CompareRequest):

    pdf_text = request.pdfText
    user_input = request.userInput

    pdf_embedding = model.encode(pdf_text, convert_to_tensor=True)
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(pdf_embedding, user_input_embedding).item()

    return {"similarityScore": similarity * 100}