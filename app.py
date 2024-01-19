from fastapi import FastAPI, File, UploadFile, HTTPException
import re
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import datetime
import ffmpeg
import speech_recognition as sr

# Initialize the NLTK stemmer
stemmer = nltk.PorterStemmer()


class ProcessRequest(BaseModel):
    texts: List[str]


class StemmingRequest(BaseModel):
    text: str


app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_status():
    return {"status": "running"}


# @app.post("/stemmer", response_model=str)
# async def preprocess_queries(request: StemmingRequest):
#     text = request.text.lower().strip()
#     text = stemmer.stem(text)
#     return text
def preprocess_queries(texts):
    text = texts.lower().strip()
    text = stemmer.stem(text)
    return text


@app.post("/process", response_model=List[str])
async def pos_queries(request: ProcessRequest):
    rawdatas = request.texts
    datas = []
    for rawdata in rawdatas:
        datas.append(preprocess_queries(rawdata))
    representative_questions = []
    df = pd.DataFrame({'question_text': [], 'cluster': []})
    # print(request.texts)
    df['question_text'] = datas
    tfidfVectorizer = TfidfVectorizer()
    query_embeddings = tfidfVectorizer.fit_transform(df['question_text'])
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(query_embeddings)
    df['cluster'] = kmeans.labels_
    for cluster_id in range(num_clusters):
        cluster_data = df.loc[df['cluster'] == cluster_id]
        centroid = kmeans.cluster_centers_[cluster_id]
        similarities = cosine_similarity(tfidfVectorizer.transform(
            cluster_data['question_text']), [centroid])
        representative_question = cluster_data.iloc[similarities.argmax(
        )]['question_text']
        representative_questions.append(representative_question)
    return representative_questions


def convertToWAV(file):
    filename = f"{datetime.datetime.now().timestamp()}"
    inloc = f"uploads/{filename}.{re.split('[/;]',file.content_type)[1]}"
    outloc = f"uploads/{filename}.wav"
    with open(inloc, "wb") as f:
        f.write(file.file.read())
    ffmpeg.input(inloc).output(outloc).run()
    return outloc


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not re.match("audio/", file.content_type):
        raise HTTPException(status_code=400, detail="File type not supported")

    if not re.match("audio/wav", file.content_type):
        file_path = convertToWAV(file)
    else:
        file_path = file.filename

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            data = recognizer.record(source)
        transcript = recognizer.recognize_google(data, key=None)
        return {"transcript": transcript}
    except Exception as e:
        print("error:", e)
        raise HTTPException(status_code=500, detail="Unable to transcribe")
