import uvicorn as uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from recommandation import get_high_recommended_activities

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/user_recommendation/{user_id}")
def get_user_recommendation(user_id: int):
    activities = get_high_recommended_activities(user_id)
    return activities


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4444, log_level="info")
