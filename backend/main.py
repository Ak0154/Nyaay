from fastapi import FastAPI

from routes.ask import router as ask_router
from routes.upload import router as upload_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)



app.include_router(
    ask_router
)

app.include_router(
    upload_router
)