import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from routers import active_cities,weather_summery
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

app.include_router(active_cities.router)
app.include_router(weather_summery.router)
# Allow requests from all origins with specific methods and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/")
async def root():
    return {"message": "Welcome Server side"}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)

