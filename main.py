from fastapi import FastAPI
import compute as c

app = FastAPI()


@app.get("/")
async def root(d: str = ''):
    data = c.compute(d)
    return {"message": data}