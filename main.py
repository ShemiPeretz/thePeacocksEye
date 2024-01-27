from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/get-variable")
def get_variable():
    my_variable = "Hello from peacocksEye server side!"
    return {"variable": my_variable}