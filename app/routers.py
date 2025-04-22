from endpoints.text import router as text_route
from fastapi import FastAPI

def config_routers(app: FastAPI):
    app.include_router(
        prefix="analise/",
        tags=["Analise de dados com Pandas"],
        router=text_route
    )


