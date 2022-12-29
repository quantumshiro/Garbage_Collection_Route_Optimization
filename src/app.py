from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import RedirectResponse

from src.lib.map_cluster import *

app = FastAPI()

@app.get("/")
def root():
    return HTMLResponse(open("src/front/templates/index.html", "r").read())

@app.get("/map/{map_id}")
async def read_root(map_id: str):
    # return index.html
    return HTMLResponse(open("src/front/templates/"+map_id+".html", "r").read())

@app.get("/geography/cluster/{map_id}")
def get_cluster(map_id: int):
    df = get_data("./data/garbage_place.xlsx")
    cluster = cluster_map(df, "X", "Y", 44)
    df["cluster_id"] = cluster
    json = make_geojson(df, map_id)
    return JSONResponse(content=json)