use actix_web::{Responder, HttpResponse, get, web};
use geojson::{GeoJson, Value};
use reqwest::{Client, Response};


#[get("/")]
pub async fn info() -> impl Responder {
    HttpResponse::Ok().body("Hello world!")
}

#[get("/map/{id}")]
async fn get_geojson(web::Path(id): web::Path<u32>) -> impl Responder {
    // get a geojson from a url
    let client = Client::new();
    let url = format!("https://127.0.0.1/cluster/{}", id);
    let response: Response = client.get(&url).send().await.unwrap();
    let geojson: GeoJson = response.json().await.unwrap();

    // return the geojson
    HttpResponse::Ok().json(geojson)
    
}