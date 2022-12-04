
use actix_web::{HttpServer, App};
use serde::{Serialize, Deserialize};
use geojson::GeoJson;

mod routes;
mod models;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Cluster {
    id: u32,
    name: String,
    geojson: GeoJson,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(routes::info).service(routes::get_geojson))
        .bind("127.0.0.1:8080")?
        .run()
        .await
}