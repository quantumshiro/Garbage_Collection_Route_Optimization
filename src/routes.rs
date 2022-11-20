use rocket_contrib::json::Json;
use geojson::{Feature, GeoJson, Geometry, Value};

#[get("/")]
pub fn index() -> &'static str {
    "Hello, world!"
}

// return geojson
#[get("/map")]
pub fn map() -> Json<GeoJson> {
    let geometry = Geometry::new(Value::Point(vec![-122.6764, 45.5165]));
    let feature = Feature {
        bbox: None,
        geometry: Some(geometry),
        id: None,
        properties: None,
        foreign_members: None,
    };
    let geojson = GeoJson::Feature(feature);
    Json(geojson)
}
