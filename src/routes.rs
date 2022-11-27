use rocket_contrib::json::Json;
use geojson::{Feature, GeoJson, Geometry, Value};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::env;

#[get("/")]
pub fn index() -> &'static str {
    "Hello, world!"
}

// return geojson
#[get("/map")]
pub fn map() -> Json<GeoJson> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let syspath: &PyList = py.import("sys").unwrap().get("path").unwrap().extract().unwrap();
    let path = env::current_dir();
    syspath.insert(0, path.unwrap().join("src").join("python").to_str().unwrap()).unwrap();

    let getdata = py.import("get_data").unwrap();
    // use getattr(name)?.call1(args)
    let response = getdata.call1("get_data", ("data/garbage_place.xlsx", ));
    print!("{:?}", response);

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
