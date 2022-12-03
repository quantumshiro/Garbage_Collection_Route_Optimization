// map_id: u32
// geojson
//

use geojson::GeoJson;
use hyper::{Client, Response};

type MapId = u32;

pub async fn get_geojson(map_id: u32) -> GeoJson {

}