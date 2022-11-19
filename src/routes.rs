use rocket_contrib::json::Json;

#[get("/map")]
pub fn map() -> Json<Map> {
    Json(Map {
        name: "Map".to_string(),
        width: 100,
        height: 100,
        tiles: vec![vec![Tile::new(); 100]; 100],
    })
}