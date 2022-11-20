use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Map {
    pub map_id: i32,
    pub description: String,
}