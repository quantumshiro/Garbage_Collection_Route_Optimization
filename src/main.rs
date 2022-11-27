#![feature(proc_macro_hygiene, decl_macro)]

extern crate rocket;
use calamine::{open_workbook, Reader, Xlsx}; 
// mod routes;

pub fn get_data(bookname: String, sheetname: String) -> Vec<Vec<String>> {
    let mut workbook: Xlsx<_> = open_workbook(bookname).unwrap();
    let mut data: Vec<Vec<String>> = Vec::new();
    if let Some(Ok(range)) = workbook.worksheet_range(&sheetname) {
        for row in range.rows() {
            let mut row_data: Vec<String> = Vec::new();
            for cell in row {
                let cell_data = cell.to_string();
                row_data.push(cell_data);
            }
            data.push(row_data);
        }
    }
    data
}


fn main() {
    /* rocket::ignite()
        .mount("/", routes![routes::index, routes::map])
        .launch();
    */
    let hoge = get_data("data/garbage_place.xlsx".to_string(), "ごみ置場".to_string());
    println!("{:?}", hoge);
}
