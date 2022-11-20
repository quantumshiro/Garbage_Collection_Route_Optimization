#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
#[warn(unused_imports)]
extern crate rocket;

mod routes;

fn main() {
    rocket::ignite()
        .mount("/", routes![routes::index, routes::map])
        .launch();
}