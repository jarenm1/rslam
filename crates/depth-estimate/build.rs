use std::panic;
extern crate pkg_config;

fn main() {
    match vcpkg::find_package("opencv4") {
        Ok(library) => {
            println!("found opencv4 via vcpkg");
        }
        Err(e) => {
            println!("failed to find opencv4");

            match pkg_config::probe_library("opencv4") {
                Ok(library) => {
                    println!("Found opencv4 with pkg-config");
                }
                Err(e) => {
                    panic!("Failed to find opencv4!!")
                }
            }
        }
    }
}
