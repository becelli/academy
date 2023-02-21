mod neighborhood;
mod def;
mod inpainting;
use image::DynamicImage;
use inpainting::{apply_bertalmio2001, apply_telea2004};
// use std::env;

fn main() {
    // env::set_var("RUST_BACKTRACE", "1");
    // env::set_var("RUST_BACKTRACE", "full");
    let radius = 5;

    let names = ["text-horse"];

    for name in names.iter() {
        let img: DynamicImage = image::open(format!("samples/{name}.png")).unwrap();

        let mask: DynamicImage = image::open(format!("samples/{name}_mask.png")).unwrap();

        let result = apply_telea2004(&img, &mask, radius).unwrap();
        println!("Telea done on {name}");
        let result2 = apply_bertalmio2001(&img, &mask, radius).unwrap();
        println!("Bertalmio done on {name}");

        result.save(format!("samples/{name}_telea.png")).unwrap();
        result2
            .save(format!("samples/{name}_bertalmio.png"))
            .unwrap();
    }
}
