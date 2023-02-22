mod def;
mod inpainting;
mod neighborhood;
use image::DynamicImage;
use inpainting::{apply_bertalmio2001, apply_telea2004};
// use std::env;

fn main() {
    // env::set_var("RUST_BACKTRACE", "1");
    // env::set_var("RUST_BACKTRACE", "full");
    let radius = 5;

    let names = ["becelli", "text-horse"];

    for name in names.iter() {
        // try open jpg, if not, try bmp
        let img: DynamicImage = image::open(format!("samples/{name}.jpg")).unwrap_or_else(|_| {
            image::open(format!("samples/{name}.bmp"))
                .unwrap_or_else(|_| image::open(format!("samples/{name}.png")).unwrap())
        });

        let mask: DynamicImage =
            image::open(format!("samples/{name}_mask.jpg")).unwrap_or_else(|_| {
                image::open(format!("samples/{name}_mask.bmp"))
                    .unwrap_or_else(|_| image::open(format!("samples/{name}_mask.png")).unwrap())
            });

        let start = std::time::Instant::now();
        let result = apply_telea2004(&img, &mask, radius).unwrap();
        let elapsed = start.elapsed().as_millis();
        println!("Telea done on {name} in {}ms", elapsed, name = name);

        let start = std::time::Instant::now();
        let result2 = apply_bertalmio2001(&img, &mask, radius).unwrap();
        let elapsed = start.elapsed().as_millis();
        println!("Bertalmio done on {name} in {}ms", elapsed, name = name);

        result
            .save(format!("samples/result/{name}_telea.bmp"))
            .unwrap();
        result2
            .save(format!("samples/result/{name}_bertalmio.bmp"))
            .unwrap();
    }
}
