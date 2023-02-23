mod def;
mod inpainting;
mod neighborhood;
use image::DynamicImage;
use inpainting::{bertalmio2001, telea2004};

fn main() {
    let radius = 5;

    let names = ["becelli", "bricks", "text-horse"];

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
        let result_telea = telea2004(&img, &mask, radius).unwrap();
        let elapsed = start.elapsed().as_millis();
        println!("Telea done on {name} in {}ms", elapsed, name = name);

        let start = std::time::Instant::now();
        let result_bertalmio = bertalmio2001(&img, &mask, radius).unwrap();
        let elapsed = start.elapsed().as_millis();
        println!("Bertalmio done on {name} in {}ms", elapsed, name = name);

        result_telea
            .save(format!("samples/result/{name}_telea.bmp"))
            .unwrap();
        result_bertalmio
            .save(format!("samples/result/{name}_bertalmio.bmp"))
            .unwrap();
    }
}
