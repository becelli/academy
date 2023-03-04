mod def;
mod inpaint;
mod neighborhood;
use inpaint::{bertalmio2001, telea2004};

fn main() {
    let radius = 7;

    let names = ["becelli", "bricks", "text-horse"];

    for name in names {
        
        let formats = ["jpg", "bmp", "png", "jpeg"];
        let img_name = |name: &str, fmt: &str| format!("samples/{name}.{fmt}");
        let mask_name = |name: &str, fmt: &str| format!("samples/{name}_mask.{fmt}");
        
        let (img, mask) = formats.iter().fold((None, None), |(img, mask), format| {
            let img = img.or_else(|| image::open(img_name(name, format)).ok());
            let mask = mask.or_else(|| image::open(mask_name(name, format)).ok());
            (img, mask)
        });
    
        let (img, mask) = match (img, mask) {
            (Some(img), Some(mask)) => (img, mask),
            _ => {
                println!("image or mask not found");
                continue;
            }
        };
        
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
