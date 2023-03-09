use image::{DynamicImage, GenericImage, GenericImageView, Rgba};
use rand::distributions::{Alphanumeric};
use rand::{thread_rng, Rng};
use rusttype::{Font, Scale};

pub fn corrupt_image(img: &DynamicImage) -> (DynamicImage, DynamicImage) {
    // rand between 0 and 5
    let rng = rand::thread_rng().gen_range(0..2);
    match rng {
        _ => corrupt_text(img),
        // _ => corrupt_drawings(img),
    }
}

pub fn corrupt_text(img: &DynamicImage) -> (DynamicImage, DynamicImage) {
    let (width, height) = img.dimensions();
    let mut imgbuf = img.to_rgba8();

    let text = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(50)
        .map(char::from)
        .collect::<String>();

    let font = Font::try_from_bytes(include_bytes!("../assets/Roboto-Regular.ttf")).unwrap();
    let scale = Scale::uniform(128.0);
    let glyphs: Vec<_> = font
        .layout(text.as_str(), scale, rusttype::point((width / 2) as f32, (height / 2) as f32))
        .collect();

    let mut mask = image::DynamicImage::new_rgba8(width, height);

    for glyph in glyphs {
        if let Some(bb) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {

                let x = x + bb.min.x as u32;
                let y = y + bb.min.y as u32;
                let color = Rgba([255, 255, 255, (v * 255.0) as u8]);
                if x < width && y < height {
                    imgbuf.put_pixel(x, y, color);
                    mask.put_pixel(x, y, color);
                }

            });
        }
    }

    (
        DynamicImage::ImageRgba8(imgbuf),
        mask
    )
}
