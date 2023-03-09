use image::{DynamicImage, GenericImage, GenericImageView, Rgba};
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use rusttype::{Font, Scale};

pub fn corrupt_image(img: &DynamicImage) -> (DynamicImage, DynamicImage) {
    // rand between 0 and 5
    let rng = rand::thread_rng().gen_range(0..2);
    match rng {
        _ => glyphs_corrupt(img),
        // _ => corrupt_drawings(img),
    }
}

pub fn glyphs_corrupt(img: &DynamicImage) -> (DynamicImage, DynamicImage) {
    let (width, height) = img.dimensions();
    let mut imgbuf = img.clone();

    let text = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(50)
        .map(char::from)
        .collect::<String>();

    let font = Font::try_from_bytes(include_bytes!("../assets/Roboto-Regular.ttf")).unwrap();
    let scale = Scale::uniform((width / 10) as f32);
    let glyphs: Vec<_> = font
        .layout(
            text.as_str(),
            scale,
            // begin at the top left corner of the image, with a 10 pixel margin
            rusttype::point(10.0, 10.0 + scale.y),
        )
        .collect();

    // mask is a full black image
    let mut mask = image::DynamicImage::new_rgb8(width, height);

    for glyph in glyphs {
        if let Some(bb) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {
                let x = x + bb.min.x as u32;
                let y = y + bb.min.y as u32;
                if x < width && y < height && v > 0.1 {
                    let color = Rgba([255, 255, 255, (v * 255.0) as u8]);
                    imgbuf.put_pixel(x, y, color);
                    let mask_color = Rgba([255, 255, 255, 255]);
                    mask.put_pixel(x, y, mask_color);
                }
            });
        }
    }

    (imgbuf, mask)
}
