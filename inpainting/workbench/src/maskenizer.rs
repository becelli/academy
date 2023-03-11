use image::{DynamicImage, GenericImage, GenericImageView, Rgba};
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use rusttype::{Font, Scale};

pub fn corrupt_image(img: &DynamicImage) -> (DynamicImage, DynamicImage) {
    // let rng = rand::thread_rng().gen_range(0..2);
    // match rng {
    //     _ => glyphs_corrupt(img),
    // }
    glyphs_corrupt(img)
}

pub fn glyphs_corrupt(img: &DynamicImage) -> (DynamicImage, DynamicImage) {
    let (width, height) = img.dimensions();
    let mut imgbuf = img.clone();

    let text = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(250)
        .map(char::from)
        .filter(|c| c.is_alphabetic())
        .map(|c| c.to_ascii_lowercase())
        .collect::<String>();

    // divide the text by 25 characters per line
    let lines = text
        .chars()
        .collect::<Vec<char>>()
        .chunks(25)
        .map(|c| c.iter().collect::<String>())
        .collect::<Vec<String>>();

    let font = Font::try_from_bytes(include_bytes!("../assets/fonts/Roboto-Regular.ttf")).unwrap();
    let scale = Scale::uniform((width / 12) as f32);
    let v_metrics = font.v_metrics(scale);

    let glyphs: Vec<_> = lines
        .iter()
        .enumerate()
        .flat_map(|(i, line)| {
            font.layout(
                line,
                scale,
                rusttype::point(25.0, v_metrics.ascent + (i as f32 * 40.0)),
            )
            .collect::<Vec<_>>()
        })
        .collect();

    let mut mask = DynamicImage::new_rgb8(width, height);

    for glyph in glyphs {
        if let Some(bb) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {
                let x = x + bb.min.x as u32;
                let y = y + bb.min.y as u32;
                if x < width && y < height && v > 0.1 {
                    let color = Rgba([255, 255, 255, 255]);
                    imgbuf.put_pixel(x, y, color);
                    mask.put_pixel(x, y, color);
                }
            });
        }
    }

    (imgbuf, mask)
}

#[allow(dead_code, unused_variables)]
pub fn corrupt_drawings(img: &DynamicImage) -> (DynamicImage, DynamicImage) {
    let (width, height) = img.dimensions();

    (img.clone(), img.clone())
}
