mod metrics_tests;
use image::{DynamicImage, GenericImageView, Pixel};

pub fn diff_pixels(original: &DynamicImage, inpainted: &DynamicImage) -> u32 {
    original
        .pixels()
        .zip(inpainted.pixels())
        .fold(0, |acc, (a, b)| acc + if a != b { 1 } else { 0 })
}

pub fn diff_color(original: &DynamicImage, inpainted: &DynamicImage) -> f64 {
    original
        .pixels()
        .zip(inpainted.pixels())
        .fold(0.0, |acc, (a, b)| {
            acc + {
                a.2.channels()
                    .iter()
                    .zip(b.2.channels().iter())
                    .fold(0.0, |acc, (a, b)| {
                        acc + (f64::from(*a) - f64::from(*b)).abs()
                    })
            }
        })
}
pub fn diff_mean_square(original: &DynamicImage, inpainted: &DynamicImage) -> f64 {
    original
        .pixels()
        .zip(inpainted.pixels())
        .fold(0.0, |acc, (a, b)| {
            acc + {
                a.2.channels()
                    .iter()
                    .zip(b.2.channels().iter())
                    .fold(0.0, |acc, (a, b)| {
                        acc + (f64::from(*a) - f64::from(*b)).powi(2)
                    })
            }
        })
        / (3.0 * original.pixels().count() as f64)
}

pub fn psnr(original: &DynamicImage, reconstructed: &DynamicImage) -> f64 {
    let mse = diff_mean_square(original, reconstructed);

    let max_pixel_value: f64 = match original {
        image::DynamicImage::ImageLuma8(_) => 255.0,
        image::DynamicImage::ImageLumaA8(_) => 255.0,
        image::DynamicImage::ImageRgb8(_) => 255.0,
        image::DynamicImage::ImageRgba8(_) => 255.0,
        image::DynamicImage::ImageLuma16(_) => 65535.0,
        image::DynamicImage::ImageLumaA16(_) => 65535.0,
        image::DynamicImage::ImageRgb16(_) => 65535.0,
        image::DynamicImage::ImageRgba16(_) => 65535.0,
        image::DynamicImage::ImageRgb32F(_) => 1.0,
        image::DynamicImage::ImageRgba32F(_) => 1.0,
        &_ => panic!("Unsupported image type"),
    };

    10.0 * (max_pixel_value.powi(2) / mse).log10()
}
