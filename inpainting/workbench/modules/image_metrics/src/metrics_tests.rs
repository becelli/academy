#[cfg(test)]
mod metrics_tests {
    use crate::*;
    use image::{GenericImage, DynamicImage, GenericImageView};

    #[test]
    pub fn diff_pixels_fn() {
        let original = DynamicImage::new_rgb8(25, 25);
        assert_eq!(diff_pixels(&original, &original), 0);

        let mut inpainted = original.clone();
        inpainted.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
        assert_eq!(diff_pixels(&original, &inpainted), 1);

        let mut inpainted = original.clone();
        let (width, height) = inpainted.dimensions();
        (0..width).for_each(|x| {
            (0..height).for_each(|y| {
                inpainted.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            })
        });
        assert_eq!(diff_pixels(&original, &inpainted), width * height);
    }

    #[test]
    pub fn diff_color_fn() {
        let original = DynamicImage::new_rgb8(25, 25);
        assert_eq!(diff_color(&original, &original), 0.0);

        let mut inpainted = original.clone();
        inpainted.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
        assert_eq!(diff_color(&original, &inpainted), 765.0);

        let mut inpainted = original.clone();
        let (width, height) = inpainted.dimensions();
        (0..width).for_each(|x| {
            (0..height).for_each(|y| {
                inpainted.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            })
        });
        assert_eq!(
            diff_color(&original, &inpainted),
            765.0 * f64::from(width) * f64::from(height)
        );
    }

    #[test]
    pub fn diff_mean_square_fn() {
        let original = DynamicImage::new_rgb8(25, 25);
        assert_eq!(diff_mean_square(&original, &original), 0.0);
        let (width, height) = original.dimensions();

        let mut inpainted = original.clone();
        inpainted.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
        assert_eq!(
            diff_mean_square(&original, &inpainted),
            (255f64 * 255f64) / (width * height) as f64
        );

        let mut inpainted = original.clone();

        (0..width).for_each(|x| {
            (0..height).for_each(|y| {
                inpainted.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            })
        });
        assert_eq!(diff_mean_square(&original, &inpainted), 255f64 * 255f64);
    }

    #[test]
    pub fn psnr_fn() {
        let original = DynamicImage::new_rgb8(25, 25);
        assert_eq!(psnr(&original, &original), f64::INFINITY);
        let (width, height) = original.dimensions();

        let mut inpainted = original.clone();
        inpainted.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
        let mean_square = diff_mean_square(&original, &inpainted);
        assert_eq!(
            psnr(&original, &inpainted),
            (255f64.powi(2) / mean_square).log10() * 10.0
        );

        let mut inpainted = original.clone();

        (0..width).for_each(|x| {
            (0..height).for_each(|y| {
                inpainted.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            })
        });

        // totally different image
        assert_eq!(psnr(&original, &inpainted), 0.0);
    }
}
