use fmm_inpaint::{bertalmio2001, telea2004};
use image::{GenericImageView, Pixel};
mod maskenizer;
use image_metrics::{diff_color, diff_mean_square, diff_pixels, psnr};


fn inpaint() {
    let radius = 7;

    if !std::path::Path::new("samples").exists() {
        std::fs::create_dir_all("samples").unwrap();
    }

    if !std::path::Path::new("samples/result").exists() {
        std::fs::create_dir_all("samples/result").unwrap();
    }

    // find all images in samples folder
    let names: Vec<String> = std::fs::read_dir("samples")
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().unwrap().is_file())
        .map(|e| e.file_name().into_string().unwrap())
        .filter(|n| (n.ends_with(".png") || n.ends_with(".bmp")) && !n.contains("_mask"))
        .collect();

    for name in names {
        let split: Vec<&str> = name.split('.').collect();
        let (name, ext) = (split[0], split[1]);
        let img_name = format!("samples/{name}.{ext}");
        let mask_name = format!("samples/{name}_mask.{ext}");
        let img = image::open(img_name.clone()).unwrap();
        let mask = image::open(mask_name.clone()).unwrap();

        let result_telea = telea2004(&img, &mask, radius).unwrap();
        let result_bertalmio = bertalmio2001(&img, &mask, radius).unwrap();

        result_telea
            .save(format!("samples/result/{name}_telea.bmp"))
            .unwrap();
        result_bertalmio
            .save(format!("samples/result/{name}_bertalmio.bmp"))
            .unwrap();
    }
}


fn create_dataset_masks() {
    if !std::path::Path::new("dataset").exists() {
        std::fs::create_dir_all("dataset").unwrap();
    }

    // find all images in samples folder
    let names: Vec<String> = std::fs::read_dir("dataset")
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().unwrap().is_file())
        .map(|e| e.file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".png") || n.ends_with(".bmp"))
        .collect();

    let first_tree = names[0..3].to_vec();

    for name in first_tree {
        let split: Vec<&str> = name.split('.').collect();
        let (name, ext) = (split[0], split[1]);
        let img_name = format!("dataset/{name}.{ext}");
        let img = image::open(img_name.clone()).unwrap();
        let (corrupted, mask) = maskenizer::corrupt_image(&img);
        corrupted.save(format!("dataset/{name}_corrupted.{ext}")).unwrap();
        mask.save(format!("dataset/{name}_mask.{ext}")).unwrap();
    }
}


fn main() {
    // inpaint();
    create_dataset_masks();
}
