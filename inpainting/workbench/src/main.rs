mod maskenizer;
use fmm_inpaint::{bertalmio2001, telea2004};
use image_metrics::{diff_color, diff_mean_square, diff_pixels, psnr};
use plotters::prelude::*;

use std::path::Path;

// use Jemalloc as global allocator
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn inpaint(max_samples: usize) {
    if !Path::new("dataset/corrupted").exists() {
        std::fs::create_dir_all("dataset/corrupted").unwrap();
    }

    if !Path::new("dataset/mask").exists() {
        std::fs::create_dir_all("dataset/mask").unwrap();
    }

    if !Path::new("dataset/result").exists() {
        std::fs::create_dir_all("dataset/result").unwrap();
    }

    let names: Vec<String> = std::fs::read_dir("dataset/original")
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().unwrap().is_file())
        .map(|e| e.file_name().into_string().unwrap())
        .take(if max_samples > 0 {
            max_samples
        } else {
            usize::MAX
        })
        .collect();

    println!("Found {} images:", names.len());

    let names_len = names.len();
    let mut timings_t2004 = Vec::with_capacity(names_len);
    let mut pixel_diffs_t2004 = Vec::with_capacity(names_len);
    let mut color_diffs_t2004 = Vec::with_capacity(names_len);
    let mut mses_t2004 = Vec::with_capacity(names_len);
    let mut psnrs_t2004 = Vec::with_capacity(names_len);

    let mut timings_b2001 = Vec::with_capacity(names_len);
    let mut pixel_diffs_b2001 = Vec::with_capacity(names_len);
    let mut color_diffs_b2001 = Vec::with_capacity(names_len);
    let mut mses_b2001 = Vec::with_capacity(names_len);
    let mut psnrs_b2001 = Vec::with_capacity(names_len);

    for name in names {
        println!("Processing image {}", name);
        let split: Vec<&str> = name.split('.').collect();
        let (name, ext) = (split[0], split[1]);
        let name_original = format!("dataset/original/{name}.{ext}");
        let name_corrupted = format!("dataset/mask/{name}_mask.{ext}");
        let name_mask = format!("dataset/corrupted/{name}_corrupted.{ext}");
        let name_result_t2004 = format!("dataset/result/{name}_telea.{ext}");
        let name_result_b2001 = format!("dataset/result/{name}_bertalmio.{ext}");
        let img_original = image::open(name_original.clone()).unwrap();
        // if fails, create mask and corrupted image
        let (img_corrupted, img_mask) =
            if Path::new(&name_corrupted).exists() && Path::new(&name_mask).exists() {
                (
                    image::open(name_corrupted.clone()).unwrap(),
                    image::open(name_mask.clone()).unwrap(),
                )
            } else {
                let (corrupted, mask) = maskenizer::corrupt_image(&img_original);
                corrupted.save(name_corrupted.clone()).unwrap();
                mask.save(name_mask.clone()).unwrap();
                (corrupted, mask)
            };

        let radius = 7;

        let start_t2004 = std::time::Instant::now();
        let result_telea = telea2004(&img_corrupted, &img_mask, radius).unwrap();
        let elapsed_t2004 = start_t2004.elapsed().as_millis() as f64;

        let start_b2001 = std::time::Instant::now();
        let result_b2001 = bertalmio2001(&img_corrupted, &img_mask, radius).unwrap();
        let elapsed_b2001 = start_b2001.elapsed().as_millis() as f64;

        result_telea.save(name_result_t2004.clone()).unwrap();
        result_b2001.save(name_result_b2001.clone()).unwrap();

        let (diff_pixels_telea, diff_color_telea, diff_mean_square_telea, psnr_telea) =
            run_metrics(&img_original, &result_telea);

        let (diff_pixels_b2001, diff_color_b2001, diff_mean_square_b2001, psnr_b2001) =
            run_metrics(&img_original, &result_b2001);

        timings_t2004.push(elapsed_t2004);
        pixel_diffs_t2004.push(diff_pixels_telea);
        color_diffs_t2004.push(diff_color_telea);
        mses_t2004.push(diff_mean_square_telea);
        psnrs_t2004.push(psnr_telea);

        timings_b2001.push(elapsed_b2001);
        pixel_diffs_b2001.push(diff_pixels_b2001);
        color_diffs_b2001.push(diff_color_b2001);
        mses_b2001.push(diff_mean_square_b2001);
        psnrs_b2001.push(psnr_b2001);
    }

    save_to_csv(
        "metrics_t2004.csv",
        vec!["timings", "pixel_diffs", "color_diffs", "mses", "psnrs"],
        vec![
            timings_t2004,
            pixel_diffs_t2004,
            color_diffs_t2004,
            mses_t2004,
            psnrs_t2004,
        ],
    );

    save_to_csv(
        "metrics_b2001.csv",
        vec!["timings", "pixel_diffs", "color_diffs", "mses", "psnrs"],
        vec![
            timings_b2001,
            pixel_diffs_b2001,
            color_diffs_b2001,
            mses_b2001,
            psnrs_b2001,
        ],
    );
}

fn run_metrics(
    original: &image::DynamicImage,
    result: &image::DynamicImage,
) -> (f64, f64, f64, f64) {
    let diff_pixels = f64::from(diff_pixels(original, result));
    let diff_color = diff_color(original, result);
    let diff_mean_square = diff_mean_square(original, result);
    let psnr = psnr(original, result);
    (diff_pixels, diff_color, diff_mean_square, psnr)
}

fn save_to_csv(filename: &str, headers: Vec<&str>, data: Vec<Vec<f64>>) {
    // use csv crate
    let mut wtr = csv::Writer::from_path(filename).unwrap();
    // write header
    wtr.write_record(&headers).unwrap();
    // write data
    for i in 0..data[0].len() {
        let mut row = Vec::with_capacity(data.len());
        for j in 0..data.len() {
            row.push(data[j][i].to_string());
        }
        wtr.write_record(row).unwrap();
    }
    wtr.flush().unwrap();

    // Plot data using plotters
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let filename_img = filename.replace(".csv", ".png");
    let mut chart = ChartBuilder::on(&root)
        .caption(filename_img, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..data[0].len() as f64, 0f64..data[0].len() as f64)
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            data[0].iter().enumerate().map(|(x, y)| (x as f64, *y)),
            &RED,
        ))
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            data[1].iter().enumerate().map(|(x, y)| (x as f64, *y)),
            &BLUE,
        ))
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            data[2].iter().enumerate().map(|(x, y)| (x as f64, *y)),
            &GREEN,
        ))
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            data[3].iter().enumerate().map(|(x, y)| (x as f64, *y)),
            &BLACK,
        ))
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            data[4].iter().enumerate().map(|(x, y)| (x as f64, *y)),
            &YELLOW,
        ))
        .unwrap();

    root.present().unwrap();
}

fn print_folder_structure() {
    println!("The directory should look like this:");
    println!("dataset/");
    println!("├── original /");
    println!("│   ├── image1.png");
    println!("│   ├── image2.png");
    println!("│   └── image3.png");
}

fn main() {
    if !Path::new("dataset").exists() {
        println!("Please create a directory called 'dataset' in the root of the project.");
        print_folder_structure();
        return;
    } else if !Path::new("dataset/original").exists() {
        println!("Please create a directory called 'original' in the 'dataset' directory.");
        print_folder_structure();
        return;
    }

    // get arg for max samples

    let args: Vec<String> = std::env::args().collect();
    let max_samples = if args.len() > 1 {
        args[1].parse::<usize>().unwrap()
    } else {
        0
    };

    inpaint(max_samples);
}
