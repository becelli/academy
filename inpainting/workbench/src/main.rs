mod maskenizer;
mod metrics;
mod metrics_plot;
use fmm_inpaint::{bertalmio2001, telea2004};
use image_metrics::{diff_color, diff_mean_square, diff_pixels, psnr};
use metrics::{MethodMetrics, Metric, Names};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn create_inpainting_paths() {
    ["corrupted", "mask", "restored", "metrics"]
        .iter()
        .for_each(|dir| {
            if !Path::new(format!("dataset/{}", dir).as_str()).exists() {
                std::fs::create_dir_all(format!("dataset/{}", dir)).unwrap();
            }
        });
}

fn inpaint(radius: u8, max_samples: usize) {
    create_inpainting_paths();

    let filenames: Vec<String> = std::fs::read_dir("dataset/original")
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().unwrap().is_file())
        .map(|e| e.file_name().into_string().unwrap())
        .take(max_samples)
        .collect();

    let names_len = filenames.len();
    let telea2004_metrics = Arc::new(Mutex::new(MethodMetrics::new(names_len)));
    let bertalmio2001_metrics = Arc::new(Mutex::new(MethodMetrics::new(names_len)));

    filenames.par_iter().for_each(|name| {
        let mut metric_telea = Metric::new();
        let mut metric_bertalmio = Metric::new();
        let name = Names::new_from(name);

        let img_original = image::open(name.original).unwrap();
        let (img_corrupted, img_mask) = maskenizer::corrupt_image(&img_original);

        let start = std::time::Instant::now();
        let img_result_telea = telea2004(&img_corrupted, &img_mask, radius).unwrap();
        metric_telea.timing = start.elapsed().as_millis() as f64;

        let start = std::time::Instant::now();
        let img_result_b2001 = bertalmio2001(&img_corrupted, &img_mask, radius).unwrap();
        metric_bertalmio.timing = start.elapsed().as_millis() as f64;

        img_corrupted.save(name.corrupted).unwrap();
        img_mask.save(name.mask).unwrap();
        img_result_telea.save(name.result_telea).unwrap();
        img_result_b2001.save(name.result_bertalmio).unwrap();

        run_metrics(&img_original, &img_result_telea, &mut metric_telea);
        run_metrics(&img_original, &img_result_b2001, &mut metric_bertalmio);

        telea2004_metrics.lock().unwrap().push(metric_telea);
        bertalmio2001_metrics.lock().unwrap().push(metric_bertalmio);
    });

    let telea2004_metrics = telea2004_metrics.lock().unwrap();
    let bertalmio2001_metrics = bertalmio2001_metrics.lock().unwrap();

    let headers = vec![
        "Timing",
        "Difference_of_pixels",
        "Difference_of_colors",
        "Mean_square_error",
        "Peak_signal_to_noise_ratio",
    ];

    let name_telea = "dataset/metrics/telea2004.csv";
    let name_bertalmio = "dataset/metrics/bertalmio2001.csv";
    save_csv(name_telea, &headers, &telea2004_metrics);
    save_csv(name_bertalmio, &headers, &bertalmio2001_metrics);

    metrics_plot::plot_metrics(&telea2004_metrics, name_telea);
    metrics_plot::plot_metrics(&bertalmio2001_metrics, name_bertalmio);
}

fn run_metrics(original: &image::DynamicImage, result: &image::DynamicImage, metric: &mut Metric) {
    let diff_pixels = f64::from(diff_pixels(original, result));
    let diff_color = diff_color(original, result);
    let diff_mean_square = diff_mean_square(original, result);
    let psnr = psnr(original, result);

    metric.pixel_diff = diff_pixels;
    metric.color_diff = diff_color;
    metric.mse = diff_mean_square;
    metric.psnr = psnr;
}

fn save_csv(filename: &str, headers: &[&str], metrics: &MethodMetrics) {
    let mut wtr = csv::Writer::from_path(filename).unwrap();

    wtr.write_record(headers).unwrap();
    let size = metrics.size;

    for row in 0..size {
        let mut record = csv::StringRecord::new();
        record.push_field(metrics.timings[row].to_string().as_str());
        record.push_field(metrics.pixel_diffs[row].to_string().as_str());
        record.push_field(metrics.color_diffs[row].to_string().as_str());
        record.push_field(metrics.mses[row].to_string().as_str());
        record.push_field(metrics.psnrs[row].to_string().as_str());
        wtr.write_record(&record).unwrap();
    }
    wtr.flush().unwrap();
}

fn print_folder_structure() {
    println!("The directory should look like this:");
    println!("dataset/");
    println!("├── original /");
    println!("│   ├── image1.png");
    println!("│   ├── image2.png");
    println!("│   └── image3.png");
}

fn verify_base_paths() {
    if !Path::new("dataset").exists() {
        println!("Please create a directory called 'dataset' in the root of the project.");
        print_folder_structure();
    } else if !Path::new("dataset/original").exists() {
        println!("Please create a directory called 'original' in the 'dataset' directory.");
        print_folder_structure();
    }
}
fn main() {
    verify_base_paths();

    let args: Vec<String> = std::env::args().collect();

    let options: std::collections::HashMap<String, String> = args
        .iter()
        .skip(1)
        .map(|arg| {
            let mut split = arg.split('=');
            let key = split.next().unwrap_or("").to_string();
            let value = split.next().unwrap_or("").to_string();
            (key, value)
        })
        .collect();

    let possible_options = vec!["max_samples", "radius", "max_threads"];
    // verify if user has passed some invalid options. If so, print error and exit.
    options.keys().for_each(|key| {
        if !possible_options.contains(&key.as_str()) {
            println!("Invalid option: {}", key);
            println!("Possible options: {:?}", possible_options);
            std::process::exit(1);
        }
    });

    let max_samples = options
        .get("max_samples")
        .map(|s| {
            s.parse::<usize>().unwrap_or_else(|_| {
                println!("Invalid value for max_samples. It must be a number.");
                std::process::exit(1);
            })
        })
        .unwrap_or(usize::MAX);

    let radius = options
        .get("radius")
        .map(|s| {
            s.parse::<u8>().unwrap_or_else(|_| {
                println!("Invalid value for radius. It must be a natural number.");
                std::process::exit(1);
            })
        })
        .unwrap_or(5);

    let max_threads = options
        .get("max_threads")
        .map(|s| {
            s.parse::<usize>().unwrap_or_else(|_| {
                println!("Invalid value for max_threads. It must be a natural number.");
                std::process::exit(1);
            })
        })
        .unwrap_or(num_cpus::get());

    env::set_var("RAYON_NUM_THREADS", max_threads.to_string());

    inpaint(radius, max_samples);
}
