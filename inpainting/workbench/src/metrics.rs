pub struct MethodMetrics {
    pub size: usize,
    pub timings: Vec<f64>,
    pub pixel_diffs: Vec<f64>,
    pub color_diffs: Vec<f64>,
    pub mses: Vec<f64>,
    pub psnrs: Vec<f64>,
}

pub struct Metric {
    pub timing: f64,
    pub pixel_diff: f64,
    pub color_diff: f64,
    pub mse: f64,
    pub psnr: f64,
}

pub struct Names {
    pub original: String,
    pub corrupted: String,
    pub mask: String,
    pub result_telea: String,
    pub result_bertalmio: String,
}

impl Names {
    pub fn new_from(filename: &str) -> Self {
        let (name, ext) = filename.split_once('.').unwrap();
        Self {
            original: format!("dataset/original/{name}.{ext}"),
            corrupted: format!("dataset/corrupted/{name}_corrupted.{ext}"),
            mask: format!("dataset/mask/{name}_mask.{ext}"),
            result_telea: format!("dataset/restored/{name}_telea.{ext}"),
            result_bertalmio: format!("dataset/restored/{name}_bertalmio.{ext}"),
        }
    }
}

impl MethodMetrics {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            timings: Vec::with_capacity(size),
            pixel_diffs: Vec::with_capacity(size),
            color_diffs: Vec::with_capacity(size),
            mses: Vec::with_capacity(size),
            psnrs: Vec::with_capacity(size),
        }
    }

    pub fn push(&mut self, metric: Metric) {
        self.timings.push(metric.timing);
        self.pixel_diffs.push(metric.pixel_diff);
        self.color_diffs.push(metric.color_diff);
        self.mses.push(metric.mse);
        self.psnrs.push(metric.psnr);
    }
}

impl Metric {
    pub fn new() -> Self {
        Self {
            timing: 0.0,
            pixel_diff: 0.0,
            color_diff: 0.0,
            mse: 0.0,
            psnr: 0.0,
        }
    }
}
