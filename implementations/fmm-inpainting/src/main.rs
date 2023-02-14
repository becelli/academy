use image::{self, DynamicImage, GenericImage, GenericImageView, Pixel, Rgba};
// use nalgebra::DMatrix;
use ordered_float::OrderedFloat;
// use rayon::prelude::{
//     IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
// };
// use rayon::slice::ParallelSlice;
use std::collections as coll;

const KNOWN: u8 = 0;
const BAND: u8 = 1;
const UNKNOWN: u8 = 2;

type Position = (i32, i32);
type Distances = Vec<Vec<f64>>;
type Conditions = Vec<Vec<u8>>;

fn area_to_inpaint(mask: &DynamicImage) -> Vec<Position> {
    mask.pixels()
        .filter(|(_, _, pixel)| pixel.channels()[0] != 0)
        .map(|(x, y, _)| (x as i32, y as i32))
        .collect()
}

fn get_neighbors_4(pos: Position) -> Vec<Position> {
    vec![
        (pos.0 - 1, pos.1),
        (pos.0 + 1, pos.1),
        (pos.0, pos.1 - 1),
        (pos.0, pos.1 + 1),
    ]
}

fn init_pixels_conditions(mask: &DynamicImage) -> Conditions {
    let (width, height) = (mask.dimensions().0 as usize, mask.dimensions().1 as usize);
    let mut conditions: Conditions = vec![vec![KNOWN; height]; width];

    mask.pixels()
        .filter(|(_, _, pixel)| pixel.channels()[0] != 0)
        .for_each(|(x, y, _)| conditions[x as usize][y as usize] = UNKNOWN);

    conditions
}

fn pixel_is_zero(mask: &DynamicImage, x: i32, y: i32) -> bool {
    mask.get_pixel(x as u32, y as u32).channels()[0] == 0
}

fn fmm_init(
    mask: &DynamicImage,
    radius: u8,
) -> Result<
    (
        Distances,
        Conditions,
        coll::BinaryHeap<(OrderedFloat<f64>, Position)>,
    ),
    String,
> {
    let (width, height) = (mask.dimensions().0 as usize, mask.dimensions().1 as usize);

    let mut distances: Distances = vec![vec![f64::MAX; height]; width];

    let mut conditions: Conditions = init_pixels_conditions(mask);

    let mut field: coll::BinaryHeap<(OrderedFloat<f64>, Position)> = coll::BinaryHeap::new();

    let pixels_to_inpaint: Vec<Position> = area_to_inpaint(mask);

    for pos in pixels_to_inpaint {
        let neighbors = get_neighbors_4(pos);
        for (nx, ny) in neighbors {
            // As we are working with unsigned integers, if (x, y) is (0, 0), then (nx, ny) will be (u32::MAX, u32::MAX).
            if nx < width as i32
                && ny < height as i32
                && nx >= 0
                && ny >= 0
                && conditions[nx as usize][ny as usize] != BAND
                && pixel_is_zero(mask, nx, ny)
            {
                conditions[nx as usize][ny as usize] = BAND;
                distances[nx as usize][ny as usize] = 0.0;
                field.push((OrderedFloat(0.0), (nx, ny)));
            }
        }
    }

    calc_outside_distances(&mut distances, &mut conditions, &mut field, radius);

    Ok((distances, conditions, field))
}

fn calc_outside_distances(
    distances: &mut Distances,
    conditions: &Conditions,
    field: &coll::BinaryHeap<(OrderedFloat<f64>, Position)>,
    radius: u8,
) {
    let (width, height) = (distances.len(), distances[0].len());

    // Swap conditions values

    let mut l_conditions: Conditions = vec![vec![UNKNOWN; height]; width];
    let mut l_field = field.clone();

    // Swap UNKNOWN and KNOWN values
    // });
    conditions.iter().enumerate().for_each(|(i, c)| {
        c.iter().enumerate().for_each(|(j, &v)| {
            l_conditions[i][j] = match v {
                UNKNOWN => KNOWN,
                KNOWN => UNKNOWN,
                _ => v,
            }
        })
    });

    let mut min_dist: f64 = 0.0;
    let diameter = radius as f64 * 2.0;

    while let Some((_, (x, y))) = l_field.pop() {
        if min_dist >= diameter {
            break;
        }

        // l_conditions[x as usize][y as usize] = KNOWN;
        l_conditions[x as usize][y as usize] = KNOWN;
        let neighbors = get_neighbors_4((x, y));
        for (nx, ny) in neighbors {
            if nx < width as i32
                && ny < height as i32
                && conditions[nx as usize][ny as usize] == UNKNOWN
            {
                let top_left = eikonal_solve(nx - 1, ny - 1, x, y, distances, conditions);
                let top_right = eikonal_solve(nx + 1, ny - 1, x, y, distances, conditions);
                let bottom_right = eikonal_solve(nx + 1, ny + 1, x, y, distances, conditions);
                let bottom_left = eikonal_solve(nx - 1, ny + 1, x, y, distances, conditions);
                min_dist = top_left.min(top_right).min(bottom_right).min(bottom_left);

                distances[nx as usize][ny as usize] = min_dist;
                l_conditions[nx as usize][ny as usize] = BAND;
                l_field.push((OrderedFloat(min_dist), (nx, ny)));
            }
        }
    }

    distances
        .iter_mut()
        .for_each(|d| d.iter_mut().for_each(|v| *v *= -1.0));
}

fn eikonal_solve(
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    distances: &Distances,
    conditions: &Conditions,
) -> f64 {
    let (width, height) = (distances.len(), distances[0].len());

    if x1 >= width as i32
        || y1 >= height as i32
        || x2 >= width as i32
        || y2 >= height as i32
        || x1 < 0
        || y1 < 0
        || x2 < 0
        || y2 < 0
    {
        return f64::MAX;
    }

    let cond_1 = conditions[x1 as usize][y1 as usize];
    let cond_2 = conditions[x2 as usize][y2 as usize];

    if cond_1 == KNOWN && cond_2 == KNOWN {
        let dist_1 = distances[x1 as usize][y1 as usize];
        let dist_2 = distances[x2 as usize][y2 as usize];

        let diff = 2.0 - (dist_1 - dist_2).powi(2);

        if diff > 0.0 {
            let r = diff.sqrt();
            let mut s = (dist_1 + dist_2 + r) / 2.0;
            if s >= dist_1 && s >= dist_2 {
                return s;
            }
            s += r;
            if s >= dist_1 && s >= dist_2 {
                return s;
            }
            return f64::MAX;
        }
    }

    if cond_1 == KNOWN {
        return 1.0 + distances[x1 as usize][y1 as usize];
    }

    if cond_2 == KNOWN {
        return 1.0 + distances[x2 as usize][y2 as usize];
    }

    f64::MAX
}

// fast_marching_method for inpainting (Telea, 2004)
fn fmm_inpainting(
    img: &DynamicImage,
    mask: &DynamicImage,
    radius: u8,
) -> Result<DynamicImage, String> {
    let (width, height) = img.dimensions();

    if (width, height) != mask.dimensions() {
        return Err("Image and mask must have the same dimensions".to_string());
    } else if radius < 1 {
        return Err("Radius must be greater than 0".to_string());
    } else if radius > 15 {
        return Err("Radius must be less than 16".to_string());
    }

    let mut result = img.clone();
    let (mut distances, mut conditions, mut field) = fmm_init(mask, radius)?;

    while let Some((_, (x, y))) = field.pop() {
        conditions[x as usize][y as usize] = KNOWN;
        let neighbors = get_neighbors_4((x, y));
        for (nx, ny) in neighbors {
            if nx < width as i32
                && ny < height as i32
                && nx > 0
                && ny > 0
                && conditions[nx as usize][ny as usize] == UNKNOWN
            {
                let top_left = eikonal_solve(nx - 1, ny - 1, x, y, &distances, &conditions);
                let top_right = eikonal_solve(nx + 1, ny - 1, x, y, &distances, &conditions);
                let bottom_right = eikonal_solve(nx + 1, ny + 1, x, y, &distances, &conditions);
                let bottom_left = eikonal_solve(nx - 1, ny + 1, x, y, &distances, &conditions);
                let min_dist = top_left.min(top_right).min(bottom_right).min(bottom_left);

                distances[nx as usize][ny as usize] = min_dist;
                inpaint_pixel(
                    &mut result,
                    nx as i32,
                    ny as i32,
                    &distances,
                    &conditions,
                    radius as i32,
                );
                conditions[nx as usize][ny as usize] = BAND;
                field.push((OrderedFloat(min_dist), (nx, ny)));
            }
        }
    }

    Ok(result)
}

fn inpaint_pixel(
    img: &mut DynamicImage,
    x: i32,
    y: i32,
    distances: &Distances,
    conditions: &Conditions,
    radius: i32,
) {
    let (width, height) = (img.width() as i32, img.height() as i32);

    let dist = distances[x as usize][y as usize];

    let (d_grad_x, d_grad_y) = pixel_gradient(&distances, x as usize, y as usize, &conditions);

    let mut new_pixel: [f64; 4] = [0.0; 4];
    let mut weight_sum = 0.0;

    for nb_x in x - radius..=x + radius {
        for nb_y in y - radius..=y + radius {
            if nb_x < width
                && nb_y < height
                && nb_x >= 0
                && nb_y >= 0
                && conditions[nb_x as usize][nb_y as usize] != UNKNOWN
            {
                let vec_x = x - nb_x;
                let vec_y = y - nb_y;
                let vec_len_squared = (vec_x.pow(2) + vec_y.pow(2)) as f64;
                let vec_len = vec_len_squared.sqrt();

                if vec_len > radius as f64 {
                    continue;
                }

                let vec_factor: f64 = (vec_x as f64 * d_grad_x + vec_y as f64 * d_grad_y)
                    .abs()
                    .max(f64::EPSILON);

                let nb_dist = distances[nb_x as usize][nb_y as usize];
                let level_factor = 1.0 / (1.0 + (nb_dist - dist).abs());

                let dist_factor = 1.0 / vec_len.powi(3);
                let weight = vec_factor * level_factor * dist_factor;

                let pixel = img.get_pixel(nb_x as u32, nb_y as u32);

                for i in 0..4 {
                    new_pixel[i] += pixel[i] as f64 * weight;
                }

                weight_sum += weight;
            }
        }
    }

    let mut pixel_val: [u8; 4] = [0; 4];
    for i in 0..4 {
        pixel_val[i] = (new_pixel[i] / weight_sum).round() as u8;
    }

    img.put_pixel(x as u32, y as u32, Rgba(pixel_val));
}

fn pixel_gradient(dists: &Distances, x: usize, y: usize, conds: &Conditions) -> (f64, f64) {
    let (grad_x, grad_y): (f64, f64);

    let (width, height) = (dists.len(), dists[0].len());

    let dist = dists[x as usize][y as usize];

    let (prev_x, next_x) = (x - 1, x + 1);
    if next_x >= width - 1 {
        grad_x = f64::MAX
    } else {
        // println!("prev_x: {}, y: {} of max: {}, {}", prev_x, y, width, height);
        let prev_cond = conds[prev_x][y];
        // println!("Next_x: {}, y: {} of max: {}, {}", next_x, y, width, height);
        let next_cond = conds[next_x][y];

        grad_x = match (prev_cond != UNKNOWN, next_cond != UNKNOWN) {
            (true, true) => (dists[next_x][y] - dists[prev_x][y]) / 2.0,
            (true, false) => dist - dists[prev_x][y],
            (false, true) => dists[next_x][y] - dist,
            (false, false) => 0.0,
        };
    }

    let (prev_y, next_y) = (y - 1, y + 1);
    if next_y >= height {
        grad_y = f64::MAX
    } else {
        let prev_cond = conds[x][prev_y];
        let next_cond = conds[x][next_y];

        grad_y = match (prev_cond != UNKNOWN, next_cond != UNKNOWN) {
            (true, true) => (dists[x][next_y] - dists[x][prev_y]) / 2.0,
            (true, false) => dist - dists[x][prev_y],
            (false, true) => dists[x][next_y] - dist,
            (false, false) => 0.0,
        };
    }

    (grad_x, grad_y)
}

fn main() {
    let img: DynamicImage = image::open("samples/text-horse.png").unwrap();
    let mask = image::open("samples/text-horse-mask.png").unwrap();
    let radius = 5;

    let result = fmm_inpainting(&img, &mask, radius).unwrap();
    result.save("samples/text-horse-inpainted.png").unwrap();
}
