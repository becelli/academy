mod defs;
use defs::{get_mat, new_heap, Distances, Heap, Position, State, States, DIST_MAX};
use image::{self, DynamicImage, GenericImage, GenericImageView, Pixel, Rgba};
use ordered_float::OrderedFloat;

fn get_neighbors_4(pos: Position) -> Vec<Position> {
    vec![
        (pos.0 - 1, pos.1),
        (pos.0 + 1, pos.1),
        (pos.0, pos.1 - 1),
        (pos.0, pos.1 + 1),
    ]
}

fn init(mask: &DynamicImage, radius: u8) -> (Distances, States, Heap) {
    let (width, height) = (mask.dimensions().0 as i32, mask.dimensions().1 as i32);

    let mut distances: Distances = get_mat(width as usize, height as usize, DIST_MAX);
    let mut states: States = get_mat(width as usize, height as usize, State::Known);
    let mut field: Heap = new_heap();

    mask.pixels()
        .filter(|(_, _, pixel)| pixel.channels()[0] != 0)
        .for_each(|(x, y, _)| {
            states[x as usize][y as usize] = State::Unknown;

            let neighbors: Vec<Position> = get_neighbors_4((x as i32, y as i32));

            for (nx, ny) in neighbors {
                if nx < (width - 1) && ny < (height - 1) && nx > 0 && ny > 0 {
                    let state = &mut states[nx as usize][ny as usize];
                    let pixel_is_zero = mask.get_pixel(nx as u32, ny as u32).channels()[0] == 0;

                    if !state.is_band() && pixel_is_zero {
                        *state = State::Band;
                        distances[nx as usize][ny as usize] = 0f64;
                        field.push((OrderedFloat(0.0), (nx, ny)));
                    }
                }
            }
        });

    calc_outside_distances(&mut distances, &mut states, &mut field, radius);

    (distances, states, field)
}

fn calc_outside_distances(
    distances: &mut Distances,
    states: &mut States,
    field: &mut Heap,
    radius: u8,
) {
    let (width, height) = (distances.len(), distances[0].len());

    let mut min_dist: f64 = 0.0;
    let diameter = radius as f64 * 2.0;

    while let Some((_, (x, y))) = field.pop() {
        if min_dist >= diameter {
            break;
        }

        states[x as usize][y as usize] = State::Change;
        let neighbors = get_neighbors_4((x, y));
        for (nx, ny) in neighbors {
            if nx < (width - 1) as i32
                && ny < (height - 1) as i32
                && nx > 0
                && ny > 0
                && states[nx as usize][ny as usize].is_unknown()
            {
                let p1 = eikonal_solve(nx - 1, ny, nx, ny - 1, &distances, &states);
                let p2 = eikonal_solve(nx + 1, ny, nx, ny + 1, &distances, &states);
                let p3 = eikonal_solve(nx - 1, ny, nx, ny + 1, &distances, &states);
                let p4 = eikonal_solve(nx + 1, ny, nx, ny - 1, &distances, &states);
                min_dist = p1.min(p2).min(p3).min(p4);

                distances[nx as usize][ny as usize] = min_dist;
                states[nx as usize][ny as usize] = State::Band;
                field.push((OrderedFloat(min_dist), (nx, ny)));
            }
        }
    }

    distances.iter_mut().enumerate().for_each(|(i, c)| {
        c.iter_mut().enumerate().for_each(|(j, v)| {
            if states[i][j].is_change() {
                *v *= -1.0;
                states[i][j] = State::Known;
            }
        })
    });
}

fn eikonal_solve(
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    distances: &Distances,
    states: &States,
) -> f64 {
    let (width, height) = (distances.len() as i32, distances[0].len() as i32);

    if x1 < 0
        || y1 < 0
        || x2 < 0
        || y2 < 0
        || x1 >= width
        || y1 >= height
        || x2 >= width
        || y2 >= height
    {
        assert!(false);
        return DIST_MAX;
    }

    let d1 = distances[x1 as usize][y1 as usize];
    let d2 = distances[x2 as usize][y2 as usize];
    let d_min = d1.min(d2);

    let state_1: &State = &states[x1 as usize][y1 as usize];
    let state_2: &State = &states[x2 as usize][y2 as usize];
    let sol: f64;

    if !state_1.is_unknown() {
        if !state_2.is_unknown() {
            sol = if (d1 - d2).abs() >= 1.0 {
                1.0 + d_min
            } else {
                (d1 + d2 + (2.0 - (d1 - d2).powi(2)).sqrt()) * 0.5
            }
        } else {
            sol = 1.0 + d1;
        }
    } else if !state_2.is_unknown() {
        sol = 1.0 + d2;
    } else {
        sol = 1.0 + d_min;
    }

    sol
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
    }

    let mut result = img.clone();
    let (mut distances, mut states, mut field) = init(mask, radius);

    while let Some((_, (x, y))) = field.pop() {
        states[x as usize][y as usize] = State::Known;
        let neighbors = get_neighbors_4((x, y));
        for (nx, ny) in neighbors {
            if nx < width as i32
                && ny < height as i32
                && nx > 0
                && ny > 0
                && states[nx as usize][ny as usize].is_unknown()
            {
                let p1 = eikonal_solve(nx - 1, ny, nx, ny - 1, &distances, &states);
                let p2 = eikonal_solve(nx + 1, ny, nx, ny + 1, &distances, &states);
                let p3 = eikonal_solve(nx - 1, ny, nx, ny + 1, &distances, &states);
                let p4 = eikonal_solve(nx + 1, ny, nx, ny - 1, &distances, &states);
                let min_dist = p1.min(p2).min(p3).min(p4);

                distances[nx as usize][ny as usize] = min_dist;
                inpaint_pixel(&mut result, nx, ny, &distances, &states, radius as i32);
                states[nx as usize][ny as usize] = State::Band;
                field.push((OrderedFloat(min_dist), (nx, ny)));
            }
        }
    }

    Ok(result)
}

#[inline(always)]
fn inpaint_pixel(
    img: &mut DynamicImage,
    x: i32,
    y: i32,
    distances: &Distances,
    states: &States,
    radius: i32,
) {
    let (width, height) = (img.width() as i32, img.height() as i32);

    let dist = distances[x as usize][y as usize];

    let (d_grad_x, d_grad_y) = pixel_gradient(distances, x as usize, y as usize, states);

    let mut new_pixel: [f64; 4] = [0.0; 4];
    let mut weight_sum = 0.0;

    for nb_x in x - radius..=x + radius {
        for nb_y in y - radius..=y + radius {
            if nb_x < width
                && nb_y < height
                && nb_x >= 0
                && nb_y >= 0
                && !states[nb_x as usize][nb_y as usize].is_unknown()
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

fn pixel_gradient(dists: &Distances, x: usize, y: usize, states: &States) -> (f64, f64) {
    let (grad_x, grad_y): (f64, f64);

    let (width, height) = (dists.len(), dists[0].len());

    let dist = dists[x][y];

    let (prev_x, next_x) = (x - 1, x + 1);
    if next_x >= width - 1 {
        grad_x = DIST_MAX
    } else {
        let prev_state: &State = &states[prev_x][y];

        let next_state: &State = &states[next_x][y];

        grad_x = match (!prev_state.is_unknown(), !next_state.is_unknown()) {
            (true, true) => (dists[next_x][y] - dists[prev_x][y]) / 2.0,
            (true, false) => dist - dists[prev_x][y],
            (false, true) => dists[next_x][y] - dist,
            (false, false) => 0.0,
        };
    }

    let (prev_y, next_y) = (y - 1, y + 1);
    if next_y >= height {
        grad_y = DIST_MAX
    } else {
        let prev_state: &State = &states[x][prev_y];
        let next_state: &State = &states[x][next_y];

        grad_y = match (!prev_state.is_unknown(), !next_state.is_unknown()) {
            (true, true) => (dists[x][next_y] - dists[x][prev_y]) / 2.0,
            (true, false) => dist - dists[x][prev_y],
            (false, true) => dists[x][next_y] - dist,
            (false, false) => 0.0,
        };
    }

    (grad_x, grad_y)
}

fn main() {
    let radius = 5;

    let names = ["becelli", "text-horse"];

    for name in names.iter() {
        let img: DynamicImage = image::open(format!("samples/{}.png", name)).unwrap();
        let mask: DynamicImage = image::open(format!("samples/{}_mask.png", name)).unwrap();
        let result = fmm_inpainting(&img, &mask, radius).unwrap();
        result.save(format!("samples/{}_result.png", name)).unwrap();
    }
}
