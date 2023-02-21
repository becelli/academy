use super::neighborhood::{get_neighbors_4, get_neighbors_n};
use super::def::{get_mat, Distances, Heap, Point, State, States, DIST_MAX};
use image::{self, DynamicImage, GenericImage, GenericImageView, Pixel, Rgba};

pub fn get_initial_conditions(mask: &DynamicImage) -> (Distances, States, Heap) {
    /*
    Parameters
    ----------
    mask : ndarray
        2D array of 0s and 1s
    radius : int
        Radius of the disk structuring element

    Returns
    -------
    distances : ndarray
        2D array of distances. Initialized to MAX_DIST everywhere except
        at the border of the mask, where it is initialized to 0.
    states : ndarray
        2D array of states. Initialized to State.KNOWN. The border of the
        mask is initialized to State.BAND. The interior of the mask is
        initialized to State.UNKNOWN.
    heap : Heap
        Binary heap of (distance, position) tuples. Initialized with the
        border of the mask.
     */

    let (width, height) = (mask.dimensions().0 as i32, mask.dimensions().1 as i32);

    let mut distances: Distances = get_mat(width as usize, height as usize, DIST_MAX);
    let mut states: States = get_mat(width as usize, height as usize, State::Known);
    let mut heap: Heap = Heap::new();

    mask.pixels()
        .filter(|(i, j, pixel)| {
            pixel.channels()[0] != 0
              //  keep borders as known
              && *i < width as u32 - 1
              && *j < height as u32 - 1
              && *i > 0
              && *j > 0
        })
        .for_each(|(x, y, _)| {
            states[x as usize][y as usize] = State::Unknown;

            let neighbors = get_neighbors_4(Point::<i32>::new(x as i32, y as i32));

            for nb in neighbors {
                if nb.x < (width - 1) && nb.y < (height - 1) && nb.x > 0 && nb.y > 0 {
                    let state = &mut states[nb.x as usize][nb.y as usize];
                    let pixel_is_zero = mask.get_pixel(nb.x as u32, nb.y as u32).channels()[0] == 0;

                    if pixel_is_zero && !state.is_band() {
                        *state = State::Band;
                        distances[nb.x as usize][nb.y as usize] = 0.0;
                        heap.push(0.0, nb);
                    }
                }
            }
        });

    (distances, states, heap)
}

pub fn get_telea_distances(mask: &DynamicImage, dist: &mut Distances, states: &States, radius: u8) {
    let (width, height) = (dist.len() as i32, dist[0].len() as i32);

    let mut heap = Heap::new();
    let mut new_states = states.clone();

    mask.pixels()
        .filter(|(i, j, pixel)| {
            pixel.channels()[0] != 0
              //  keep borders as known
              && *i < width as u32 - 1
              && *j < height as u32 - 1
              && *i > 0
              && *j > 0
        })
        .for_each(|(x, y, _)| {
            // dilate with disk structuring element of (2 * radius + 1), marking as band
            let neighbors = get_neighbors_n(Point::<i32>::new(x as i32, y as i32), radius as i32);

            for nb in neighbors {
                if nb.x < (width - 1) && nb.y < (height - 1) && nb.x > 0 && nb.y > 0 {
                    let state = &mut new_states[nb.x as usize][nb.y as usize];
                    let pixel_is_not_mask =
                        mask.get_pixel(nb.x as u32, nb.y as u32).channels()[0] == 0;

                    if pixel_is_not_mask && !state.is_band() {
                        // todo: check if this is correct
                        *state = State::Change;
                        dist[nb.x as usize][nb.y as usize] = DIST_MAX;
                        heap.push(DIST_MAX, nb);
                    } else if pixel_is_not_mask && state.is_band() {
                        dist[nb.x as usize][nb.y as usize] = 0.0;
                        heap.push(0.0, nb);
                    }
                }
            }
        });

    while let Some((_, pos)) = heap.pop() {
        new_states[pos.x as usize][pos.y as usize] = State::Change;

        let neighbors = get_neighbors_4(pos);
        for nb in neighbors {
            if nb.x < width - 1
                && nb.y < height - 1
                && nb.x > 0
                && nb.y > 0
                && new_states[nb.x as usize][nb.y as usize].is_unknown()
            {
                let min = get_min_dist(&nb, dist, &new_states);

                dist[nb.x as usize][nb.y as usize] = min;
                new_states[nb.x as usize][nb.y as usize] = State::Band;
                heap.push(min, nb);
            }
        }
    }

    for i in 0..width as usize {
        for j in 0..height as usize {
            if new_states[i][j].is_change() {
                // remove this
                // states[i][j] = State::Known;
                dist[i][j] *= -1.0;
            }
        }
    }
}

pub fn solve_eikonal(x1: i32, y1: i32, x2: i32, y2: i32, dist: &Distances, states: &States) -> f64 {
    let (width, height) = (dist.len() as i32, dist[0].len() as i32);

    if x1 < 0
        || y1 < 0
        || x2 < 0
        || y2 < 0
        || x1 >= width
        || y1 >= height
        || x2 >= width
        || y2 >= height
    {
        panic!("Index out of bounds");
    }

    let dist1 = dist[x1 as usize][y1 as usize];
    let dist2 = dist[x2 as usize][y2 as usize];
    let dist_min = dist1.min(dist2);
    let dist_sub = (dist1 - dist2).abs();

    let state_1: &State = &states[x1 as usize][y1 as usize];
    let state_2: &State = &states[x2 as usize][y2 as usize];

    let solution: f64 = match (state_1, state_2) {
        (State::Unknown, State::Unknown) => 1.0 + dist_min,
        (_, State::Unknown) => 1.0 + dist1,
        (State::Unknown, _) => 1.0 + dist2,
        _ => match dist_sub >= 1.0 {
            true => 1.0 + dist_min,
            false => 0.5 * (dist1 + dist2 + (2.0 - dist_sub.powi(2)).sqrt()),
        },
    };

    solution
}

pub fn telea_pixel(
    img: &mut DynamicImage,
    i: i32,
    j: i32,
    distances: &Distances,
    states: &States,
    radius: i32,
) {
    let (width, height) = (img.width() as i32, img.height() as i32);

    let dist = distances[i as usize][j as usize];

    let d_grad = get_pixel_gradient(distances, i as usize, j as usize, states);

    let mut r: Point<f64> = Point::<f64>::new(0.0, 0.0);
    let mut grad_i: Point<f64> = Point::<f64>::new(0.0, 0.0);
    let mut j_x = [0.0; 3];
    let mut j_y = [0.0; 3];
    let mut i_a = [0.0; 3];
    let mut s = [f64::EPSILON; 3];
    let (mut weight, mut dist_factor, mut level_factor, mut vec_factor);

    for k in i - radius..=i + radius {
        for l in j - radius..=j + radius {
            let km = (k - 1 + (k == 1) as i32) as u32;
            let kp = (k - 1 - (k == (width - 2)) as i32) as u32;
            let lm = (l - 1 + (l == 1) as i32) as u32;
            let lp = (l - 1 - (l == (height - 2)) as i32) as u32;

            if k > 0
                && l > 0
                && k < width - 1
                && l < height - 1
                && !states[k as usize][l as usize].is_unknown()
                && ((k - i).pow(2) + (l - j).pow(2)) <= radius.pow(2)
            {
                for color in 0..3 {
                    r.x = (j - l).into();
                    r.y = (i - k).into();

                    let vec_len = r.len();

                    dist_factor = 1.0 / (vec_len * vec_len.sqrt());

                    level_factor = 1.0 / (1.0 + (distances[k as usize][l as usize] - dist).abs());
                    // vector scalar multiplication
                    vec_factor = r.dot(&d_grad[color]);

                    if vec_factor.abs() <= 1e-2 {
                        vec_factor = 1e-6;
                    }

                    weight = (dist_factor * level_factor * vec_factor).abs();

                    let s_next_l = states[k as usize][(l + 1) as usize];
                    let s_prev_l = states[k as usize][(l - 1) as usize];

                    grad_i.x = match (s_next_l, s_prev_l) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (_, State::Unknown) => {
                            let p1 = pixel_ch(img, km, lp + 1, color);
                            let p2 = pixel_ch(img, km, lm, color);
                            p1 - p2
                        }
                        (State::Unknown, _) => {
                            let p1 = pixel_ch(img, km, lp, color);
                            let p2 = pixel_ch(img, km, lm - 1, color);
                            p1 - p2
                        }
                        (_, _) => {
                            let p1 = pixel_ch(img, km, lp + 1, color);
                            let p2 = pixel_ch(img, km, lm - 1, color);
                            2.0 * (p1 - p2)
                        }
                    };

                    grad_i.y = match (
                        states[(k + 1) as usize][l as usize],
                        states[(k - 1) as usize][l as usize],
                    ) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (_, State::Unknown) => {
                            let p1 = pixel_ch(img, kp + 1, lm, color);
                            let p2 = pixel_ch(img, km, lm, color);
                            p1 - p2
                        }
                        (State::Unknown, _) => {
                            let p1 = pixel_ch(img, kp, lm, color);
                            let p2 = pixel_ch(img, km - 1, lm, color);
                            p1 - p2
                        }
                        (_, _) => {
                            let p1 = pixel_ch(img, kp + 1, lm, color);
                            let p2 = pixel_ch(img, km - 1, lm, color);
                            2.0 * (p1 - p2)
                        }
                    };

                    let p = pixel_ch(img, km, lm, color);
                    i_a[color] += p * weight;
                    j_x[color] -= weight * grad_i.x * r.x;
                    j_y[color] -= weight * grad_i.y * r.y;
                    s[color] += weight;
                }
            }
        }
    }
    let mut sat: [f64; 3] = [0.0; 3];

    for color in 0..3 {
        sat[color] = i_a[color] / s[color]
            + (j_x[color] + j_y[color])
                / ((j_x[color] * j_x[color] + j_y[color] * j_y[color]).sqrt() + f64::EPSILON)
            + 0.5;
    }

    let rgb = Rgba([
        (sat[0].round().min(255.0).max(0.0)) as u8,
        (sat[1].round().min(255.0).max(0.0)) as u8,
        (sat[2].round().min(255.0).max(0.0)) as u8,
        255,
    ]);

    img.put_pixel((i - 1) as u32, (j - 1) as u32, rgb);
}

fn pixel_ch(img: &DynamicImage, i: u32, j: u32, ch: usize) -> f64 {
    img.get_pixel(i, j).channels()[ch] as f64
}

pub fn bertalmio_pixel(img: &mut DynamicImage, i: i32, j: i32, states: &States, radius: i32) {
    let (width, height) = (img.width() as i32, img.height() as i32);
    let mut ia: [f64; 3] = [0.0; 3];
    let mut sum = [f64::EPSILON; 3];
    // let (mut w, mut dst, mut dir): (f64, f64, f64);
    let mut grad_i: Point<f64> = Point::<f64>::new(0.0, 0.0);
    let mut r: Point<f64> = Point::<f64>::new(0.0, 0.0);

    for k in i - radius..=i + radius {
        for l in j - radius..=j + radius {
            let k_start = (k - 1 + (k == 1) as i32) as u32;
            let l_start = (l - 1 + (l == 1) as i32) as u32;
            let k_end = (k - 1 - (k == (width - 2)) as i32) as u32;
            let l_end = (l - 1 - (l == (height - 2)) as i32) as u32;

            if k > 0
                && l > 0
                && k < width - 1
                && l < height - 1
                && !states[k as usize][l as usize].is_unknown()
                && (l - j).pow(2) + (k - i).pow(2) <= radius.pow(2)
            {
                for ch in 0..3 {
                    r.x = (l - j) as f64;
                    r.y = (k - i) as f64;

                    let dist_factor = 1.0 / (r.len().powi(2) + 1.0);

                    let s_knext = states[(k + 1) as usize][l as usize];
                    let s_kprev = states[(k - 1) as usize][l as usize];

                    grad_i.x = match (s_knext, s_kprev) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (State::Unknown, _) => {
                            let p1 = pixel_ch(img, k_end + 1, l_start, ch);
                            let p2 = pixel_ch(img, k_end, l_start, ch);
                            2.0 * (p1 - p2).abs()
                        }
                        (_, State::Unknown) => {
                            let p1 = pixel_ch(img, k_end, l_start, ch);
                            let p2 = pixel_ch(img, k_start - 1, l_start, ch);
                            2.0 * (p1 - p2).abs()
                        }
                        (_, _) => {
                            let p1 = pixel_ch(img, k_end + 1, l_start, ch);
                            let p2 = pixel_ch(img, k_end, l_start, ch);
                            let p3 = pixel_ch(img, k_start - 1, l_start, ch);
                            (p1 - p2).abs() + (p2 - p3).abs()
                        }
                    };

                    let s_lnext = states[k as usize][(l + 1) as usize];
                    let s_lprev = states[k as usize][(l - 1) as usize];

                    grad_i.y = match (s_lnext, s_lprev) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (State::Unknown, _) => {
                            let p1 = pixel_ch(img, k_start, l_end + 1, ch);
                            let p2 = pixel_ch(img, k_start, l_start, ch);
                            2.0 * (p1 - p2).abs()
                        }
                        (_, State::Unknown) => {
                            let p1 = pixel_ch(img, k_start, l_start, ch);
                            let p2 = pixel_ch(img, k_start, l_start - 1, ch);
                            2.0 * (p1 - p2).abs()
                        }
                        (_, _) => {
                            let p1 = pixel_ch(img, k_start, l_end + 1, ch);
                            let p2 = pixel_ch(img, k_start, l_start, ch);
                            let p3 = pixel_ch(img, k_start, l_start - 1, ch);
                            (p1 - p2).abs() + (p2 - p3).abs()
                        }
                    };

                    grad_i.x = -grad_i.x;

                    let aux = r.dot(&grad_i);

                    let vec_factor = match aux.abs() <= 1e-2 {
                        true => 1e-6,
                        false => (aux / (r.len() * grad_i.len()).sqrt()).abs(),
                    };

                    let weight = dist_factor * vec_factor;
                    ia[ch] += weight * pixel_ch(img, k_start, l_start, ch);
                    sum[ch] += weight;
                }
            }
        }
    }

    let color: [u8; 4] = [
        ((ia[0] / sum[0]).round() as u8).min(255).max(0),
        ((ia[1] / sum[1]).round() as u8).min(255).max(0),
        ((ia[2] / sum[2]).round() as u8).min(255).max(0),
        255,
    ];
    img.put_pixel((i - 1) as u32, (j - 1) as u32, Rgba(color));
}

pub fn get_pixel_gradient(
    dist: &Distances,
    i: usize,
    j: usize,
    states: &States,
) -> [Point<f64>; 3] {
    let mut gradient: [Point<f64>; 3] = [Point::<f64>::new(0.0, 0.0); 3];

    let distance = dist[i][j];

    for channel in &mut gradient {
        channel.x = match (states[i][j + 1], states[i][j - 1]) {
            (State::Unknown, State::Unknown) => 0.0,
            (State::Unknown, _) => distance - dist[i][j - 1],
            (_, State::Unknown) => dist[i][j + 1] - distance,
            (_, _) => (dist[i][j + 1] - dist[i][j - 1]) * 0.5,
        };

        channel.y = match (states[i + 1][j], states[i - 1][j]) {
            (State::Unknown, State::Unknown) => 0.0,
            (State::Unknown, _) => distance - dist[i - 1][j],
            (_, State::Unknown) => dist[i + 1][j] - distance,
            (_, _) => (dist[i + 1][j] - dist[i - 1][j]) * 0.5,
        };
    }

    gradient
}

pub fn get_min_dist(nb: &Point<i32>, distances: &Distances, states: &States) -> f64 {
    let p1 = solve_eikonal(nb.x - 1, nb.y, nb.x, nb.y - 1, distances, states);
    let p2 = solve_eikonal(nb.x + 1, nb.y, nb.x, nb.y + 1, distances, states);
    let p3 = solve_eikonal(nb.x - 1, nb.y, nb.x, nb.y + 1, distances, states);
    let p4 = solve_eikonal(nb.x + 1, nb.y, nb.x, nb.y - 1, distances, states);

    p1.min(p2).min(p3).min(p4)
}

pub fn apply_telea2004(
    img: &DynamicImage,
    mask: &DynamicImage,
    radius: u8,
) -> Result<DynamicImage, String> {
    let (width, height) = (img.width() as i32, img.height() as i32);

    if width != mask.width() as i32 || height != mask.height() as i32 {
        return Err("Image and mask must have the same dimensions".to_string());
    }

    let mut result = img.clone();

    let (mut distances, mut states, mut heap) = get_initial_conditions(mask);
    get_telea_distances(mask, &mut distances, &states, radius);

    while let Some((_, p)) = heap.pop() {
        states[p.x as usize][p.y as usize] = State::Known;
        let neighbors = get_neighbors_4(p);

        for nb in neighbors {
            if nb.x < width - 1
                && nb.y < height - 1
                && nb.x > 0
                && nb.y > 0
                && states[nb.x as usize][nb.y as usize].is_unknown()
            {
                let min_dist = get_min_dist(&nb, &distances, &states);
                distances[nb.x as usize][nb.y as usize] = min_dist;
                telea_pixel(&mut result, nb.x, nb.y, &distances, &states, radius as i32);
                states[nb.x as usize][nb.y as usize] = State::Band;
                heap.push(min_dist, nb);
            }
        }
    }

    Ok(result)
}

pub fn apply_bertalmio2001(
    img: &DynamicImage,
    mask: &DynamicImage,
    radius: u8,
) -> Result<DynamicImage, String> {
    let (width, height) = (img.width() as i32, img.height() as i32);

    if width != mask.width() as i32 || height != mask.height() as i32 {
        return Err("Image and mask must have the same dimensions".to_string());
    }

    let mut result = img.clone();

    let (mut distances, mut states, mut heap) = get_initial_conditions(mask);

    while let Some((_, p)) = heap.pop() {
        states[p.x as usize][p.y as usize] = State::Known;
        let neighbors = get_neighbors_4(p);

        for nb in neighbors {
            if nb.x > 0
                && nb.y > 0
                && nb.x < width
                && nb.y < height
                && states[nb.x as usize][nb.y as usize].is_unknown()
            {
                let min_dist = get_min_dist(&nb, &distances, &states);

                distances[nb.x as usize][nb.y as usize] = min_dist;
                bertalmio_pixel(&mut result, nb.x, nb.y, &states, radius as i32);
                states[nb.x as usize][nb.y as usize] = State::Band;
                heap.push(min_dist, nb);
            }
        }
    }

    Ok(result)
}
