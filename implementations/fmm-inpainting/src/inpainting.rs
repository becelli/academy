use super::def::{get_mat, Distances, Heap, Point, State, States, DIST_MAX};
use super::neighborhood::{get_connectivity_4, get_neighbors_n};
use image::{self, DynamicImage, GenericImage, GenericImageView, Pixel, Rgba};

fn get_initial_conditions(mask: &DynamicImage) -> (Distances, States, Heap) {
    let (width, height) = (mask.dimensions().0, mask.dimensions().1);

    let mut distances: Distances = get_mat((width + 2) as usize, (height + 2) as usize, DIST_MAX);
    let mut states: States = get_mat((width + 2) as usize, (height + 2) as usize, State::Known);
    let mut heap: Heap = Heap::new();

    mask.pixels()
        .filter(|(i, j, pixel)| {
            pixel.channels()[0] != 0 && *i > 0 && *j > 0 && *i < width - 1 && *j < height - 1
        })
        .for_each(|(x, y, _)| {
            states[(x + 1) as usize][(y + 1) as usize] = State::Unknown;

            let neighbors = get_connectivity_4(Point::<i32>::new((x + 1) as i32, (y + 1) as i32));

            for nb in neighbors {
                let state = &mut states[nb.x as usize][nb.y as usize];

                if !state.is_unknown() {
                    *state = State::Band;
                    distances[nb.x as usize][nb.y as usize] = 0.0;
                    heap.push(0.0, nb);
                }
            }
        });

    (distances, states, heap)
}

// TODO: Fix this function
fn telea_distances(mask: &DynamicImage, dist: &mut Distances, states: &States, radius: u8) {
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

        let neighbors = get_connectivity_4(pos);
        for nb in neighbors {
            if nb.x < width - 1
                && nb.y < height - 1
                && nb.x > 0
                && nb.y > 0
                && new_states[nb.x as usize][nb.y as usize].is_unknown()
            {
                let min = solve_fmm(&nb, dist, &new_states);

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

fn solve_eikonal(
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    dist: &Distances,
    states: &States,
) -> f64 {
    let dist1 = dist[x1][y1];
    let dist2 = dist[x2][y2];
    let dist_min = dist1.min(dist2);
    let dist_sub = (dist1 - dist2).abs();

    let solution: f64 = match (&states[x1][y1], &states[x2][y2]) {
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

fn telea_pixel(
    img: &mut DynamicImage,
    i: i32,
    j: i32,
    distances: &Distances,
    states: &States,
    radius: i32,
) {
    let (width, height) = (img.width() as i32, img.height() as i32);

    let pixel_gradient = get_pixel_gradient(distances, i as usize, j as usize, states);

    let mut r: Point<f64> = Point::<f64>::new(0.0, 0.0);
    let mut grad_i: Point<f64> = Point::<f64>::new(0.0, 0.0);
    let mut jx = [0.0; 3];
    let mut jy = [0.0; 3];
    let mut ia = [0.0; 3];
    let mut s = [f64::EPSILON; 3];
    let (mut weight, mut dist_factor, mut level_factor, mut vec_factor) = (0.0, 0.0, 0.0, 0.0);

    for k in i - radius..=i + radius {
        for l in j - radius..=j + radius {
            let k_start = (k - 1 + i32::from(k == 1)) as u32;
            let k_end = (k - 1 - i32::from(k == width - 2)) as u32;
            let l_start = (l - 1 + i32::from(l == 1)) as u32;
            let l_end = (l - 1 - i32::from(l == height - 2)) as u32;

            if k > 0
                && l > 0
                && k < width - 1
                && l < height - 1
                && !states[k as usize][l as usize].is_unknown()
                && ((k - i).pow(2) + (l - j).pow(2)) <= radius.pow(2)
            {
                // for color in 0..3 {
                ["R", "G", "B"].iter().enumerate().for_each(|(color, _)| {
                    r.x = (j - l).into();
                    r.y = (i - k).into();

                    let vec_len = r.len();

                    dist_factor = 1.0 / (vec_len * vec_len.sqrt());

                    level_factor = 1.0
                        / (1.0
                            + (distances[k as usize][l as usize]
                                - distances[i as usize][j as usize])
                                .abs());

                    vec_factor = r.dot(&pixel_gradient[color]);

                    if vec_factor.abs() <= 1e-2 {
                        vec_factor = 1e-6;
                    }

                    weight = (dist_factor * level_factor * vec_factor).abs();

                    let s_next_l = states[k as usize][(l + 1) as usize];
                    let s_prev_l = states[k as usize][(l - 1) as usize];

                    grad_i.x = match (s_next_l, s_prev_l) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (_, State::Unknown) => {
                            let p1 = pixel_ch(img, k_start, l_end + 1, color);
                            let p2 = pixel_ch(img, k_start, l_start, color);
                            p1 - p2
                        }
                        (State::Unknown, _) => {
                            let p1 = pixel_ch(img, k_start, l_end, color);
                            let p2 = pixel_ch(img, k_start, l_start - 1, color);
                            p1 - p2
                        }
                        (_, _) => {
                            let p1 = pixel_ch(img, k_start, l_end + 1, color);
                            let p2 = pixel_ch(img, k_start, l_start - 1, color);
                            2.0 * (p1 - p2)
                        }
                    };

                    grad_i.y = match (
                        states[(k + 1) as usize][l as usize],
                        states[(k - 1) as usize][l as usize],
                    ) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (_, State::Unknown) => {
                            let p1 = pixel_ch(img, k_end + 1, l_start, color);
                            let p2 = pixel_ch(img, k_start, l_start, color);
                            p1 - p2
                        }
                        (State::Unknown, _) => {
                            let p1 = pixel_ch(img, k_end, l_start, color);
                            let p2 = pixel_ch(img, k_start - 1, l_start, color);
                            p1 - p2
                        }
                        (_, _) => {
                            let p1 = pixel_ch(img, k_end + 1, l_start, color);
                            let p2 = pixel_ch(img, k_start - 1, l_start, color);
                            2.0 * (p1 - p2)
                        }
                    };

                    let channel = pixel_ch(img, k_start, l_start, color);
                    ia[color] += channel * weight;
                    jx[color] -= weight * grad_i.x * r.x;
                    jy[color] -= weight * grad_i.y * r.y;
                    s[color] += weight;
                })
            }
        }
    }

    let color = telea_color(ia, s, jx, jy);

    img.put_pixel((i - 1) as u32, (j - 1) as u32, color);
}

fn telea_color(ia: [f64; 3], s: [f64; 3], jx: [f64; 3], jy: [f64; 3]) -> Rgba<u8> {
    let mut color: [u8; 4] = [255; 4];

    ["R", "G", "B"].iter().enumerate().for_each(|(channel, _)| {
        let float = ia[channel] / s[channel]
            + (jx[channel] + jy[channel])
                / ((jx[channel] * jx[channel] + jy[channel] * jy[channel]).sqrt() + f64::EPSILON)
            + 0.5;

        color[channel] = float.round().max(0.0).min(255.0) as u8;
    });

    Rgba(color)
}

#[inline]
fn pixel_ch(img: &DynamicImage, i: u32, j: u32, ch: usize) -> f64 {
    img.get_pixel(i, j).channels()[ch] as f64
}

fn bertalmio_pixel(img: &mut DynamicImage, i: i32, j: i32, states: &States, radius: i32) {
    let (width, height) = (img.width() as i32, img.height() as i32);
    let mut ia: [f64; 3] = [0.0; 3];
    let mut sum = [f64::EPSILON; 3];
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

fn get_pixel_gradient(dist: &Distances, i: usize, j: usize, states: &States) -> [Point<f64>; 3] {
    let mut gradient: [Point<f64>; 3] = [Point::<f64>::new(0.0, 0.0); 3];

    for channel in &mut gradient {
        channel.x = match (states[i][j + 1], states[i][j - 1]) {
            (State::Unknown, State::Unknown) => 0.0,
            (State::Unknown, _) => dist[i][j] - dist[i][j - 1],
            (_, State::Unknown) => dist[i][j + 1] - dist[i][j],
            (_, _) => (dist[i][j + 1] - dist[i][j - 1]) * 0.5,
        };

        channel.y = match (states[i + 1][j], states[i - 1][j]) {
            (State::Unknown, State::Unknown) => 0.0,
            (State::Unknown, _) => dist[i][j] - dist[i - 1][j],
            (_, State::Unknown) => dist[i + 1][j] - dist[i][j],
            (_, _) => (dist[i + 1][j] - dist[i - 1][j]) * 0.5,
        };
    }

    gradient
}

fn solve_fmm(nb: &Point<i32>, distances: &Distances, states: &States) -> f64 {
    let (x, y) = (nb.x as usize, nb.y as usize);
    let p1 = solve_eikonal(x - 1, y, x, y - 1, distances, states);
    let p2 = solve_eikonal(x + 1, y, x, y + 1, distances, states);
    let p3 = solve_eikonal(x - 1, y, x, y + 1, distances, states);
    let p4 = solve_eikonal(x + 1, y, x, y - 1, distances, states);

    p1.min(p2).min(p3).min(p4)
}

pub fn telea2004(
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

    // telea_distances(mask, &mut distances, &states, radius);

    while let Some((_, p)) = heap.pop() {
        states[p.x as usize][p.y as usize] = State::Known;
        let neighbors = get_connectivity_4(p);

        for nb in neighbors {
            if nb.x < width - 1
                && nb.y < height - 1
                && nb.x > 0
                && nb.y > 0
                && states[nb.x as usize][nb.y as usize].is_unknown()
            {
                let min_dist = solve_fmm(&nb, &distances, &states);
                distances[nb.x as usize][nb.y as usize] = min_dist;
                telea_pixel(&mut result, nb.x, nb.y, &distances, &states, radius as i32);
                states[nb.x as usize][nb.y as usize] = State::Band;
                heap.push(min_dist, nb);
            }
        }
    }

    Ok(result)
}

pub fn bertalmio2001(
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

        for nb in get_connectivity_4(p) {
            if nb.x > 0
                && nb.y > 0
                && nb.x < width
                && nb.y < height
                && states[nb.x as usize][nb.y as usize].is_unknown()
            {
                let min_dist = solve_fmm(&nb, &distances, &states);

                distances[nb.x as usize][nb.y as usize] = min_dist;
                bertalmio_pixel(&mut result, nb.x, nb.y, &states, radius as i32);
                states[nb.x as usize][nb.y as usize] = State::Band;
                heap.push(min_dist, nb);
            }
        }
    }

    Ok(result)
}
