use super::def::{Distances, Heap, Point, State, States};
use image::{self, DynamicImage, GenericImage, GenericImageView, Pixel, Rgba};


enum InpaintMethod {
    NavierStokes,
    Telea,
}

/// Computes the initial conditions for the Fast Marching Method algorithm.
///
/// Given a binary mask image, this function initializes the distances and states matrices and the heap data structure
/// used in the Fast Marching Method algorithm. The distances and states matrices are both 2D arrays with dimensions
/// `w_2` by `h_2`, where `w_2` and `h_2` are the width and height of the mask image plus two (padding for border pixels).
/// The heap data structure is used to store the points with their associated tentative distances.
///
/// # Arguments
///
/// * `mask` - A `DynamicImage` binary mask with white pixels representing the region of interest and black pixels
/// representing the background.
///
/// # Returns
///
/// A tuple containing the initialized distances and states matrices and heap data structure.
///
fn get_connectivity_4(pos: Point<i32>) -> [Point<i32>; 4] {
    [
        Point::<i32>::new(pos.x - 1, pos.y),
        Point::<i32>::new(pos.x, pos.y - 1),
        Point::<i32>::new(pos.x + 1, pos.y),
        Point::<i32>::new(pos.x, pos.y + 1),
    ]
}

fn get_initial_conditions(mask: &DynamicImage) -> (Distances, States, Heap) {
    // Determine the width and height of the mask image
    let (width, height) = (mask.dimensions().0, mask.dimensions().1);
    // Add padding to the width and height for the border pixels
    let (w_2, h_2) = ((width + 2) as usize, (height + 2) as usize);

    // Initialize the distances and states matrices and the heap data structure
    let mut distances: Distances = Distances::new(w_2, h_2);
    let mut states: States = States::new(w_2, h_2);
    let mut heap: Heap = Heap::new();

    // Iterate over all pixels in the mask image
    mask.pixels()
        .filter(|(i, j, pixel)| {
            // Ignore the border pixels and black pixels (background)
            pixel.channels()[0] != 0 && *i > 0 && *j > 0 && *i < width - 1 && *j < height - 1
        })
        .for_each(|(x, y, _)| {
            // Mark the corresponding pixel in the states matrix as unknown

            states.set((x + 1) as usize, (y + 1) as usize, State::Unknown);

            // Compute the 4-neighborhood of the current pixel
            let neighbors = get_connectivity_4(Point::<i32>::new((x + 1) as i32, (y + 1) as i32));

            // Iterate over the neighbors of the current pixel
            for nb in neighbors {
                let (nb_x, nb_y) = (nb.x as usize, nb.y as usize);

                let state = states.get_mut(nb_x, nb_y);

                // If the neighbor is not already known, mark it as a band point and initialize its distance to 0
                if !state.is_unknown() {
                    *state = State::Band;
                    distances.set(nb_x, nb_y, 0.0);
                    heap.push(0.0, nb);
                }
            }
        });

    // Return the initialized distances and states matrices and heap data structure
    (distances, states, heap)
}

fn telea_distances(distances: &mut Distances, states: &States, heap: &Heap, radius: u8) {
    let width = distances.width as i32;
    let height = distances.height as i32;

    let mut aux_heap = heap.clone();
    let mut aux_states = states.clone();

    aux_states.values.iter_mut().for_each(|row| {
        row.iter_mut().for_each(|state| match state {
            State::Unknown => *state = State::Known,
            State::Known => *state = State::Unknown,
            _ => (),
        })
    });

    let mut last_dist = 0.0;
    let double_radius = radius as f64 * 2.0;

    while let Some((_, point)) = aux_heap.pop() {
        let (x, y) = (point.x as usize, point.y as usize);
        aux_states.set(x, y, State::Change);

        let neighbors = get_connectivity_4(point);

        for nb in neighbors {
            if nb.x <= 0
                || nb.x >= width - 1
                || nb.y <= 0
                || nb.y >= height - 1
                || !aux_states.get(nb.x as usize, nb.y as usize).is_unknown()
            {
                continue;
            }

            last_dist = solve_fast_marching_method(&nb, distances, &aux_states);

            distances.set(nb.x as usize, nb.y as usize, last_dist);
            aux_states.set(nb.x as usize, nb.y as usize, State::Band);
            aux_heap.push(last_dist, nb);
        }

        if last_dist > double_radius {
            break;
        }
    }

    aux_states.values.iter().enumerate().for_each(|(x, row)| {
        row.iter().enumerate().for_each(|(y, state)| {
            if state.is_change() {
                distances.set(x, y, -distances.get(x, y));
            }
        })
    });
}

/// This function solves the eikonal equation for two points.
///
/// The eikonal equation determines the distance between two points in a field
/// given a known speed function. In this case, the distance is the Euclidean
/// distance and the speed function is the distance to the nearest boundary.
///
/// # Arguments
///
/// * `x1`: x-coordinate of the first point.
/// * `y1`: y-coordinate of the first point.
/// * `x2`: x-coordinate of the second point.
/// * `y2`: y-coordinate of the second point.
/// * `distances`: distance matrix.
/// * `states`: state matrix.
///
/// # Returns
///
/// The solution of the eikonal equation for the two points.
fn solve_eikonal_eq(
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    distances: &Distances,
    states: &States,
) -> f64 {
    let dist1 = distances.get(x1, y1);
    let dist2 = distances.get(x2, y2);
    let dist_min = dist1.min(dist2);
    let dist_sub = (dist1 - dist2).abs();

    let solution: f64 = match (states.get(x1, y1), states.get(x2, y2)) {
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

/// Computes the color for a pixel using the Telea algorithm.
///
/// # Arguments
///
/// * `img`: a mutable reference to the input image.
/// * `i`: the vertical coordinate of the pixel to be processed.
/// * `j`: the horizontal coordinate of the pixel to be processed.
/// * `distances`: a 2D array of distances used to weight the contributions of nearby pixels.
/// * `states`: a 2D array indicating whether pixels are known, unknown, or have been filled.
/// * `radius`: the radius of the search window for nearby pixels.
///
fn telea_inpaint_pixel(
    img: &mut DynamicImage,
    i: i32,
    j: i32,
    distances: &Distances,
    states: &States,
    radius: i32,
) {
    // Calculate image dimensions

    let (width, height) = (img.width() as i32, img.height() as i32);

    // Get the gradient of the current pixel

    let pixel_gradient = get_pixel_gradient(distances, i as usize, j as usize, states);

    // Initialize variables
    let mut pixel_offset: Point<f64> = Point::<f64>::new(0.0, 0.0);
    let mut color_gradient: Point<f64> = Point::<f64>::new(0.0, 0.0);
    let mut x_gradient = [0.0; 3];
    let mut y_gradient = [0.0; 3];
    let mut source_color = [0.0; 3];
    let mut smoothness = [f64::EPSILON; 3];
    let (mut weight, mut dist_factor, mut level_factor, mut vec_factor) = (0.0, 0.0, 0.0, 0.0);

    // Iterate over nearby pixels
    for k in i - radius..=i + radius {
        for l in j - radius..=j + radius {
            let k_start = (k - 1 + i32::from(k == 1)) as u32;
            let k_end = (k - 1 - i32::from(k == width - 2)) as u32;
            let l_start = (l - 1 + i32::from(l == 1)) as u32;
            let l_end = (l - 1 - i32::from(l == height - 2)) as u32;

            // If the pixel is within the image bounds and has a known state

            if k > 0
                && l > 0
                && k < width - 1
                && l < height - 1
                && !states.get(k as usize, l as usize).is_unknown()
                && ((k - i).pow(2) + (l - j).pow(2)) <= radius.pow(2)
            {
                // Iterate over each color channel
                ["R", "G", "B"].iter().enumerate().for_each(|(color, _)| {
                    pixel_offset.x = (j - l).into();
                    pixel_offset.y = (i - k).into();

                    let vec_len = pixel_offset.len();

                    dist_factor = 1.0 / (vec_len * vec_len.sqrt());

                    level_factor = 1.0
                        / (1.0
                            + (distances.get(k as usize, l as usize)
                                - distances.get(i as usize, j as usize))
                            .abs());

                    vec_factor = pixel_offset.dot(&pixel_gradient[color]);

                    if vec_factor.abs() <= 1e-2 {
                        vec_factor = 1e-6;
                    }

                    weight = (dist_factor * level_factor * vec_factor).abs();

                    let s_next_l = states.get(k as usize, (l + 1) as usize);
                    let s_prev_l = states.get(k as usize, (l - 1) as usize);

                    color_gradient.x = match (s_next_l, s_prev_l) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (_, State::Unknown) => {
                            let p1 = get_channel_value(img, k_start, l_end + 1, color);
                            let p2 = get_channel_value(img, k_start, l_start, color);
                            p1 - p2
                        }
                        (State::Unknown, _) => {
                            let p1 = get_channel_value(img, k_start, l_end, color);
                            let p2 = get_channel_value(img, k_start, l_start - 1, color);
                            p1 - p2
                        }
                        (_, _) => {
                            let p1 = get_channel_value(img, k_start, l_end + 1, color);
                            let p2 = get_channel_value(img, k_start, l_start - 1, color);
                            2.0 * (p1 - p2)
                        }
                    };

                    color_gradient.y = match (
                        states.get((k + 1) as usize, l as usize),
                        states.get((k - 1) as usize, l as usize),
                    ) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (_, State::Unknown) => {
                            let p1 = get_channel_value(img, k_end + 1, l_start, color);
                            let p2 = get_channel_value(img, k_start, l_start, color);
                            p1 - p2
                        }
                        (State::Unknown, _) => {
                            let p1 = get_channel_value(img, k_end, l_start, color);
                            let p2 = get_channel_value(img, k_start - 1, l_start, color);
                            p1 - p2
                        }
                        (_, _) => {
                            let p1 = get_channel_value(img, k_end + 1, l_start, color);
                            let p2 = get_channel_value(img, k_start - 1, l_start, color);
                            2.0 * (p1 - p2)
                        }
                    };

                    let channel = get_channel_value(img, k_start, l_start, color);
                    source_color[color] += channel * weight;
                    x_gradient[color] -= weight * color_gradient.x * pixel_offset.x;
                    y_gradient[color] -= weight * color_gradient.y * pixel_offset.y;
                    smoothness[color] += weight;
                })
            }
        }
    }

    let color = calculate_telea_color(source_color, smoothness, x_gradient, y_gradient);

    img.put_pixel((i - 1) as u32, (j - 1) as u32, color);
}

/// Calculates the minimum cost to reach a point using the fast marching method.
///
/// # Arguments
///
/// * `nb` - A reference to a `Point<i32>` struct representing the point to reach.
/// * `distances` - A reference to a `Distances` struct representing the distances to the starting point.
/// * `states` - A reference to a `States` struct representing the states of the points (obstacle or not).
///
/// # Returns
///
/// The minimum cost to reach the point represented by `nb`.
fn solve_fast_marching_method(nb: &Point<i32>, distances: &Distances, states: &States) -> f64 {
    let (x, y) = (nb.x as usize, nb.y as usize);
    let p1 = solve_eikonal_eq(x - 1, y, x, y - 1, distances, states);
    let p2 = solve_eikonal_eq(x + 1, y, x, y + 1, distances, states);
    let p3 = solve_eikonal_eq(x - 1, y, x, y + 1, distances, states);
    let p4 = solve_eikonal_eq(x + 1, y, x, y - 1, distances, states);

    p1.min(p2).min(p3).min(p4)
}

/// Calculates the color of a pixel in the output image using the method described in the Telea inpainting algorithm.
///
/// # Arguments:
///
/// * source_color: `[f64; 3]`: An array  of three f64 values representing the color of the corresponding pixel in the input image.
/// * smoothness: `[f64; 3]`: An array of three f64 values representing the smoothness of the corresponding pixel in the input image.
/// * x_gradient: `[f64; 3]`: An array of three f64 values representing the horizontal gradient of the corresponding pixel in the input image.
/// * y_gradient: `[f64; 3]`: An array of three f64 values representing the vertical gradient of the corresponding pixel in the input image.
///
/// # Returns:
///
/// Rgba<u8>: An Rgba struct representing the color of the pixel in the output image.
fn calculate_telea_color(
    source_color: [f64; 3],
    smoothness: [f64; 3],
    x_gradient: [f64; 3],
    y_gradient: [f64; 3],
) -> Rgba<u8> {
    let mut result_color: [u8; 4] = [255; 4];

    for channel in 0..3 {
        let intermediate_value = source_color[channel] / smoothness[channel]
            + (x_gradient[channel] + y_gradient[channel])
                / ((x_gradient[channel] * x_gradient[channel]
                    + y_gradient[channel] * y_gradient[channel])
                    .sqrt()
                    + f64::EPSILON)
            + 0.5;

        let clamped_value = intermediate_value.round().max(0.0).min(255.0) as u8;
        result_color[channel] = clamped_value;
    }

    Rgba(result_color)
}

/// Returns the value of a specific color channel of a pixel in an image.
///
/// # Arguments
///
/// * `img` - A reference to a `DynamicImage` representing the image.
/// * `i` - The x-coordinate of the pixel.
/// * `j` - The y-coordinate of the pixel.
/// * `ch` - The index of the color channel to retrieve.
///
/// # Returns
///
/// The value of the specified color channel of the pixel as a `f64`.
#[inline]
fn get_channel_value(img: &DynamicImage, i: u32, j: u32, ch: usize) -> f64 {
    img.get_pixel(i, j).channels()[ch] as f64
}

/// Computes the color for a pixel using the Bertalmio inpainting algorithm.
///
/// # Arguments
///
/// * `img`: a mutable reference to the input image.
/// * `i`: the vertical coordinate of the pixel to be processed.
/// * `j`: the horizontal coordinate of the pixel to be processed.
/// * `distances`: a 2D array of distances used to weight the contributions of nearby pixels.
/// * `states`: a 2D array indicating whether pixels are known, unknown, or have been filled.
/// * `radius`: the radius of the search window for nearby pixels.
///
#[allow(unused_variables)]
fn bertalmio_pixel(
    img: &mut DynamicImage,
    i: i32,
    j: i32,
    distances: &Distances,
    states: &States,
    radius: i32,
) {
    let (width, height) = (img.width() as i32, img.height() as i32);
    let mut source_color: [f64; 3] = [0.0; 3];
    let mut sum = [f64::EPSILON; 3];
    let mut color_gradient: Point<f64> = Point::<f64>::new(0.0, 0.0);
    let mut pixel_offset: Point<f64> = Point::<f64>::new(0.0, 0.0);

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
                && !states.get(k as usize, l as usize).is_unknown()
                && (l - j).pow(2) + (k - i).pow(2) <= radius.pow(2)
            {
                for ch in 0..3 {
                    pixel_offset.x = (l - j) as f64;
                    pixel_offset.y = (k - i) as f64;

                    let dist_factor = 1.0 / (pixel_offset.len().powi(2) + 1.0);

                    let s_knext = states.get((k + 1) as usize, l as usize);
                    let s_kprev = states.get((k - 1) as usize, l as usize);

                    color_gradient.x = match (s_knext, s_kprev) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (State::Unknown, _) => {
                            let p1 = get_channel_value(img, k_end + 1, l_start, ch);
                            let p2 = get_channel_value(img, k_end, l_start, ch);
                            2.0 * (p1 - p2).abs()
                        }
                        (_, State::Unknown) => {
                            let p1 = get_channel_value(img, k_end, l_start, ch);
                            let p2 = get_channel_value(img, k_start - 1, l_start, ch);
                            2.0 * (p1 - p2).abs()
                        }
                        (_, _) => {
                            let p1 = get_channel_value(img, k_end + 1, l_start, ch);
                            let p2 = get_channel_value(img, k_end, l_start, ch);
                            let p3 = get_channel_value(img, k_start - 1, l_start, ch);
                            (p1 - p2).abs() + (p2 - p3).abs()
                        }
                    };

                    let s_lnext = states.get(k as usize, (l + 1) as usize);
                    let s_lprev = states.get(k as usize, (l - 1) as usize);

                    color_gradient.y = match (s_lnext, s_lprev) {
                        (State::Unknown, State::Unknown) => 0.0,
                        (State::Unknown, _) => {
                            let p1 = get_channel_value(img, k_start, l_end + 1, ch);
                            let p2 = get_channel_value(img, k_start, l_start, ch);
                            2.0 * (p1 - p2).abs()
                        }
                        (_, State::Unknown) => {
                            let p1 = get_channel_value(img, k_start, l_start, ch);
                            let p2 = get_channel_value(img, k_start, l_start - 1, ch);
                            2.0 * (p1 - p2).abs()
                        }
                        (_, _) => {
                            let p1 = get_channel_value(img, k_start, l_end + 1, ch);
                            let p2 = get_channel_value(img, k_start, l_start, ch);
                            let p3 = get_channel_value(img, k_start, l_start - 1, ch);
                            (p1 - p2).abs() + (p2 - p3).abs()
                        }
                    };

                    color_gradient.x = -color_gradient.x;

                    let aux = pixel_offset.dot(&color_gradient);

                    let vec_factor = match aux.abs() <= 1e-2 {
                        true => 1e-6,
                        false => (aux / (pixel_offset.len() * color_gradient.len()).sqrt()).abs(),
                    };

                    let weight = dist_factor * vec_factor;
                    source_color[ch] += weight * get_channel_value(img, k_start, l_start, ch);
                    sum[ch] += weight;
                }
            }
        }
    }

    let color: [u8; 4] = [
        ((source_color[0] / sum[0]).round() as u8).min(255).max(0),
        ((source_color[1] / sum[1]).round() as u8).min(255).max(0),
        ((source_color[2] / sum[2]).round() as u8).min(255).max(0),
        255,
    ];
    img.put_pixel((i - 1) as u32, (j - 1) as u32, Rgba(color));
}

/// Calculates the pixel gradient at position (i,j) using distance and state information
///
/// # Arguments
///
/// * `dist` - A reference to a `Distances` object containing the distance information
/// * `i` - The row index of the pixel
/// * `j` - The column index of the pixel
/// * `states` - A reference to a `States` object containing the state information
///
/// # Returns
///
/// A `[Point<f64>; 3]` array containing the pixel gradient in x, y, and z directions
///
fn get_pixel_gradient(
    distances: &Distances,
    i: usize,
    j: usize,
    states: &States,
) -> [Point<f64>; 3] {
    let mut gradient: [Point<f64>; 3] = [Point::<f64>::new(0.0, 0.0); 3];

    // Calculate the gradient for each channel (x and y)
    for channel in &mut gradient {
        // Calculate the x-component of the gradient
        channel.x = match (states.get(i, j + 1), states.get(i, j - 1)) {
            // Unknown on both sides
            (State::Unknown, State::Unknown) => 0.0,
            // Unknown on left side only
            (State::Unknown, _) => distances.get(i, j) - distances.get(i, j - 1),
            // Unknown on right side only
            (_, State::Unknown) => distances.get(i, j + 1) - distances.get(i, j),
            // Known on both sides
            (_, _) => (distances.get(i, j + 1) - distances.get(i, j - 1)) * 0.5,
        };

        // Calculate the y-component of the gradient
        channel.y = match (states.get(i + 1, j), states.get(i - 1, j)) {
            // Unknown above and below
            (State::Unknown, State::Unknown) => 0.0,
            // Unknown above
            (State::Unknown, _) => distances.get(i, j) - distances.get(i - 1, j),
            // Unknown below
            (_, State::Unknown) => distances.get(i + 1, j) - distances.get(i, j),
            // Known above and below
            (_, _) => (distances.get(i + 1, j) - distances.get(i - 1, j)) * 0.5,
        };
    }

    gradient
}

fn inpaint(
    img: &DynamicImage,
    mask: &DynamicImage,
    radius: u8,
    method: InpaintMethod,
) -> Result<DynamicImage, String> {
    let (width, height) = (img.width() as i32, img.height() as i32);

    if width != mask.width() as i32 || height != mask.height() as i32 {
        return Err("Image and mask must have the same dimensions".to_string());
    }

    let mut result = img.clone();

    let (mut distances, mut states, mut heap) = get_initial_conditions(mask);

    if let InpaintMethod::Telea = method {
        telea_distances(&mut distances, &states, &heap, radius);
    }

    let inpaint_pixel = match method {
        InpaintMethod::NavierStokes => bertalmio_pixel,
        InpaintMethod::Telea => telea_inpaint_pixel,
    };

    while let Some((_, p)) = heap.pop() {
        states.set(p.x as usize, p.y as usize, State::Known);
        let neighbors = get_connectivity_4(p);

        for nb in neighbors {
            if nb.x < width - 1
                && nb.y < height - 1
                && nb.x > 0
                && nb.y > 0
                && states.get(nb.x as usize, nb.y as usize).is_unknown()
            {
                let min_dist = solve_fast_marching_method(&nb, &distances, &states);
                distances.set(nb.x as usize, nb.y as usize, min_dist);
                inpaint_pixel(&mut result, nb.x, nb.y, &distances, &states, radius as i32);
                states.set(nb.x as usize, nb.y as usize, State::Band);
                heap.push(min_dist, nb);
            }
        }
    }

    Ok(result)
}

/// Applies the Bertalmio 2001 inpainting algorithm to the input image `img`,
/// using the provided binary mask `mask` to identify the areas to be inpainted.
///
/// The `radius` parameter controls the size of the neighborhood used for the inpainting,
/// and should be a non-negative integer.
///
/// # Returns
///
/// a new `DynamicImage` with the inpainted result on success, or an error message
/// as a string if the inpainting fails.
pub fn bertalmio2001(
    img: &DynamicImage,
    mask: &DynamicImage,
    radius: u8,
) -> Result<DynamicImage, String> {
    inpaint(img, mask, radius, InpaintMethod::NavierStokes)
}

/// Applies the Telea 2004 inpainting algorithm to the input image `img`,
/// using the provided binary mask `mask` to identify the areas to be inpainted.
///
/// The `radius` parameter controls the size of the neighborhood used for the inpainting,
/// and should be a non-negative integer.
///
/// Returns a new `DynamicImage` with the inpainted result on success, or an error message
/// as a string if the inpainting fails.
pub fn telea2004(
    img: &DynamicImage,
    mask: &DynamicImage,
    radius: u8,
) -> Result<DynamicImage, String> {
    inpaint(img, mask, radius, InpaintMethod::Telea)
}
