use super::def::Point;

pub fn get_neighbors_4(pos: Point<i32>) -> [Point<i32>; 4] {
    [
        Point::<i32>::new(pos.x - 1, pos.y),
        Point::<i32>::new(pos.x, pos.y - 1),
        Point::<i32>::new(pos.x + 1, pos.y),
        Point::<i32>::new(pos.x, pos.y + 1),
    ]
}

pub fn get_neighbors_n(pos: Point<i32>, radius: i32) -> Vec<Point<i32>> {
    // vec with capacity of 8 * radius
    let mut neighbors = Vec::with_capacity((8 * radius) as usize);

    for x in -radius..radius + 1 {
        for y in -radius..radius + 1 {
            if x == 0 && y == 0 {
                continue;
            }

            neighbors.push(Point::<i32>::new(pos.x + x, pos.y + y));
        }
    }

    neighbors
}
