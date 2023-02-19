pub const DIST_MAX: f64 = 1e6f64;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
pub enum State {
    Change,
    Known,
    Band,
    Unknown,
}

impl State {
    #[allow(dead_code)]
    pub fn is_known(&self) -> bool {
        if let State::Known = self {
            return true;
        }
        false
    }
    pub fn is_change(&self) -> bool {
        if let State::Change = self {
            return true;
        }
        false
    }
    pub fn is_band(&self) -> bool {
        if let State::Band = self {
            return true;
        }
        false
    }
    pub fn is_unknown(&self) -> bool {
        if let State::Unknown = self {
            return true;
        }
        false
    }
}

pub type Distances = Vec<Vec<f64>>;
pub type States = Vec<Vec<State>>;
// pub type Heap = BinaryHeap<(OrderedFloat<f64>, Position)>;
#[derive(Clone)]
pub struct Heap(BinaryHeap<(OrderedFloat<f64>, Point<i32>)>);

impl Heap {
    pub fn new() -> Heap {
        Heap(BinaryHeap::new())
    }
    // pub fn push(&mut self, item: (f64, Position)) {
    pub fn push(&mut self, index: f64, item: Point<i32>) {
        self.0.push((OrderedFloat(index), item));
    }
    pub fn pop(&mut self) -> Option<(OrderedFloat<f64>, Point<i32>)> {
        self.0.pop()
    }
    // pub fn is_empty(&self) -> bool {
    //     self.0.is_empty()
    // }
}

pub fn get_mat<T: std::clone::Clone>(width: usize, height: usize, default: T) -> Vec<Vec<T>> {
    vec![vec![default; height]; width]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl Point<f64> {
    #[inline(always)]
    pub fn new(x: f64, y: f64) -> Point<f64> {
        Point { x, y }
    }

    // length of the vector (manhattan distance)
    pub fn len(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn dot(&self, other: &Point<f64>) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

impl Point<i32> {
    #[inline(always)]
    pub fn new(x: i32, y: i32) -> Point<i32> {
        Point { x, y }
    }
}
