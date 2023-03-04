pub const DIST_MAX: f64 = 1e6;
use ordered_float::OrderedFloat;
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Clone, Copy)]
pub enum State {
    Known,
    Band,
    Unknown,
    Change,
}

impl State {
    pub fn is_known(&self) -> bool {
        matches!(self, State::Known)
    }
    pub fn is_change(&self) -> bool {
        matches!(self, State::Change)
    }
    pub fn is_band(&self) -> bool {
        matches!(self, State::Band)
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self, State::Unknown)
    }
}

pub type Distances = Vec<Vec<f64>>;
pub type States = Vec<Vec<State>>;

#[derive(Clone, Debug)]
pub struct Heap(BinaryHeap<(Reverse<OrderedFloat<f64>>, Point<i32>)>);

impl Heap {
    pub fn new() -> Heap {
        Heap(BinaryHeap::new())
    }
    pub fn push(&mut self, index: f64, item: Point<i32>) {
        self.0.push((Reverse(OrderedFloat(index)), item));
    }
    pub fn pop(&mut self) -> Option<(Reverse<OrderedFloat<f64>>, Point<i32>)> {
        self.0.pop()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[inline]
pub fn get_mat<T: std::clone::Clone>(width: usize, height: usize, default: T) -> Vec<Vec<T>> {
    vec![vec![default; height]; width]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl Point<f64> {
    pub fn new(x: f64, y: f64) -> Point<f64> {
        Point { x, y }
    }
    pub fn len(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }
    pub fn dot(&self, other: &Point<f64>) -> f64 {
        self.x * other.x + self.y * other.y
    }
}


impl Point<i32> {
    pub fn new(x: i32, y: i32) -> Point<i32> {
        Point { x, y }
    }
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y) as f64
    }
}
