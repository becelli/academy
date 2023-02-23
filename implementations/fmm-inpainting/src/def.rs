pub const DIST_MAX: f64 = 1e6f64;
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
    #[allow(dead_code)]
    pub fn is_known(&self) -> bool {
        matches!(self, State::Known)
    }
    pub fn is_change(&self) -> bool {
        matches!(self, State::Change)
    }
    #[allow(dead_code)]
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
    #[inline]
    pub fn new() -> Heap {
        Heap(BinaryHeap::new())
    }
    #[inline]
    pub fn push(&mut self, index: f64, item: Point<i32>) {
        self.0.push((Reverse(OrderedFloat(index)), item));
    }
    #[inline]
    pub fn pop(&mut self) -> Option<(Reverse<OrderedFloat<f64>>, Point<i32>)> {
        self.0.pop()
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
    #[inline]
    pub fn new(x: f64, y: f64) -> Point<f64> {
        Point { x, y }
    }
    #[inline]
    pub fn len(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn dot(&self, other: &Point<f64>) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

impl Point<i32> {
    #[inline]
    pub fn new(x: i32, y: i32) -> Point<i32> {
        Point { x, y }
    }
}
