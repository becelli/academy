const DIST_MAX: f64 = 1e6;
use ordered_float::OrderedFloat;
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Clone, Copy)]
pub enum State {
    Known,
    Band,
    Unknown,
    Change,
}

pub fn get_mat<T: std::clone::Clone>(width: usize, height: usize, default: T) -> Vec<Vec<T>> {
    vec![vec![default; height]; width]
}

impl State {
    pub fn is_change(&self) -> bool {
        matches!(self, State::Change)
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self, State::Unknown)
    }
}

pub struct Distances {
    pub values: Vec<Vec<f64>>,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone)]
pub struct States {
    pub values: Vec<Vec<State>>,
    pub width: usize,
    pub height: usize,
}

impl Distances {
    pub fn new(width: usize, height: usize) -> Distances {
        Distances {
            values: get_mat(width, height, DIST_MAX),
            width,
            height,
        }
    }
    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.values[x][y]
    }
    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        self.values[x][y] = value;
    }
}

impl States {
    pub fn new(width: usize, height: usize) -> States {
        States {
            values: get_mat(width, height, State::Known),
            width,
            height,
        }
    }
    pub fn get(&self, x: usize, y: usize) -> State {
        self.values[x][y]
    }
    pub fn set(&mut self, x: usize, y: usize, value: State) {
        self.values[x][y] = value;
    }
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut State {
        &mut self.values[x][y]
    }
}

#[derive(Clone)]
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
}
