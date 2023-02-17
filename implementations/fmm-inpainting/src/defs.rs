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

pub type Position = (i32, i32);
pub type Distances = Vec<Vec<f64>>;
pub type States = Vec<Vec<State>>;
pub type Heap = BinaryHeap<(OrderedFloat<f64>, Position)>;
// new Heap
pub fn new_heap() -> Heap {
    BinaryHeap::new()
}

pub fn get_mat<T: std::clone::Clone>(width: usize, height: usize, default: T) -> Vec<Vec<T>> {
    vec![vec![default; height]; width]
}
