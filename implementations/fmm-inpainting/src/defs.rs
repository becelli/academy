pub const DIST_MAX: f64 = 1e6 as f64;

#[derive(Clone)]
pub enum State {
    Known,
    Band,
    Unknown,
}

impl State {
    pub fn is_known(&self) -> bool {
        if let State::Known = self {
            return true;
        }
        return false;
    }
    pub fn is_band(&self) -> bool {
        if let State::Band = self {
            return true;
        }
        return false;
    }
    pub fn is_unknown(&self) -> bool {
        if let State::Unknown = self {
            return true;
        }
        return false;
    }
}

pub type Position = (i32, i32);
pub type Distances = Vec<Vec<f64>>;
pub type States = Vec<Vec<State>>;
