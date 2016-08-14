use rand::{Rand, random};

pub trait Matrix2d<T> {
    fn get_cols(&self) -> usize;
    fn get_rows(&self) -> usize;
    fn get_axis(&self, axis: usize) -> Vec<T>;
}

pub trait Matrix2dFill<T: Rand> {
    fn fill(n_cols: usize, n_rows: usize) -> Vec<Vec<T>>;
    fn fill_rng(n_cols: usize, n_rows: usize) -> Vec<Vec<T>>;
}

pub trait Matrix2dOps<T: Matrix2d<T>> {
    fn dot(&self, m: &T) -> T {
        let mut dot = Vec::new();

        for()

        return dot;
    }
}

impl<T: Rand> Matrix2d<T> for Vec<Vec<T>> {
    fn get_rows(&self) -> usize {
        self.len()
    }

    fn get_cols(&self) -> usize {
        match self.get(0) {
            Some(v) => v.len(),
            None => 0usize
        }
    }

    fn get_axis(&self, axis: usize) -> Vec<T> {

    }
}

impl<T: Rand> Matrix2dFill<T> for Vec<T> {
    fn fill(n_cols: usize, n_rows: usize) -> Vec<Vec<T>> {
        (0..n_rows)
            .map(move |_| {
                Vec::with_capacity(n_cols)
            })
            .collect::<Vec<Vec<T>>>()
    }
    fn fill_rng(n_cols: usize, n_rows: usize) -> Vec<Vec<T>>  {
        (0..n_rows)
            .map(move |_| {
                (0..n_cols)
                    .map(move |_| random::<T>())
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
    }
}
