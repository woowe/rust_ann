use rand::{Rand, random};

#[derive(Debug)]
pub struct Matrix2d {
    n_rows: usize,
    n_cols: usize,
    martix: Vec<Vec<f64>>
}

impl Matrix2d {
    pub fn new(n_rows: usize, n_cols: usize) -> Matrix2d {
        Matrix2d {
            n_rows: n_rows,
            n_cols: n_cols,
            martix: (0..n_rows).map(|_| {
                (0..n_cols).map(|_| 0f64).collect::<Vec<f64>>()
            }).collect::<Vec<Vec<f64>>>()
        }
    }

    pub fn from_vec(vec: &Vec<Vec<f64>>) -> Matrix2d {
        Matrix2d {
            n_rows: vec.len(),
            n_cols: vec[0].len(),
            martix: vec.clone()
        }
    }

    pub fn fill_rng(n_rows: usize, n_cols: usize, ) -> Matrix2d {
        Matrix2d {
            n_rows: n_rows,
            n_cols: n_cols,
            martix: (0..n_rows).map(|row| {
                (0..n_cols).map(|col| random::<f64>()).collect::<Vec<f64>>()
            }).collect::<Vec<Vec<f64>>>()
        }
    }

    pub fn transpose(&self) -> Matrix2d {
        Matrix2d {
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            martix: (0..self.n_cols).map(|col| {
                (0..self.n_rows).map(|row| self.martix[row][col]).collect::<Vec<f64>>()
            }).collect::<Vec<Vec<f64>>>()
        }
    }
}

pub trait ToMatrix2d {
    fn to_matrix_2d(&self) -> Option<Matrix2d>;
}

impl ToMatrix2d for Vec<Vec<f64>> {
    fn to_matrix_2d(&self) -> Option<Matrix2d> {
        if self.len() > 0 {
            let col_len = self[0].len();
            for row in self.iter() {
                if col_len != row.len() {
                    return None;
                }
            }
            return Some(Matrix2d::from_vec(self));
        }
        None
    }
}
