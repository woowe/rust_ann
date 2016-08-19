use rand::{Rand, random};

#[derive(Debug, PartialEq)]
pub struct Matrix2d {
    n_rows: usize,
    n_cols: usize,
    matrix: Vec<Vec<f64>>
}

pub enum AxisDir {
    Row,
    Column
}

impl Matrix2d {
    pub fn new(n_rows: usize, n_cols: usize) -> Matrix2d {
        Matrix2d {
            n_rows: n_rows,
            n_cols: n_cols,
            matrix: (0..n_rows).map(|_| {
                (0..n_cols).map(|_| 0f64).collect::<Vec<f64>>()
            }).collect::<Vec<Vec<f64>>>()
        }
    }

    pub fn from_vec(vec: &Vec<Vec<f64>>) -> Matrix2d {
        Matrix2d {
            n_rows: vec.len(),
            n_cols: vec[0].len(),
            matrix: vec.clone()
        }
    }

    pub fn fill_rng(n_rows: usize, n_cols: usize, ) -> Matrix2d {
        Matrix2d {
            n_rows: n_rows,
            n_cols: n_cols,
            matrix: (0..n_rows).map(|row| {
                (0..n_cols).map(|col| random::<f64>()).collect::<Vec<f64>>()
            }).collect::<Vec<Vec<f64>>>()
        }
    }

    pub fn transpose(&self) -> Matrix2d {
        Matrix2d {
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            matrix: (0..self.n_cols).map(|col| {
                (0..self.n_rows).map(|row| self.matrix[row][col]).collect::<Vec<f64>>()
            }).collect::<Vec<Vec<f64>>>()
        }
    }

    #[inline]
    pub fn get_cols(&self) -> usize { self.n_cols }

    #[inline]
    pub fn get_rows(&self) -> usize { self.n_rows }

    #[inline]
    pub fn get_matrix(&self) -> &Vec<Vec<f64>> { &self.matrix }

    pub fn get_matrix_mut(&mut self) -> &mut Vec<Vec<f64>> { &mut self.matrix }

    pub fn get_col(&self, n_col: usize) -> Option<Vec<f64>> {
        if n_col > self.matrix[0].len() - 1 {
            return None;
        }
        Some((0..self.matrix.len()).map( |row|
            self.matrix[row][n_col]
        ).collect::<Vec<f64>>())
    }

    pub fn get_row(&self, n_row: usize) -> Option<Vec<f64>> {
        if n_row > self.matrix.len() - 1 {
            return None;
        }
        Some((0..self.matrix[0].len()).map( |col|
            self.matrix[n_row][col]
        ).collect::<Vec<f64>>())
    }

    pub fn get_axis(&self, axis: usize, dir: AxisDir) -> Option<Vec<f64>> {
        match dir {
            AxisDir::Row => {
                self.get_row(axis)
            },
            AxisDir::Column => {
                self.get_col(axis)
            }
        }
    }

    pub fn dot(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if self.get_cols() == m.get_rows() {
            return Some(Matrix2d {
                n_rows: self.get_rows(),
                n_cols: m.get_cols(),
                matrix: (0..self.get_rows()).map(move |row| {
                    let _row = self.get_row(row).unwrap();
                    (0..m.get_cols()).map(|col| {
                        let _col = m.get_col(col).unwrap();

                        _row.iter().enumerate().fold(0f64, |acc, (i, x)| {
                            (x * _col[i]) + acc
                        })
                    }).collect::<Vec<f64>>()
                }).collect::<Vec<Vec<f64>>>()
            });
        }
        None
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

impl ToMatrix2d for Vec<f64> {
    fn to_matrix_2d(&self) -> Option<Matrix2d> {
        if self.len() > 0 {
            return Some(Matrix2d::from_vec(
                &self.iter().map(|i| vec![*i]).collect::<Vec<Vec<f64>>>()
            ));
        }
        None
    }
}

#[cfg(test)]
mod test {
    use std::option::Option;
    use matrix_utils::{ToMatrix2d, Matrix2d, AxisDir};

    #[test]
    fn to_matrix_2d_vec() {
        let vec = vec![1f64, 2f64, 3f64].to_matrix_2d().unwrap();
        let m = Matrix2d {
            n_rows: 3usize,
            n_cols: 1usize,
            matrix: vec![vec![1f64], vec![2f64], vec![3f64]]
        };
        assert!(vec == m);
    }

    #[test]
    fn to_matrix_2d_nested_vec() {
        let vec = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();
        let m = Matrix2d {
            n_rows: 3usize,
            n_cols: 1usize,
            matrix: vec![vec![1f64], vec![2f64], vec![3f64]]
        };
        assert!(vec == m);
    }

    #[test]
    fn to_matrix_2d_transpose() {
        let m = Matrix2d {
            n_rows: 3usize,
            n_cols: 1usize,
            matrix: vec![vec![1f64], vec![2f64], vec![3f64]]
        };

        let tm = Matrix2d {
            n_rows: 1usize,
            n_cols: 3usize,
            matrix: vec![vec![1f64, 2f64, 3f64]]
        };

        assert!(m.transpose() == tm);
    }

    #[test]
    fn get_matrix2d_col_size() {
        let m = Matrix2d {
            n_rows: 3usize,
            n_cols: 1usize,
            matrix: vec![vec![1f64], vec![2f64], vec![3f64]]
        };

        assert!(m.get_cols() == 1usize);
    }

    #[test]
    fn get_matrix2d_row_size() {
        let m = Matrix2d {
            n_rows: 3usize,
            n_cols: 1usize,
            matrix: vec![vec![1f64], vec![2f64], vec![3f64]]
        };

        assert!(m.get_rows() == 3usize);
    }

    #[test]
    fn get_matrix2d_col() {
        let m = Matrix2d {
            n_rows: 3usize,
            n_cols: 1usize,
            matrix: vec![vec![1f64], vec![2f64], vec![3f64]]
        };

        assert!(m.get_col(0).unwrap() == vec![1f64, 2f64, 3f64]);
    }

    #[test]
    fn get_matrix2d_row() {
        let m = Matrix2d {
            n_rows: 3usize,
            n_cols: 1usize,
            matrix: vec![vec![1f64], vec![2f64], vec![3f64]]
        };

        assert!(m.get_row(0).unwrap() == vec![1f64]);
    }

    #[test]
    fn get_matrix2d_axis() {
        let m = Matrix2d {
            n_rows: 3usize,
            n_cols: 1usize,
            matrix: vec![vec![1f64], vec![2f64], vec![3f64]]
        };

        assert!(m.get_axis(0, AxisDir::Column).unwrap() == vec![1f64, 2f64, 3f64]);
        assert!(m.get_axis(0, AxisDir::Row).unwrap() == vec![1f64]);
    }

    #[test]
    fn dot() {
        let m = Matrix2d {
            n_rows: 2usize,
            n_cols: 2usize,
            matrix: vec![vec![2f64, 2f64], vec![2f64, 2f64]]
        };

        let m1 = Matrix2d {
            n_rows: 2usize,
            n_cols: 2usize,
            matrix: vec![vec![2f64, 2f64], vec![2f64, 2f64]]
        };
        let m = vec![vec![5f64, 8f64, -4f64], vec![6f64, 9f64, -5f64], vec![4f64, 7f64, -2f64]].to_matrix_2d().unwrap();
        let m1 = vec![2f64, -3f64, 1f64].to_matrix_2d().unwrap();

        let pm = vec![-18f64, -20f64, -15f64].to_matrix_2d().unwrap();

        assert!(m.dot(&m1).unwrap() == pm);
    }


}
