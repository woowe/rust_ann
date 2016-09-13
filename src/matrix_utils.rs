use std::fmt;
use std::cmp::PartialEq;
use std::ops::{Neg, Sub};
use rand::random;
use matrixmultiply;

#[derive(Clone)]
pub struct Matrix2d {
    n_rows: usize,
    n_cols: usize,
    rs: usize,
    cs: usize,
    matrix: Vec<f64>
}

impl Matrix2d {
    pub fn new(n_rows: usize, n_cols: usize) -> Matrix2d {
        Matrix2d {
            n_rows: n_rows,
            n_cols: n_cols,
            rs: n_cols,
            cs: 1,
            matrix: (0..n_rows*n_cols).map(|_| 0.0).collect::<Vec<f64>>()
        }
    }

    pub fn from_vec(vec: &Vec<Vec<f64>>) -> Matrix2d {
        Matrix2d {
            n_rows: vec.len(),
            n_cols: vec[0].len(),
            rs: vec[0].len(),
            cs: 1,
            matrix: vec.iter().flat_map(|el| el.iter().cloned() ).collect::<Vec<f64>>(),
        }
    }

    pub fn fill_rng(n_rows: usize, n_cols: usize) -> Matrix2d {
        Matrix2d {
            n_rows: n_rows,
            n_cols: n_cols,
            rs: n_cols,
            cs: 1,
            matrix: (0..n_rows*n_cols)
                .map(|_| random::<f64>()).collect::<Vec<f64>>()
        }
    }

    pub fn get_col(&self, n_col: usize) -> Option<Vec<f64>> {
        if n_col > self.n_cols - 1 {
            return None;
        }
        Some((0..self.n_rows)
            .map(|row| self.matrix[row * self.rs + n_col * self.cs])
            .collect::<Vec<f64>>())
    }

    pub fn get_row(&self, n_row: usize) -> Option<Vec<f64>> {
        if n_row > self.n_rows - 1 {
            return None;
        }
        // Some(&self.get_matrix()[n_row * self.rs .. n_row * self.rs + self.n_cols])
        Some((0..self.n_cols)
            .map(|col| self.matrix[n_row * self.rs + col * self.cs])
            .collect::<Vec<f64>>())
    }

    pub fn transpose(&self) -> Matrix2d {
        Matrix2d {
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            rs: self.cs,
            cs: self.rs,
            matrix: self.matrix.clone()
        }
    }

    #[inline]
    pub fn get_cols(&self) -> usize {
        self.n_cols
    }

    #[inline]
    pub fn get_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    pub fn get_row_stride(&self) -> usize {
        self.rs
    }

    #[inline]
    pub fn get_col_stride(&self) -> usize {
        self.cs
    }

    #[inline]
    pub fn get_matrix(&self) -> &Vec<f64> {
        &self.matrix
    }

    pub fn get_matrix_mut(&mut self) -> &mut Vec<f64> {
        &mut self.matrix
    }

    pub fn dot(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if self.n_cols == m.get_rows() {
            let mut c = vec![0.; self.n_rows * m.get_cols()];
            // amazing magic happens here
            unsafe {
                matrixmultiply::dgemm(self.n_rows, self.n_cols, m.get_cols(),
                    1., self.get_matrix().as_ptr(), self.rs as isize, self.cs as isize,
                    m.get_matrix().as_ptr(), m.get_row_stride() as isize, m.get_col_stride() as isize,
                    0., c.as_mut_ptr(), m.get_cols() as isize, 1);
            }

            return Some(Matrix2d {
                n_rows: self.n_rows,
                n_cols: m.get_cols(),
                rs: m.get_cols(),
                cs: 1,
                matrix: c,
            });
        }
        None
    }

    pub fn apply_fn<F>(&self, f: F) -> Matrix2d
        where F: Fn(f64) -> f64
    {
        Matrix2d {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            rs: self.rs,
            cs: self.cs,
            matrix: self.get_matrix().iter().map(move |el| f(*el) ).collect::<Vec<f64>>()
        }
    }

    pub fn scale(&self, scalar: f64) -> Matrix2d {
        Matrix2d {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            rs: self.rs,
            cs: self.cs,
            matrix: self.get_matrix().iter().map(move |el| *el * scalar).collect::<Vec<f64>>()
        }
    }

    pub fn mult(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if  self.get_cols() == m.get_cols() &&
            self.get_rows() == m.get_rows() {
            let mut m_matrix_iter = m.get_matrix().iter();
            return Some(
                Matrix2d {
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                    rs: self.rs,
                    cs: self.cs,
                    matrix: self.matrix.iter().map(|el| el * m_matrix_iter.next().unwrap())
                .collect::<Vec<f64>>()
            });
        }
        None
    }

    pub fn subtract(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if  self.get_cols() == m.get_cols() &&
            self.get_rows() == m.get_rows() {
            let mut m_matrix_iter = m.get_matrix().iter();
            return Some(
                Matrix2d {
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                    rs: self.rs,
                    cs: self.cs,
                    matrix: self.matrix.iter().map(|el| el - m_matrix_iter.next().unwrap())
                        .collect::<Vec<f64>>()
                });
        }
        None
    }

    pub fn addition(&self, m: &Matrix2d) -> Option<Matrix2d> {
        if  self.get_cols() == m.get_cols() &&
            self.get_rows() == m.get_rows() {
            let mut m_matrix_iter = m.get_matrix().iter();
            return Some(
                Matrix2d {
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                    rs: self.rs,
                    cs: self.cs,
                    matrix: self.matrix.iter().map(|el| el + m_matrix_iter.next().unwrap())
                        .collect::<Vec<f64>>()
                });
        }
        None
    }

    pub fn ravel(&self) -> Vec<f64> {
        self.matrix.clone()
    }

    pub fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        // let vec = self.ravel();
        if self.matrix.len() / n_cols == n_rows {
            return Some(
                Matrix2d {
                    n_rows: n_rows,
                    n_cols: n_cols,
                    rs: n_cols,
                    cs: 1,
                    matrix: self.matrix.clone()
                }
            );
        }
        None
    }

    pub fn reshape_from_vec(vec: &Vec<f64>, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        if vec.len() / n_cols == n_rows {
            return Some(
                Matrix2d {
                    n_rows: n_rows,
                    n_cols: n_cols,
                    rs: n_cols,
                    cs: 1,
                    matrix: vec.clone()
                }
            );
        }
        None
    }

    pub fn normalize(&self) -> Matrix2d {
        let mut maxes = Vec::new();
        let mut matrix_clone = self.get_matrix().clone();
        for idx in 0..self.n_cols {
            maxes.push(self.get_col(idx).unwrap().iter()
                .fold(0f64, |acc, &x| {
                        if acc < x.abs() {
                            return x.abs();
                        }
                        return acc;
                    }));
        }

        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                matrix_clone[row * self.rs + col * self.cs] /= maxes[col];
            }
        }

        Matrix2d {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            rs: self.rs,
            cs: self.cs,
            matrix: matrix_clone.clone()
        }
    }
}

impl PartialEq for Matrix2d {
    fn eq(&self, other: &Matrix2d) -> bool {
        self.n_cols == other.get_cols() &&
        self.n_rows == other.get_rows() &&
        &self.matrix == other.get_matrix()
    }
}

impl fmt::Debug for Matrix2d {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output_string = format!("\nMatrix2d {{\n    n_rows: {},\n    n_cols: {},\n    rs: {}\n    cs: {}\n    matrix: ",
                                        self.n_rows,
                                        self.n_cols,
                                        self.rs,
                                        self.cs);

        let spacing = (0..self.get_cols())
            .map(|idx| {
                let col = self.get_col(idx).unwrap();

                col.iter().map(|x| x.to_string().len()).max().unwrap()
            })
            .collect::<Vec<usize>>();

        for idx in 0..self.n_rows {
            if idx > 0 {
                output_string = format!("{}            ", output_string);
            }

            output_string.push('[');
            output_string.push(' ');
            let row = self.get_row(idx).unwrap();
            for (i, x) in row.iter().enumerate() {
                let x_str = x.to_string();
                let tmp = format!("{}{}",
                                  x_str,
                                  (0..(spacing[i] - x_str.len())).map(|_| ' ').collect::<String>());
                output_string = format!("{}{}", output_string, tmp);
                if i < self.get_cols() - 1 {
                    output_string = format!("{}, ", output_string);
                }
            }

            output_string.push(' ');
            output_string.push(']');
            if idx < self.get_rows() - 1 {
                output_string.push('\n');
            }
        }

        write!(f, "{},\n }}\n", output_string)
    }
}

impl Neg for Matrix2d {
    type Output = Matrix2d;

    fn neg(self) -> Matrix2d {
        self.scale(-1f64)
    }
}

impl Sub for Matrix2d {
    type Output = Option<Matrix2d>;

    fn sub(self, _rhs: Matrix2d) -> Option<Matrix2d> {
        self.subtract(&_rhs)
    }
}

pub trait ToMatrix2d {
    fn to_matrix_2d(&self) -> Option<Matrix2d>;
    fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d>;
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

    fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        self.to_matrix_2d().expect("Provided vec is of len <= 0").reshape(n_rows, n_cols)
    }
}

impl ToMatrix2d for Vec<f64> {
    fn to_matrix_2d(&self) -> Option<Matrix2d> {
        if self.len() > 0 {
            return Some(Matrix2d::from_vec(&self.iter()
                .map(|i| vec![*i])
                .collect::<Vec<Vec<f64>>>()));
        }
        None
    }

    fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        Matrix2d::reshape_from_vec(&self, n_rows, n_cols)
    }
}

impl ToMatrix2d for [f64] {
    fn to_matrix_2d(&self) -> Option<Matrix2d> {
        if self.len() > 0 {
            return Some(Matrix2d::from_vec(&self.iter()
                .map(|i| vec![*i])
                .collect::<Vec<Vec<f64>>>()));
        }
        None
    }

    fn reshape(&self, n_rows: usize, n_cols: usize) -> Option<Matrix2d> {
        Matrix2d::reshape_from_vec(&self.to_vec(), n_rows, n_cols)
    }
}

#[cfg(test)]
mod test {
    use std::option::Option;
    use matrix_utils::{ToMatrix2d, Matrix2d};

    #[test]
    fn to_matrix_2d_vec() {
        let vec = vec![1f64, 2f64, 3f64].to_matrix_2d().expect("1d vec to matrix problem");
        let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().expect("2d vec to matrix problem");

        assert!(vec == m);
    }

    #[test]
    fn to_matrix_2d_transpose() {
        let m   = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();
        let tm  = vec![vec![1f64, 2f64, 3f64]].to_matrix_2d().unwrap();

        assert!(m.transpose() == tm);
    }

    #[test]
    fn get_matrix2d_col_size() {
        let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();

        assert!(m.get_cols() == 1usize);
    }

    #[test]
    fn get_matrix2d_row_size() {
        let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();

        assert!(m.get_rows() == 3usize);
    }

    #[test]
    fn get_matrix2d_col() {
        let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();

        assert!(m.get_col(0).unwrap() == vec![1f64, 2f64, 3f64]);
    }

    #[test]
    fn get_matrix2d_row() {
        let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();

        assert!(m.get_row(0).unwrap() == &[1f64]);
    }

    #[test]
    fn dot() {
        let m = vec![vec![5f64, 8f64, -4f64], vec![6f64, 9f64, -5f64], vec![4f64, 7f64, -2f64]].to_matrix_2d().unwrap();
        let m1 = vec![2f64, -3f64, 1f64].to_matrix_2d().unwrap();

        let pm = vec![-18f64, -20f64, -15f64].to_matrix_2d().unwrap();

        assert!(m.dot(&m1).unwrap() == pm);

        let m = vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0]].to_matrix_2d().unwrap(); // 3 x4
        let m1 = vec![vec![1.0, 2.0, 3.0]].to_matrix_2d().unwrap(); // 1 x 3

        let pm = vec![vec![0.0, 0.0, 0.0, 0.0]].to_matrix_2d().unwrap(); // 1 x 4
        assert!(m1.dot(&m).unwrap() == pm);
    }

    #[test]
    fn dot_transpose() {
        let m = vec![vec![1., 2.], vec![3., 4.], vec![5., 6.]].to_matrix_2d().unwrap();
        let tm = m.transpose();
        let dtm = vec![vec![5., 11., 17.], vec![11., 25., 39.], vec![17., 39., 61.]].to_matrix_2d().unwrap();

        assert!(m.dot(&tm).unwrap() == dtm);
    }

    #[test]
    fn apply_fn() {
        let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();
        let sm = vec![vec![1f64], vec![4f64], vec![9f64]].to_matrix_2d().unwrap();

        fn squared(x: f64) -> f64 {
            x * x
        }

        let c_squared = |x: f64| -> f64 { x * x };


        assert!(m.apply_fn(squared) == sm);
        assert!(m.apply_fn(c_squared) == sm);
    }

    #[test]
    fn scale() {
        let m = vec![vec![1f64], vec![2f64], vec![3f64]].to_matrix_2d().unwrap();
        let sm = vec![vec![2f64], vec![4f64], vec![6f64]].to_matrix_2d().unwrap();

        assert!(m.scale(2f64) == sm);
    }

    #[test]
    fn subtract() {
        let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].to_matrix_2d().unwrap();
        let m1 = vec![vec![0f64, -4f64, 3f64], vec![9f64, -4f64, -3f64]].to_matrix_2d().unwrap();
        let sm = vec![vec![-1f64, 6f64, -3f64], vec![-9f64, 7f64, 9f64]].to_matrix_2d().unwrap();


        assert!(m.subtract(&m1).unwrap() == sm);
    }

    #[test]
    fn add() {
        let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].to_matrix_2d().unwrap();
        let m1 = vec![vec![0f64, -4f64, 3f64], vec![9f64, -4f64, -3f64]].to_matrix_2d().unwrap();
        let sm = vec![vec![-1f64, -2f64, 3f64], vec![9f64, -1f64, 3f64]].to_matrix_2d().unwrap();

        assert!(m.addition(&m1).unwrap() == sm);
    }

    #[test]
    fn mult() {
        let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].to_matrix_2d().unwrap();
        let m1 = vec![vec![0f64, -4f64, 3f64], vec![9f64, -4f64, -3f64]].to_matrix_2d().unwrap();
        let sm = vec![vec![0f64, -8f64, 0f64], vec![0f64, -12f64, -18f64]].to_matrix_2d().unwrap();


        assert!(m.mult(&m1).unwrap() == sm);
    }

    #[test]
    fn ravel() {
        let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].to_matrix_2d().unwrap();
        let rvec = vec![-1f64, 2f64, 0f64, 0f64, 3f64, 6f64];

        assert!(m.ravel() == rvec);
    }

    #[test]
    fn reshape_2d() {
        let m = vec![vec![-1f64, 2f64, 0f64], vec![0f64, 3f64, 6f64]].reshape(3, 2).unwrap();
        let rm = vec![vec![-1f64, 2f64], vec![0f64, 0f64], vec![3f64, 6f64]].to_matrix_2d().unwrap();

        assert!(m == rm);
    }

    #[test]
    fn reshape_1d() {
        let m = vec![-1f64, 2f64, 0f64, 0f64, 3f64, 6f64].reshape(3, 2).unwrap();
        let rm = vec![vec![-1f64, 2f64], vec![0f64, 0f64], vec![3f64, 6f64]].to_matrix_2d().unwrap();

        assert!(m == rm);
    }
}
