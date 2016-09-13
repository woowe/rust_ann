use matrix_utils::{ToMatrix2d, Matrix2d};

pub fn sum_vec(vec: &[f64]) -> f64 {
    vec.iter().fold(0f64, |acc, x| acc + x)
}

pub fn frobenius_norm(m: &Matrix2d) -> f64 {
    m.get_matrix().iter().fold(0f64, |acc, el| {
        acc + (el * el)
    }).sqrt()
}


#[test]
fn sum_vec_test() {
    assert!(10.0 == sum_vec(vec![1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn frobenius_norm_test() {
    // 4 + 9 + 16 25 29
    assert!((30f64).sqrt() == frobenius_norm(&vec![1.0, 2.0, 3.0, 4.0].to_matrix_2d().unwrap()));
}
