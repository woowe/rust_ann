use matrix_utils::{ToMatrix2d, Matrix2d};
use std::cmp;

pub fn sum_vec(vec: &[f64]) -> f64 {
    let mut mc = vec.clone();
    unrolled_sum(&mut mc)
}

pub fn frobenius_norm(m: &Matrix2d) -> f64 {
    let mut mc = vec_bin_op(m.get_matrix(), m.get_matrix(), |x, y| x * y);
    unrolled_sum(&mut mc).sqrt()
}

// from rulinalg, originally from bluss / ndarray
pub fn unrolled_sum(mut xs: &[f64]) -> f64
{
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let mut sum = 0.;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0., 0., 0., 0., 0., 0., 0., 0.);
    while xs.len() >= 8 {
        p0 = p0 + xs[0].clone();
        p1 = p1 + xs[1].clone();
        p2 = p2 + xs[2].clone();
        p3 = p3 + xs[3].clone();
        p4 = p4 + xs[4].clone();
        p5 = p5 + xs[5].clone();
        p6 = p6 + xs[6].clone();
        p7 = p7 + xs[7].clone();

        xs = &xs[8..];
    }
    sum = sum.clone() + (p0 + p4);
    sum = sum.clone() + (p1 + p5);
    sum = sum.clone() + (p2 + p6);
    sum = sum.clone() + (p3 + p7);
    for elt in xs {
        sum = sum.clone() + elt.clone();
    }
    sum
}

// from rulinalg
pub fn vec_bin_op<F>(u: &[f64], v: &[f64], f: F) -> Vec<f64>
    where F: Fn(f64, f64) -> f64
{
    debug_assert_eq!(u.len(), v.len());
    let len = cmp::min(u.len(), v.len());

    let xs = &u[..len];
    let ys = &v[..len];

    let mut out_vec = Vec::with_capacity(len);
    unsafe {
        out_vec.set_len(len);
    }

    {
        let out_slice = &mut out_vec[..len];

        for i in 0..len {
            out_slice[i] = f(xs[i], ys[i]);
        }
    }

    out_vec
}


#[test]
fn sum_vec_test() {
    assert!(10.0 == sum_vec(&vec![1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn vec_bin_op_test() {
    let m = vec![1., 2., 3., 4.];
    assert!(vec![1., 4., 9., 16.] == vec_bin_op(&m, &m, |x, y| x * y))
}

#[test]
fn frobenius_norm_test() {
    // 4 + 9 + 16 25 29
    assert!((30f64).sqrt() == frobenius_norm(&vec![1.0, 2.0, 3.0, 4.0].to_matrix_2d().unwrap()));
}
