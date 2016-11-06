use num_rust::Matrix2d;
use num_rust::ext::traits::ToMatrix2d;
use std::cmp;
use neural_net::NeuralNet;
use cost_function::CostFunction;

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

fn set_params<NN: NeuralNet>(nn: &mut NN, params: Vec<f64>) {
    let mut w_end = 0;
    let mut weights = Vec::new();

    for n in 0..nn.get_layers().len() - 1 {
        let layer_size = &nn.get_layers()[n].len();
        let next_layer_size = &nn.get_layers()[n+1].len();
        let w_start = w_end;
        w_end = w_start + (layer_size * next_layer_size);
        weights.push(params[w_start..w_end].reshape(*layer_size, *next_layer_size).unwrap());
    }

    let _ = nn.set_weights(weights);
}

fn get_params<NN: NeuralNet>(nn: &NN) -> Vec<f64> {
    let mut params = nn.get_weights()[0].ravel();
    for w in nn.get_weights().iter().skip(1) {
        params.extend(w.ravel());
    }
    params
}

fn compute_gradients<NN: NeuralNet, C: CostFunction>(nn: &mut NN, cost: &mut C,input: &Matrix2d, output: &Matrix2d) -> Vec<f64> {
    let pred = nn.predict(input).unwrap();
    let ds = cost.cost_prime(nn, &input, &output, &pred).unwrap();
    let mut vec = Vec::new();
    for d in ds.iter() {
        vec.extend(d.ravel());
    }
    return vec;
}

// numeric estimation of DJDW
fn compute_numerical_gradients<NN: NeuralNet, C: CostFunction>(nn: &mut NN, cost: &mut C, x: &Matrix2d, y: &Matrix2d) -> Vec<f64> {
    let params_init = get_params(nn).to_matrix_2d().unwrap();
    let mut num_grad = Matrix2d::new(params_init.get_rows(), 1);
    let mut peturb = Matrix2d::new(params_init.get_rows(), 1);

    let e = 1e-4 as f64;

    for p in 0..params_init.get_rows() {
        peturb.get_matrix_mut()[p] = e;
        set_params(nn, params_init.addition(&peturb).unwrap().ravel());
        let pred1 = nn.predict(x).unwrap();
        let loss2 = cost.cost(nn, y, &pred1).unwrap();
        set_params(nn, params_init.subtract(&peturb).unwrap().ravel());
        let pred2 = nn.predict(x).unwrap();
        let loss1 = cost.cost(nn, y, &pred2).unwrap();

        num_grad.get_matrix_mut()[p] = (loss2 - loss1) / (2. * e);

        peturb.get_matrix_mut()[p] = 0.;
    }
    set_params(nn, params_init.ravel());

    return num_grad.ravel();
}

pub fn check_gradient<NN: NeuralNet, C: CostFunction>(nn: &mut NN, cost: &mut C, x: &Matrix2d, y: &Matrix2d) -> f64 {
    let cng = compute_numerical_gradients(nn, cost, x, y).to_matrix_2d().unwrap();
    // println!("CNG: {:?}", cng);
    let nn_cg = compute_gradients(nn, cost, x, y).to_matrix_2d().unwrap();
    // println!("NN_CNG: {:?}", nn_cg);

    // check if I am computing my gradients correctly
    // ‖ nn_cg - cng ‖ / ‖ nn_cg + cng ‖ < 10e-8
    frobenius_norm(&nn_cg.subtract(&cng).unwrap()) / frobenius_norm(&nn_cg.addition(&cng).unwrap())
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
