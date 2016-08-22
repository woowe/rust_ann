extern crate rand;

mod matrix_utils;
use matrix_utils::*;

fn sum_vec(vec: Vec<f64>) -> f64 {
    vec.iter().fold(0f64, |acc, x| acc + x)
}

struct ForwardNeuralNet {
    topology: Vec<usize>,
    weights: Vec<Matrix2d>,
    activities: Vec<Matrix2d>,
    activations: Vec<Matrix2d>
}

impl ForwardNeuralNet {
    fn new(topology: Vec<usize>) -> Option<ForwardNeuralNet> {
        if topology.len() > 2 {
            return Some(ForwardNeuralNet {
                topology: topology.clone(),
                weights: (0..topology.len() - 1).map(|idx| {
                    Matrix2d::fill_rng(topology[idx], topology[idx + 1])
                }).collect::<Vec<Matrix2d>>(),
                activities: Vec::new(),
                activations: Vec::new()
            })
        }
        None
    }

    fn feed_forward(&mut self, input: &Matrix2d) -> Matrix2d {
        if None == self.activities.get(0) {
            self.activities.push(input.dot(&self.weights[0]).unwrap());
        } else {
            self.activities[0] = input.dot(&self.weights[0]).unwrap();
        }
        for (idx, weight) in self.weights.iter().enumerate().skip(1) {
            let activation  = self.activities[idx - 1].apply_fn(sigmoid);
            if None == self.activations.get(idx - 1) {
                self.activations.push(activation);
            } else {
                self.activations[idx - 1] = activation;
            }
            let activity    = self.activations[idx - 1].dot(weight).unwrap();
            if None == self.activities.get(idx) {
                self.activities.push(activity);
            }else {
                self.activities[idx] = activity
            }
        }
        return self.activities.last().unwrap().apply_fn(sigmoid);
    }

    fn cost_function(&mut self, input: &Matrix2d, output: &Matrix2d) -> f64 {
        let y_hat = self.feed_forward(&input);
        let cost = ((*output).clone() - y_hat).unwrap().apply_fn(|x| x * x);

        return 0.5f64 * sum_vec(cost.ravel());
    }

    fn cost_function_prime(&mut self, input: &Matrix2d, output: &Matrix2d) -> Vec<Matrix2d> {
        let mut deltas = Vec::new();
        let mut djdw = Vec::new();

        let y_hat = self.feed_forward(&input);
        let z_last = &self.activities.last().unwrap();
        let cost_matrix = -((*output).clone() - y_hat).unwrap();
        deltas.push(cost_matrix.mult(&z_last.apply_fn(sigmoid_prime)).unwrap());
        // A(2).T.dot(D3)
        // let a2 = &self.activations[0];
        // djdw.push(a2.transpose().dot(&deltas[0]).unwrap());
        // D3.dot(W(2).T) * sigmoid_prime(z2)
        // let w2 = &self.weights[1];
        // let z2 = &self.activities[0];
        // let delta = deltas[0].dot(&w2.transpose()).unwrap().mult(&z2.apply_fn(sigmoid_prime)).unwrap();
        // deltas.push(delta);

        for n in (0..(self.topology.len() - 2)) {
            let idx = (self.topology.len() - 2) - n;
            // println!("IDX: {}", idx);
            let a_t = &self.activations[idx - 1].transpose();
            let prev_delta = deltas.last().unwrap().clone();
            // println!("djdw equal?: {}", a_t.dot(&prev_delta).unwrap() == djdw[0]);
            djdw.push(a_t.dot(&prev_delta).unwrap());
            //
            let w_t = &self.weights[idx].transpose();
            // println!("WEIGHT IS EQUAL?: {}", *w_t == self.weights[1].transpose());
            let z_prime = &self.activities[idx - 1].apply_fn(sigmoid_prime);
            let delta = prev_delta.dot(w_t).unwrap().mult(z_prime).unwrap();
            deltas.push(delta);
        }

        // X.T.dot(D2)
        djdw.push(input.transpose().dot(&deltas.last().unwrap()).unwrap());
        djdw.reverse();
        return djdw;
    }

    // fn get_dimensions(&self) -> f64 {
    //     // self.topology.iter().fold(0f64, |acc, x| )
    // }

    fn get_params(&self) -> Vec<f64> {
        let mut params = self.weights[0].ravel();
        for w in self.weights.iter().skip(1) {
            params.extend(w.ravel());
        }
        params
    }

    fn set_params(&mut self, params: Vec<f64>) {
        // let W1_start = 0;
        // let input_layer_size = &self.topology[0];
        // let first_hidden_layer_size = &self.topology[1];
        // let last_hidden_layer_size = &self.topology[self.topology.len() - 2];
        // let output_layer_size = &self.topology[2];
        let mut W_end = 0;
        self.weights = Vec::new();
        // self.weights.push(params[W1_start..W_end].reshape(*input_layer_size, *first_hidden_layer_size).unwrap());
        for n in (0..self.topology.len() - 1) {
            let layer_size = &self.topology[n];
            let next_layer_size = &self.topology[n+1];
            let W_start = W_end;
            W_end = W_start + (layer_size * next_layer_size);
            self.weights.push(params[W_start..W_end].reshape(*layer_size, *next_layer_size).unwrap());
        }
        // let W2_end = W_end + (last_hidden_layer_size*output_layer_size);
        // self.weights.push(params[W_end..W2_end].reshape(*last_hidden_layer_size, *output_layer_size).unwrap());
    }

    fn compute_gradients(&mut self, input: &Matrix2d, output: &Matrix2d) -> Vec<f64> {
        let ds = self.cost_function_prime(&input, &output);
        let mut vec = Vec::new();
        for d in ds.iter() {
            // println!("DJDW: {:?}", &d);
            vec.extend(d.ravel());
        }
        return vec;
    }
}

fn compute_numerical_gradients(N: &mut ForwardNeuralNet, X: &Matrix2d, y: &Matrix2d) -> Vec<f64> {
    let params_init = N.get_params().to_matrix_2d().unwrap();
    let mut num_grad = Matrix2d::new(params_init.get_rows(), 1);
    let mut peturb = Matrix2d::new(params_init.get_rows(), 1);

    let e = 1e-4 as f64;

    for p in 0..params_init.get_rows() {
        peturb.get_matrix_mut()[p][0] = e;
        N.set_params(params_init.addition(&peturb).unwrap().ravel());
        let loss2 = N.cost_function(X, y);
        N.set_params(params_init.subtract(&peturb).unwrap().ravel());
        let loss1 = N.cost_function(X, y);

        num_grad.get_matrix_mut()[p][0] = (loss2 - loss1) / (2f64 * e);

        peturb.get_matrix_mut()[p][0] = 0f64;
    }
    N.set_params(params_init.ravel());

    return num_grad.ravel();
}

fn sigmoid(z: f64) -> f64 {
    1f64 / (1f64 + (-z as f64).exp())
}

fn sigmoid_prime(z: f64) -> f64 {
    (-z).exp() / ( (1f64 + (-z).exp()).powf(2f64) )
}

fn frobenius_norm(m: &Matrix2d) -> f64 {
    m.get_matrix().iter().fold(0f64, |acc, row| {
        acc + row.iter().fold(0f64, |acc, x| acc + (x * x))
    }).sqrt()
}

fn normalize_input_matrix(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let maxes_in_rows = input.iter()
        .map(move |row| {
            row.iter().fold(0f64, |acc, &x| {
                if acc < x.abs() {
                    return x.abs();
                }
                return acc;
            })
        })
        .collect::<Vec<f64>>();

    input.iter()
        .enumerate()
        .map(move |(idx, row)| row.iter().map(|n| n / maxes_in_rows[idx]).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>()
}

fn main() {
    let x = vec![vec![3f64, 5f64, 10f64], vec![5f64, 1f64, 2f64]];
    let y = vec![75f64, 82f64, 93f64];

    let norm_x = normalize_input_matrix(&x).to_matrix_2d().unwrap().transpose();
    let norm_y = y.iter().map(|out| out / 100f64).collect::<Vec<f64>>().to_matrix_2d().unwrap();
    // println!("OUTPUT NORM: {:?}", norm_y);

    let mut nn = ForwardNeuralNet::new(vec![2, 30, 8, 8, 8, 8, 8, 1]).unwrap();

    // println!("PREDICTIONS: {:?}", nn.feed_forward(&norm_x));
    // println!("PREDICTIONS: {:?}", nn.feed_forward(&norm_x));

    // for (i, d) in nn.cost_function_prime(&norm_x, &norm_y).iter().enumerate() {
    //     println!("DJDW({}): {:?}", i + 1, d);
    // }

    // println!("{:?}", nn.feed_forward(&norm_x));

    let cng = compute_numerical_gradients(&mut nn, &norm_x, &norm_y).to_matrix_2d().unwrap();
    let nn_cg = nn.compute_gradients(&norm_x, &norm_y).to_matrix_2d().unwrap();

    // println!("NUMGRAD: {:?}", cng.ravel());
    // println!("NN GRAD: {:?}", nn_cg.ravel());
    let grad_norm = frobenius_norm(&nn_cg.subtract(&cng).unwrap()) / frobenius_norm(&nn_cg.addition(&cng).unwrap());
    println!("FROBENIUS NORM: {:e}",  grad_norm);
    println!("norm < {:e}: {:?}", 10f64.powf(-8f64),  grad_norm < 10f64.powf(-8f64) as f64);
}
