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
            let ws = vec![
                vec![vec![ 0.75260908,  0.06237675,  0.04749953],
                     vec![ 0.00078329,  0.4331744 ,  0.12382278]],
                vec![vec![ 0.42299088],
                     vec![ 0.14220764],
                     vec![ 0.70950682]]
            ];
            return Some(ForwardNeuralNet {
                topology: topology.clone(),
                weights: (0..topology.len() - 1).map(|idx| {
                    // Matrix2d::fill_rng(topology[idx], topology[idx + 1])
                    ws[idx].to_matrix_2d().unwrap()
                }).collect::<Vec<Matrix2d>>(),
                activities: Vec::new(),
                activations: Vec::new()
                // w1: Matrix2d::fill_rng(2, 3),
                // w2: Matrix2d::fill_rng(3, 1),
            })
        }
        None
    }

    fn feed_forward(&mut self, input: &Matrix2d) -> Matrix2d {
        self.activities.push(input.dot(&self.weights[0]).unwrap());
        // println!("{:?}", &self.weights[0]);
        for (idx, weight) in self.weights.iter().enumerate().skip(1) {
            let activation  = self.activities.last().unwrap().apply_fn(sigmoid);
            // println!("A({}): {:?}", idx + 1, &activation);
            self.activations.push(activation);
            let activity    = self.activations.last().unwrap().dot(weight).unwrap();
            // println!("Z({}): {:?}", idx + 2, &activity);
            self.activities.push(activity);
            // println!("{:?}", weight);
        }

        return self.activities.last().unwrap().apply_fn(sigmoid);
    }

    fn cost_function(&mut self, input: &Matrix2d, output: Matrix2d) -> f64 {
        let y_hat = self.feed_forward(&input);
        let cost = (output - y_hat).unwrap().apply_fn(|x| x * x);

        return 0.5f64 * sum_vec(cost.ravel());
    }

    fn cost_function_prime(&mut self, input: &Matrix2d, output: &Matrix2d) -> Vec<Matrix2d> {
        let y_hat = self.feed_forward(&input);
        let z3 = &self.activities[1];
        let cost_matrix = -((*output).clone() - y_hat).unwrap();
        // -(y - y_hat) * sigmoid_prime(z3)
        let delta3 = cost_matrix.mult(&z3.apply_fn(sigmoid_prime)).unwrap();
        // A(2).T.dot(D3)
        let a2 = &self.activations[0];
        let djdw2 = a2.transpose().dot(&delta3).unwrap();

        // D3.dot(W(2).T) * sigmoid_prime(z2)
        let w2 = self.weights.last().unwrap();
        let z2 = &self.activities[0];
        let delta2 = delta3.dot(&w2.transpose()).unwrap().mult(&z2.apply_fn(sigmoid_prime)).unwrap();
        // X.T.dot(D2)
        let djdw1 = input.transpose().dot(&delta2).unwrap();

        return vec![djdw1, djdw2];
    }

    fn get_params(&self) -> Vec<f64> {
        let mut params = self.weights[0].ravel();
        for w in self.weights.iter().skip(1) {
            params.extend(w.ravel());
        }
        params
    }

    fn set_params(&mut self, params: Vec<f64>) {
        let W1_start = 0;
        let input_layer_size = &self.topology[0];
        let hidden_layer_size = &self.topology[1];
        let output_layer_size = &self.topology[2];
        let W1_end = input_layer_size*hidden_layer_size;
        self.weights[0] = params[W1_start..W1_end].reshape(*input_layer_size, *hidden_layer_size).unwrap();
        let W2_end = W1_end + (hidden_layer_size*output_layer_size);
        self.weights[1] = params[W1_end..W2_end].reshape(*hidden_layer_size, *output_layer_size).unwrap();
    }

    fn compute_gradients(&mut self, input: &Matrix2d, output: &Matrix2d) -> Vec<f64> {
        let ds = self.cost_function_prime(&input, &output);
        let mut vec = Vec::new();
        for d in ds.iter() {
            println!("DJDW: {:?}", &d);
            vec.extend(d.ravel());
        }
        return vec;
    }
}

fn compute_numerical_gradients(N: &mut ForwardNeuralNet, X: &Matrix2d, y: &Matrix2d) -> Vec<f64> {
    let params_init = N.get_params().to_matrix_2d().unwrap();
    // println!("PARAMS INIT: {:?}", params_init);
    let mut num_grad = Matrix2d::new(params_init.get_rows(), 1);
    // println!("NUM GRAD: {:?}", num_grad);
    let mut peturb = Matrix2d::new(params_init.get_rows(), 1);
    // println!("PETURB: {:?}", peturb);

    let e = 1e-4 as f64;

    for p in 0..params_init.get_rows() {
        peturb.get_matrix_mut()[p][0] = e;
        N.set_params(params_init.addition(&peturb).unwrap().ravel());
        let loss2 = N.cost_function(X, (*y).clone());
        N.set_params(params_init.subtract(&peturb).unwrap().ravel());
        let loss1 = N.cost_function(X, (*y).clone());

        num_grad.get_matrix_mut()[p][0] = (loss2 - loss1) / (2f64 * e);

        peturb.get_matrix_mut()[p][0] = 0f64;
        // let loss2 = N.cost
    }

    // println!("NUM GRAD AFTER: {:?}", num_grad);
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

    let mut nn = ForwardNeuralNet::new(vec![2, 3, 1]).unwrap();

    // println!("PREDICTIONS: {:?}", nn.feed_forward(&norm_x));
    // println!("PREDICTIONS: {:?}", nn.feed_forward(&norm_x));

    // for (i, d) in nn.cost_function_prime(&norm_x, &norm_y).iter().enumerate() {
    //     println!("DJDW({}): {:?}", i + 1, d);
    // }

    let cng = compute_numerical_gradients(&mut nn, &norm_x, &norm_y).to_matrix_2d().unwrap();
    let nn_cg = nn.compute_gradients(&norm_x, &norm_y).to_matrix_2d().unwrap();

    println!("NUMGRAD: {:?}", cng.ravel());
    println!("NN GRAD: {:?}", nn_cg.ravel());
    let grad_norm = frobenius_norm(&nn_cg.subtract(&cng).unwrap()) / frobenius_norm(&nn_cg.addition(&cng).unwrap());
    println!("FROBENIUS NORM: {:e}",  grad_norm);
    println!("Did stuff correct?: {:?}",  grad_norm < 1e-8 as f64);
}
