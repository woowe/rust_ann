extern crate rand;

mod matrix_utils;
use matrix_utils::*;

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
                // w1: Matrix2d::fill_rng(2, 3),
                // w2: Matrix2d::fill_rng(3, 1),
            })
        }
        None
    }

    fn feed_forward(&mut self, input: &Matrix2d) -> Matrix2d {
        self.activities.push(input.dot(&self.weights[0]).unwrap());
        // println!("Z({}): {:?}", 2, input.dot(&self.weights[0]).unwrap());
        for (idx, weight) in self.weights.iter().enumerate().skip(1) {
            let activation  = self.activities.last().unwrap().apply_fn(sigmoid);
            // println!("A({}): {:?}", idx + 1, &activation);
            self.activations.push(activation);
            let activity    = self.activations.last().unwrap().dot(weight).unwrap();
            // println!("Z({}): {:?}", idx + 2, &activity);
            self.activities.push(activity);
        }

        return self.activities.last().unwrap().apply_fn(sigmoid);
    }

    fn cost_function_prime(&mut self, input: &Matrix2d, output: Matrix2d) -> Vec<Matrix2d> {
        let y_hat = self.feed_forward(&input);
        let z3 = &self.activities[1];
        let cost_matrix = (output - y_hat).unwrap().scale(-1f64);
        let delta3 = z3.apply_fn(sigmoid_prime).mult(&cost_matrix).unwrap();
        let djdw2 = self.activations.last().unwrap().transpose().dot(&delta3).unwrap();

        let delta2 = delta3.dot(&self.weights.last().unwrap().transpose()).unwrap().mult(&self.activities[0].apply_fn(sigmoid_prime)).unwrap();
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

    fn compute_gradients(&self, input: &Matrix2d, output: &Matrix2d) ->
}

fn sigmoid(z: f64) -> f64 {
    1f64 / (1f64 + (-z as f64).exp())
}

fn sigmoid_prime(z: f64) -> f64 {
    (-z).exp() / ( (1f64 + (-z).exp()).powf(2f64) )
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

    let mut nn = ForwardNeuralNet::new(vec![2, 3, 1]).unwrap();

    println!("PREDICTIONS: {:?}", nn.feed_forward(&norm_x));
    // println!("PREDICTIONS: {:?}", nn.feed_forward(&norm_x));

    for (i, d) in nn.cost_function_prime(&norm_x, norm_y).iter().enumerate() {
        println!("DJDW({}): {:?}", i + 1, d);
    }
}
