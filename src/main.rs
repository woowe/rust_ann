use std::fs::File;
use std::io::Read;

extern crate rand;
extern crate matrixmultiply;
extern crate num_rust;

mod matrix_utils;
use matrix_utils::*;

mod utils;
use utils::*;

mod neural_net;
use neural_net::Sequential;

mod layer;
use layer::Dense;

mod cost_function;
use cost_function::MSE_Reg;

mod activation_function;
use activation_function::Sigmoid;

mod trainer;
use trainer::MiniBatchSGD;

struct NetData {
    train_data: (Matrix2d, Matrix2d),
    test_data: (Matrix2d, Matrix2d)
}

impl NetData {
    pub fn read_csv_file<F, P>(path: &str, f: F, pred: P) -> std::io::Result<NetData>
        where F: Fn(&str) -> (Vec<f64>, Vec<f64>), P: Fn(usize) -> bool
    {
        let mut file = try!(File::open(path));
        let mut train_data: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        let mut test_data: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        let mut contents = String::new();

        // read contents of csv file
        let _ = file.read_to_string(&mut contents);
        for (i, line) in contents.lines().enumerate() {
            // apply the parsing function on the current line
            let data = f(line);

            // if the pred function is true add to test else add the training
            if pred(i) {
                test_data.push(data);
            } else {
                train_data.push(data);
            }
        }

        // seperate out train and test to in & out matrixicies
        let train_data_in = train_data.iter().map(|&(ref in_data, _)| in_data.clone()).collect::<Vec<Vec<f64>>>().to_matrix_2d().unwrap();
        let train_data_out = train_data.iter().map(|&(_, ref out_data)| out_data.clone()).collect::<Vec<Vec<f64>>>().to_matrix_2d().unwrap();

        let test_data_in = test_data.iter().map(|&(ref in_data, _)| in_data.clone()).collect::<Vec<Vec<f64>>>().to_matrix_2d().unwrap();
        let test_data_out = test_data.iter().map(|&(_, ref out_data)| out_data.clone()).collect::<Vec<Vec<f64>>>().to_matrix_2d().unwrap();

        Ok(NetData {
            train_data: (train_data_in, train_data_out),
            test_data: (test_data_in, test_data_out)
        })
    }

    pub fn normalized_train_data(&self) -> (Matrix2d, Matrix2d) {
        (self.train_data.0.normalize(), self.train_data.1.normalize())
    }

    pub fn normalized_test_data(&self) -> (Matrix2d, Matrix2d) {
        (self.test_data.0.normalize(), self.test_data.1.normalize())
    }
}

// struct ForwardNeuralNet {
//     topology: Vec<usize>, // description of the net
//     weights: Vec<Matrix2d>, // weights
//     // deltaWeights: Vec<Matrix2d>, // gradient of weights
//     activities: Vec<Matrix2d>, // the activity of the neurons: sum(W_i * x_i) where x is the inputs to the neuron
//     activations: Vec<Matrix2d>, // the activation of the neurons: activation(sum(W_i * x_i))
//     lambda: f64 // the regularization rate
// }
//
// impl ForwardNeuralNet {
//     fn new(topology: Vec<usize>, lambda: f64) -> Option<ForwardNeuralNet> {
//         if topology.len() > 2 {
//             return Some(ForwardNeuralNet {
//                 topology: topology.clone(),
//                 // init weights with random values
//                 weights: (0..topology.len() - 1).map(|idx| {
//                     Matrix2d::fill_rng(topology[idx], topology[idx + 1])
//                 }).collect::<Vec<Matrix2d>>(),
//                 // // init gradient of weights with zeros
//                 // deltaWeights: (0..topology.len() - 1).map(|idx| {
//                 //     Matrix2d::new(topology[idx], topology[idx + 1])
//                 // }).collect::<Vec<Matrix2d>>(),
//                 activities: (0..topology.len() - 1).map(|_| {
//                     Matrix2d::new(1, 1)
//                 }).collect::<Vec<Matrix2d>>(),
//                 activations: (0..topology.len() - 2).map(|_| {
//                     Matrix2d::new(1, 1)
//                 }).collect::<Vec<Matrix2d>>(),
//                 lambda: lambda
//             })
//         }
//         None
//     }
//
//     fn feed_forward(&mut self, input: &Matrix2d) -> Matrix2d {
//         // compute activity of X ⊗ W(0)
//         self.activities[0] = input.dot(&self.weights[0]).expect("Dot product went wrong X*W(0)");
//
//         for (idx, weight) in self.weights.iter().enumerate().skip(1) {
//             // compute the activation of activation( activities(idx - 1) )
//             self.activations[idx - 1] = self.activities[idx - 1].apply_fn(sigmoid);
//             // compute activity of activation(idx - 1) ⊗ W(idx)
//             self.activities[idx] = self.activations[idx - 1].dot(weight).expect(&format!("Dot product went wrong: a({})*W({})", idx - 1, idx));
//         }
//
//         // compute the last activation activation( activities(last) )
//         return self.activities.last().unwrap().apply_fn(sigmoid);
//     }
//
//     fn cost_function(&mut self, output: &Matrix2d, pred: &Matrix2d) -> f64 {
//         let y_hat = pred;
//         let cost = ((*output).clone() - (*y_hat).clone()).expect("Subtract went wrong (Y - yhat)").apply_fn(|x| x * x);
//         let w_sum = self.weights.iter().fold(0f64, |acc, w| acc + sum_vec(&w.apply_fn(|x| x*x).get_matrix()[..]) );
//         // J = 1/(2m) ∑(y - ŷ)^2 + λ/2 * ∑(W(i))
//         return 0.5f64 * sum_vec(&cost.get_matrix()[..]) / (pred.get_rows() as f64) + ( (self.lambda/2.0)* w_sum );
//     }
//
//     fn cost_function_prime(&mut self, input: &Matrix2d, output: &Matrix2d, pred: &Matrix2d) -> Vec<Matrix2d> {
//         let mut deltas = Vec::new();
//         let mut djdw = Vec::new();
//
//         let y_hat = pred;
//         let z_last = &self.activities.last().unwrap();
//         let cost_matrix = -((*output).clone() - (*y_hat).clone()).expect("Subtract gone wrong (Y - yhat)");
//         // δ(last) = -(y - ŷ) * activation_prime(activities(last))
//         deltas.push(cost_matrix.mult(&z_last.apply_fn(sigmoid_prime)).expect("Mult gone wrong cost_matrix x sigmoid(z_last)"));
//
//         // just a compute of the reciprocal of y_hat.get_rows() so I don't recompute in the loop
//         let r_yhat: f64 = 1.0 / (y_hat.get_rows() as f64);
//
//         for n in 0..(self.topology.len() - 2) {
//             let idx = (self.topology.len() - 2) - n;
//             let a_t = &self.activations[idx - 1].transpose();
//             let prev_delta = deltas.last().unwrap().clone();
//             let l_w = self.weights[idx].scale(self.lambda);
//             // DJDW(idx) = activations(idx - 1).T ⊗ δ(idx - 1) * 1/m + W(idx) * λ
//             djdw.push(a_t.dot(&prev_delta).expect("Dot product gone wrong a_t * prev_delta").scale(r_yhat)
//                         .addition(&l_w).expect("Addition gone wrong a_t * prev_delta + lambda * W"));
//
//             let w_t = &self.weights[idx].transpose();
//             let z_prime = &self.activities[idx - 1].apply_fn(sigmoid_prime);
//
//             // δ(idx) = δ(idx - 1) ⊗ W(idx).T * activation_prime(activities(idx - 1))
//             let delta = prev_delta.dot(w_t).expect("Dot product gone wrong prev_delta * w_t").mult(z_prime).expect("Mult gone wrong (prev_delta * w_t) x z_prime");
//             deltas.push(delta);
//         }
//
//         // δ(0) = X.T ⊗ δ(1) * 1/m + W(0) * λ
//         djdw.push(input.transpose().dot(&deltas.last().unwrap()).expect("Dot gone wrong X_t * delta_last").scale(r_yhat)
//                     .addition(&self.weights[0].scale(self.lambda)).expect("Addition gone wrong X_t * delta_last + lambda * W(0)"));
//         djdw.reverse();
//         return djdw;
//     }
//
//     fn get_params(&self) -> Vec<f64> {
//         let mut params = self.weights[0].ravel();
//         for w in self.weights.iter().skip(1) {
//             params.extend(w.ravel());
//         }
//         params
//     }
//
//     fn set_params(&mut self, params: Vec<f64>) {
//         let mut w_end = 0;
//         self.weights = Vec::new();
//
//         for n in 0..self.topology.len() - 1 {
//             let layer_size = &self.topology[n];
//             let next_layer_size = &self.topology[n+1];
//             let w_start = w_end;
//             w_end = w_start + (layer_size * next_layer_size);
//             self.weights.push(params[w_start..w_end].reshape(*layer_size, *next_layer_size).unwrap());
//         }
//     }
//
//     fn compute_gradients(&mut self, input: &Matrix2d, output: &Matrix2d) -> Vec<f64> {
//         let pred = self.feed_forward(input);
//         let ds = self.cost_function_prime(&input, &output, &pred);
//         let mut vec = Vec::new();
//         for d in ds.iter() {
//             vec.extend(d.ravel());
//         }
//         return vec;
//     }
//
//     fn train(&mut self, input: &Matrix2d, output: &Matrix2d, batch_size: usize, alpha: f64, max_iters: usize) -> () {
//         let seed: &[_] = &[rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>()];
//         let shuffled_input = input.shuffle(seed).mini_batch(batch_size);
//         let shuffled_output = output.shuffle(seed).mini_batch(batch_size);
//
//         let mut training_data = shuffled_input.iter().map(|el| el.clone()).zip(shuffled_output).collect::<Vec<(Matrix2d,Matrix2d)>>();
//
//         for i in 0..max_iters {
//             for &(ref s_input, ref s_output) in training_data.iter() {
//                 let pred = self.feed_forward(&s_input);
//                 let djdws = self.cost_function_prime(&s_input, &s_output, &pred);
//                 let mut djdws_iter = djdws.iter();
//                 for weight in self.weights.iter_mut() {
//                     let djdw = (*djdws_iter.next().unwrap()).scale(alpha);
//                     let tmp_weight = ((*weight).clone() - djdw).unwrap();
//                     // gradient descent
//                     // W(i) = W(i) - alhpa * DJDW
//                     *weight = tmp_weight;
//                 }
//             }
//             let seed: &[_] = &[rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>()];
//             let _ = training_data.iter_mut().map(|&mut (ref mut si,ref mut so)| {
//                 *si = si.shuffle(seed);
//                 *so = so.shuffle(seed);
//             });
//         }
//     }
// }
//
// // numeric estimation of DJDW
// fn compute_numerical_gradients(nn: &mut ForwardNeuralNet, x: &Matrix2d, y: &Matrix2d) -> Vec<f64> {
//     let params_init = nn.get_params().to_matrix_2d().unwrap();
//     let mut num_grad = Matrix2d::new(params_init.get_rows(), 1);
//     let mut peturb = Matrix2d::new(params_init.get_rows(), 1);
//
//     let e = 1e-4 as f64;
//
//     for p in 0..params_init.get_rows() {
//         peturb.get_matrix_mut()[p] = e;
//         nn.set_params(params_init.addition(&peturb).unwrap().ravel());
//         let pred1 = nn.feed_forward(x);
//         let loss2 = nn.cost_function(y, &pred1);
//         nn.set_params(params_init.subtract(&peturb).unwrap().ravel());
//         let pred2 = nn.feed_forward(x);
//         let loss1 = nn.cost_function(y, &pred2);
//
//         num_grad.get_matrix_mut()[p] = (loss2 - loss1) / (2f64 * e);
//
//         peturb.get_matrix_mut()[p] = 0f64;
//     }
//     nn.set_params(params_init.ravel());
//
//     return num_grad.ravel();
// }
//
// fn sigmoid(z: f64) -> f64 {
//     1f64 / (1f64 + (-z).exp())
// }
//
// fn sigmoid_prime(z: f64) -> f64 {
//     sigmoid(z) * (1. - sigmoid(z))
//     // (-z).exp() / ( (1f64 + (-z).exp()).powf(2f64) )
// }

fn main() {
    let net_data = NetData::read_csv_file("./data_sets/iris.txt",
                        |line| {
                            let vals = line.split(',').collect::<Vec<&str>>();
                            let inputs: Vec<f64> = vals[..vals.len() - 1].iter().map(|x| x.parse().unwrap()).collect::<Vec<f64>>();
                            let last_val = match vals[vals.len() - 1]{
                                "Iris-setosa" => vec![1.0, 0.0, 0.0],
                                "Iris-versicolor" => vec![0.0, 1.0, 0.0],
                                "Iris-virginica" => vec![0.0, 0.0, 1.0],
                                _ => vec![0.0, 0.0, 0.0]
                            };
                            let outputs: Vec<f64> = last_val;
                            (inputs, outputs)
                        }, |idx| {
                            idx % 2 == 0
                        }).unwrap();

    let (norm_x, norm_y) = net_data.normalized_train_data();
    let (norm_test_x, norm_test_y) = net_data.normalized_test_data();

    // let mut nn = ForwardNeuralNet::new(vec![4, 5, 3], 0.0001).unwrap();

    // let cng = compute_numerical_gradients(&mut nn, &norm_x, &norm_y).to_matrix_2d().unwrap();
    // let nn_cg = nn.compute_gradients(&norm_x, &norm_y).to_matrix_2d().unwrap();

    // check if I am computing my gradients correctly
    // ‖ nn_cg - cng ‖ / ‖ nn_cg + cng ‖ < 10e-8
    // let grad_norm = frobenius_norm(&nn_cg.subtract(&cng).unwrap()) / frobenius_norm(&nn_cg.addition(&cng).unwrap());
    // assert!(grad_norm < 1e-8);
    // println!("FROBENIUS NORM: {:e}",  grad_norm);
    // println!("norm < {:e}: {:?}", 10f64.powf(-8f64),  grad_norm < 10f64.powf(-8f64) as f64);

    // println!("ACTUAL INPUT: {:?}", &norm_x);
    // println!("ACTUAL OUTPUT: {:?}", &norm_y);



    // nn.train(&norm_x, &norm_y, 15, 0.5, 10000);
    //
    // let pred_test = nn.feed_forward(&norm_test_x);
    //
    // let mut num_right = 0.0;
    //
    // for i in 0..pred_test.get_rows() {
    //     let pred = pred_test.get_row(i).unwrap();
    //     let mut max_pred = 0.0;
    //     let mut max_pred_idx = 0;
    //     for (i, p) in pred.iter().enumerate() {
    //         if *p > max_pred {
    //             max_pred = *p;
    //             max_pred_idx = i;
    //         }
    //     }
    //
    //     let actual_idx = norm_test_y.get_row(i).unwrap().iter().position(|x| *x == 1.0).unwrap();
    //
    //
    //     let nn_pred = match max_pred_idx {
    //         0 => "Iris-setosa",
    //         1 => "Iris-versicolor",
    //         2 => "Iris-virginica",
    //         _ => ""
    //     };
    //
    //     let actual_pred = match actual_idx {
    //         0 => "Iris-setosa",
    //         1 => "Iris-versicolor",
    //         2 => "Iris-virginica",
    //         _ => ""
    //     };
    //
    //     println!("ACTUAL: {}, PRED: {}, %{} CONFIDENCE", actual_pred, nn_pred, ((max_pred) * 100.0).round());
    //
    //     if actual_idx == max_pred_idx {
    //         num_right += 1.0;
    //     }
    // }
    //
    // println!("{} OUT OF {} RIGHT", num_right, norm_test_y.get_rows());
    // println!("ACCURACY: {}%", (num_right / (norm_test_y.get_rows() as f64)) * 100.0);
}
