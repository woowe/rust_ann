use std::fs::File;
use std::io::Read;

extern crate rand;
extern crate matrixmultiply;
extern crate num_rust;

use num_rust::Matrix2d;
use num_rust::ext::traits::ToMatrix2d;

// mod matrix_utils;
// use matrix_utils::*;
//
mod utils;
use utils::*;

#[macro_use]
mod neural_net;
use neural_net::{NeuralNet, Sequential};

mod layer;
use layer::{Input, Dense};

mod cost_function;
use cost_function::MSE_Reg;

mod activation_function;
use activation_function::Sigmoid;

mod trainer;
use trainer::{Trainer, MiniBatchSGD};

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

    // create the neural net!
    let mut net = define_net!(
        Sequential[
            Input(4),
            Dense(5, Sigmoid),
            Dense(3, Sigmoid)
        ]
    );

    // define the cost function (MSE w/ l2 regularization, planning to lift the regularization out of the cost function implementation)
    // would look more like MSE::new(Regularization::L2(0.001))
    let mut cost_func = MSE_Reg::new(0.001);

    // make sure the gradients are being calculated correct
    println!("Calculating derivatives correct: {}", 10e-8 > check_gradient(&mut net, &mut cost_func, &norm_x, &norm_y));

    {
        // set up the trainer
        let mut trainer = print_try!(MiniBatchSGD::new(&mut net, &mut cost_func, 200_000, 5, 0.01));
        // optimize the weights!
        let _ = print_try!(trainer.optimize(&norm_x, &norm_y));
    }

    // now that we have optimized everything get the predictions of the opimized net!
    let pred_test = net.predict(&norm_test_x).unwrap();

    let mut num_right = 0.0;

    for i in 0..pred_test.get_rows() {
        let pred = pred_test.get_row(i).unwrap();
        let mut max_pred = 0.0;
        let mut max_pred_idx = 0;
        for (i, p) in pred.iter().enumerate() {
            if *p > max_pred {
                max_pred = *p;
                max_pred_idx = i;
            }
        }

        let actual_idx = norm_test_y.get_row(i).unwrap().iter().position(|x| *x == 1.0).unwrap();


        let nn_pred = match max_pred_idx {
            0 => "Iris-setosa",
            1 => "Iris-versicolor",
            2 => "Iris-virginica",
            _ => ""
        };

        let actual_pred = match actual_idx {
            0 => "Iris-setosa",
            1 => "Iris-versicolor",
            2 => "Iris-virginica",
            _ => ""
        };

        println!("ACTUAL: {}, PRED: {}", actual_pred, nn_pred);

        if actual_idx == max_pred_idx {
            num_right += 1.0;
        }
    }

    println!("{} OUT OF {} RIGHT", num_right, norm_test_y.get_rows());
    println!("ACCURACY: {}%", (num_right / (norm_test_y.get_rows() as f64)) * 100.0);
}
