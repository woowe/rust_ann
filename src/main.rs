extern crate rand;

mod matrix_utils;
use matrix_utils::*;

struct ForwardNeuralNet {
    input_layers: u32,
    output_layers: u32,
    hidden_layers: u32,
    w1: Vec<f64>,
    w2: Vec<f64>,
}

impl ForwardNeuralNet {
    fn new(input_layers: u32, output_layers: u32, hidden_layers: u32) -> ForwardNeuralNet {
        ForwardNeuralNet {
            input_layers: input_layers,
            output_layers: output_layers,
            hidden_layers: hidden_layers,
            w1: Vec::new(),
            w2: Vec::new(),
        }
    }

    fn feed_forward(&self, inputs: &[Vec<f64>]) {}
}

fn sigmoid(z: f64) -> f64 {
    1f64 / 1f64 + (-z as f64).exp()
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

    let norm_x = normalize_input_matrix(&x);
    let norm_y = y.iter().map(|out| out / 100f64).collect::<Vec<f64>>();

    let nn = ForwardNeuralNet::new(2, 3, 1);

    println!("NORMALIZED INPUT: {:?}", &norm_x[..]);
    println!("NORMALIZED OUTPUT: {:?}", &norm_y[..]);

    println!("RAND MARTIX FILL: {:?}", Vec::<f64>::fill_rng(2, 2));
}
