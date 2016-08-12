
struct Neuron<'a> {
    weight: f64,
    output_neurons: Vec<&'a Neuron<'a>>
}

impl<'a> Neuron<'a> {
    fn new(init_weight: f64) -> Neuron<'a> {
        Neuron {
            weight: init_weight,
            output_neurons: Vec::new()
        }
    }

    fn activate(inputs_weights: &[f64]) -> f64 {
        let activity = inputs_weights.iter().fold(0f64, |acc, x| acc + x);
        1f64 / 1f64 + (-activity as f64).exp()
    }
}

fn normalize_input_matrix(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let maxes_in_rows = input.iter().map(move |row| row.iter().fold(0f64, |acc, &x| {
        if acc < x.abs() {
            return x.abs();
        }
        return acc;
    }) ).collect::<Vec<f64>>();

    input.iter().enumerate().map(move |(idx, row)| {
        row.iter().map( |n| n / maxes_in_rows[idx]).collect::<Vec<f64>>()
    }).collect::< Vec<Vec<f64>> >()
}

fn main() {
    let x = vec![vec![3f64, 5f64, 10f64], vec![5f64, 1f64, 2f64]];
    let y = vec![75f64, 82f64, 93f64];

    let norm_x = normalize_input_matrix(&x);
    let norm_y = y.iter().map(|out| out / 100f64).collect::<Vec<f64>>();

    println!("NORMALIZED INPUT: {:?}", &norm_x[..]);
    println!("NORMALIZED OUTPUT: {:?}", &norm_y[..]);
}
