
use num_rust::Matrix2d;

use layer::Layer;

#[derive(Debug)]
pub enum NNetError {
    DescSize,
    LayerSize,
    ActivityError,
    // ActivationError,
    CostError,
    PredictError,
    // FitError,
    OptimizeError,
    // MiniBatchSGDInitError,
    GradientError,
    GenericError
}

macro_rules! print_try {
    ($expr:expr) => (match $expr {
        Ok(val) => val,
        Err(err) => {
            print!("{:?}", err);
            panic!();
        }
    })
}

macro_rules! define_net {
    ($net:ident[ $ilayer:ident($insize:expr), $($layer:ident($size:expr, $activ:ident)),+]) => {
        print_try!($net::new(vec![
            Box::new(print_try!($ilayer::new($insize))),
            $(
                Box::new(print_try!($layer::new($size, $activ))),
            )+
        ]))
    };
}

pub trait NeuralNet {
    type Net: NeuralNet;
    // type L: Layer;

    fn new(desc: Vec<Box<Layer>>) -> Result<Self::Net, NNetError>;
    fn predict(&mut self, input: &Matrix2d) -> Result<Matrix2d, NNetError>;

    fn set_weights(&mut self, n_weights: Vec<Matrix2d>) -> Result<(), NNetError>;
    fn get_weights(&self) -> &[Matrix2d];
    fn get_layers(&self) -> &[Box<Layer>];
}

pub struct Sequential {
    layers: Vec<Box<Layer>>,
    pub weights: Vec<Matrix2d>
}

impl NeuralNet for Sequential {
    type Net = Sequential;
    // type L = L;

    fn new(desc: Vec<Box<Layer>>) -> Result<Sequential, NNetError> {
        if desc.len() > 2 {
            let desc_len = desc.len();
            let weights = (0..desc_len - 1).map(|idx| {
                Matrix2d::fill_rng(desc[idx].len(), desc[idx + 1].len())
            }).collect::<Vec<Matrix2d>>();
            return Ok(
                Sequential {
                    layers: desc,
                    weights: weights,
                }
            )
        }
        Err(NNetError::DescSize)
    }

    fn predict(&mut self, input: &Matrix2d) -> Result<Matrix2d, NNetError> {
        let _ = self.layers[0].set_input(&input);
        // println!("LAYER(0) activity: {:?}", self.layers[0].get_activity());
        let _ = try!(self.layers[1].set_activity(input, &self.weights[0]));
        // println!("LAYER(0) activity: {:?}", self.layers[1].get_activity());

        for (idx, weight) in self.weights.iter().enumerate().skip(1) {
            // println!("LAYER({}) activation: {:?}", idx, self.layers[1].get_activation());
            let prev_activation = self.layers[idx].get_activation();
            let _ = try!(self.layers[idx + 1].set_activity(&prev_activation, weight));
            // println!("LAYER({}) activity: {:?}", idx + 1, self.layers[1].get_activity());
        }

        match self.layers.last() {
            Some(l) => Ok(l.get_activation()),
            None => Err(NNetError::PredictError)
        }
    }

    fn set_weights(&mut self, n_weights: Vec<Matrix2d>) -> Result<(), NNetError> {
        if self.weights.len() == n_weights.len() {
            for (w, nw) in n_weights.iter().zip(self.weights.iter()) {
                if  w.get_rows() != nw.get_rows() &&
                    w.get_cols() != nw.get_cols() {
                    return Err(NNetError::GenericError)
                }
            }
        } else {
            return Err(NNetError::GenericError)
        }
        self.weights = n_weights.clone();
        Ok(())
    }

    fn get_weights(&self) -> &[Matrix2d] {
        &self.weights[..]
    }

    fn get_layers(&self) -> &[Box<Layer>] {
        &self.layers[..]
    }
}
