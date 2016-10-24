
use num_rust::Matrix2d;

use layer::Layer;

#[derive(Debug)]
pub enum NNetError {
    DescSize,
    LayerSize,
    ActivityError,
    ActivationError,
    CostError,
    PredictError,
    FitError,
    OptimizeError,
    MiniBatchSGDInitError,
    GradientError,
    GenericError
}

pub trait NeuralNet {
    type Net: NeuralNet;
    type L: Layer;

    fn new(desc: Vec<Self::L>) -> Result<Self::Net, NNetError>;
    fn predict(&mut self, input: &Matrix2d) -> Result<Matrix2d, NNetError>;

    fn set_weights(&mut self, n_weights: Vec<Matrix2d>) -> Result<(), NNetError>;
    fn get_weights(&self) -> &[Matrix2d];
    fn get_layers(&self) -> &[Self::L];
}

pub struct Sequential<L: Layer> {
    layers: Vec<L>,
    pub weights: Vec<Matrix2d>
}

impl<L: Layer> NeuralNet for Sequential<L> {
    type Net = Sequential<L>;
    type L = L;

    fn new(desc: Vec<L>) -> Result<Sequential<L>, NNetError> {
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
        let _ = try!(self.layers[1].set_activity(input, self.weights[0]));
        
        for (idx, weight) in self.weights.iter().enumerate().skip(1) {
            let mut prev_activation = self.layers[idx].get_activation();
            let _ = try!(self.layers[idx + 1].set_activity(&prev_activation, weight));
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

    fn get_layers(&self) -> &[Self::L] {
        &self.layers[..]
    }
}
