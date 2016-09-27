
use num_rust::Matrix2d;

use layer::Layer;
use cost_function::CostFunction;
use trainer::Trainer;

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
    type C: CostFunction;
    type T: Trainer;

    fn new(&self, desc: Vec<Self::L>, cost: Self::C, trainer: Self::T) -> Result<Self::Net, NNetError>;
    fn predict(&mut self, input: &Matrix2d) -> Result<Matrix2d, NNetError>;
    fn fit(&mut self,  input: &Matrix2d, output: &Matrix2d) -> Result<(), NNetError>;

    fn get_weights(&self) -> &[Matrix2d];
    fn get_layers(&self) -> &[Self::L];
    fn get_cost(&self) -> &Self::C;
    fn get_trainer(&self) -> &Self::T;
}

pub struct Sequential<L: Layer, CF: CostFunction, T: Trainer> {
    layers: Vec<L>,
    weights: Vec<Matrix2d>,
    cost: CF,
    trainer: T
}

impl<L: Layer, CF: CostFunction, T: Trainer> NeuralNet for Sequential<L, CF, T> {
    type Net = Sequential<L, CF, T>;
    type L = L;
    type C = CF;
    type T = T;

    fn new(&self, desc: Vec<L>, cost: CF, trainer: T) -> Result<Sequential<L, CF, T>, NNetError> {
        if desc.len() > 2 {
            return Ok(
                Sequential {
                    layers: desc,
                    weights: (0..desc.len() - 1).map(|idx| {
                        Matrix2d::fill_rng(desc[idx].len(), desc[idx + 1].len())
                    }).collect::<Vec<Matrix2d>>(),
                    cost: cost,
                    trainer: trainer
                }
            )
        }
        return Err(NNetError::DescSize)
    }

    fn predict(&mut self, input: &Matrix2d) -> Result<Matrix2d, NNetError> {
        let _ = try!(self.layers[0].set_activity(&input, &self.weights[0]));

        for (idx, weight) in self.weights.iter().skip(1).enumerate() {
            let prev_activation = self.layers[idx - 1].get_activation();
            let _ = try!(self.layers[idx].set_activity(&input, &self.weights[idx]));
        }

        match self.layers.last() {
            Some(v) => Ok(v.get_activation()),
            None => Err(NNetError::PredictError)
        }
    }

    fn fit(&mut self,  input: &Matrix2d, output: &Matrix2d) -> Result<(), NNetError> {
        let updated_weights  = try!(self.trainer.optimize(input, output));
        self.weights = updated_weights;
        Ok(())
    }

    fn get_weights(&self) -> &[Matrix2d] {
        &self.weights[..]
    }

    fn get_layers(&self) -> &[Self::L] {
        &self.layers[..]
    }

    fn get_cost(&self) -> &Self::C {
        &self.cost
    }

    fn get_trainer(&self) -> &Self::T {
        &self.trainer
    }
}
