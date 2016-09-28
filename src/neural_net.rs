
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
    fn fit(&mut self,  input: &Matrix2d, output: &Matrix2d, trainer: &Self::T, cost: &Self::C) -> Result<(), NNetError>;

    fn get_activities(&self) -> &[Matrix2d];
    fn get_weights(&self) -> &[Matrix2d];
    fn get_layers(&self) -> &[Self::L];
}

pub struct Sequential<L: Layer> {
    layers: Vec<L>,
    weights: Vec<Matrix2d>,
    activities: Vec<Matrix2d>
}

impl<L: Layer> NeuralNet for Sequential<L, CF, T> {
    type Net = Sequential<L, CF, T>;
    type L = L;
    type C = CF;
    type T = T;

    fn new(&self, desc: Vec<L>, cost: CF, trainer: T) -> Result<Sequential<L, CF, T>, NNetError> {
        if desc.len() > 2 {
            let desc_len = desc.len();
            let weights = (0..desc_len - 1).map(|idx| {
                Matrix2d::fill_rng(desc[idx].len(), desc[idx + 1].len())
            }).collect::<Vec<Matrix2d>>();

            let mut activities = Vec::new();
            unsafe {
                activities.set_len(desc_len  - 1);
            }
            return Ok(
                Sequential {
                    layers: desc,
                    weights: weights,
                    activities: activities,
                    cost: cost,
                    trainer: trainer
                }
            )
        }
        Err(NNetError::DescSize)
    }

    fn predict(&mut self, input: &Matrix2d) -> Result<Matrix2d, NNetError> {
        // match input.dot(&self.weights[0]) {
        //     Some(v) => { self.activities[0] = v; },
        //     None => return Err(NNetError::ActivityError)
        // };

        let _ = try!(self.layers[0].set_activity(&input, &self.weights[0]));

        for (idx, weight) in self.weights.iter().skip(1).enumerate() {
            let activation = self.layers[idx - 1].get_activation();
            let _ = try!(self.layers[idx].set_activity(activation, weight));
            // match activation.dot(weight) {
            //     Some(v) => { self.activities[idx] = v; },
            //     None => return Err(NNetError::ActivityError)
            // };
        }

        match self.layers.last() {
            Some(v) => match self.activities.last() {
                Some(a) => Ok(v.get_activation(&a)),
                None => Err(NNetError::PredictError)
            },
            None => Err(NNetError::PredictError)
        }
    }

    fn fit(&mut self,  input: &Matrix2d, output: &Matrix2d, trainer: &T, cost: &CF) -> Result<(), NNetError> {
        let updated_weights  = try!(trainer.optimize(input, output, cost));
        self.weights = updated_weights;
        Ok(())
    }

    fn get_activities(&self) -> &[Matrix2d] {
        &self.activities[..]
    }

    fn get_weights(&self) -> &[Matrix2d] {
        &self.weights[..]
    }

    fn get_layers(&self) -> &[Self::L] {
        &self.layers[..]
    }

    fn get_cost(&mut self) -> &Self::C {
        &self.cost
    }

    fn get_trainer(&mut self) -> &Self::T {
        &self.trainer
    }
}
