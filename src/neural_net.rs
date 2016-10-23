
use num_rust::Matrix2d;

use layer::Layer;
// use cost_function::CostFunction;
// use trainer::Trainer;

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
    // type C: CostFunction;
    // type T: Trainer;

    fn new(desc: Vec<Self::L>) -> Result<Self::Net, NNetError>;
    fn predict(&mut self, input: &Matrix2d) -> Result<Matrix2d, NNetError>;
    // fn fit(&mut self,  input: &Matrix2d, output: &Matrix2d, trainer: &Self::T, cost: &Self::C) -> Result<(), NNetError>;

    fn set_weights(&mut self, n_weights: Vec<Matrix2d>) -> Result<(), NNetError>;
    // fn get_activities(&self) -> &[Matrix2d];
    fn get_weights(&self) -> &[Matrix2d];
    fn get_layers(&self) -> &[Self::L];
}

#[derive(Clone)]
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
            // print!("{:?}", weights);

            // let mut activities = Vec::new();
            // unsafe {
            //     activities.set_len(desc_len  - 1);
            // }
            return Ok(
                Sequential {
                    layers: desc,
                    weights: weights,
                    // activities: activities,
                    // cost: cost,
                    // trainer: trainer
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

            // fn feed_forward(&mut self, input: &Matrix2d) -> Matrix2d {
            //     // compute activity of X ⊗ W(0)
            //     self.activities[0] = input.dot(&self.weights[0]).expect("Dot product went wrong X*W(0)");
            //
            //     for (idx, weight) in self.weights.iter().enumerate().skip(1) {
            //         // compute the activation of activation( activities(idx - 1) )
            //         self.activations[idx - 1] = self.activities[idx - 1].apply_fn(sigmoid);
            //         // compute activity of activation(idx - 1) ⊗ W(idx)
            //         self.activities[idx] = self.activations[idx - 1].dot(weight).expect(&format!("Dot product went wrong: a({})*W({})", idx - 1, idx));
            //     }
            //
            //     // compute the last activation activation( activities(last) )
            //     return self.activities.last().unwrap().apply_fn(sigmoid);
            // }

        let _ = self.layers[0].set_input(&input);
        // println!("{:?}", self.layers[idx].get_activity());
        for (idx, weight) in self.weights.iter().enumerate() {
            // println!("{:?}", idx);
            let prev_activation = self.layers[idx].get_activation();
            let _ = try!(self.layers[idx + 1].set_activity(&prev_activation, weight));
            // println!("{:?}", self.layers[idx].get_activity());
            // println!("{:?}", self.layers[idx].get_activation());
            // match activation.dot(weight) {
            //     Some(v) => { self.activities[idx] = v; },
            //     None => return Err(NNetError::ActivityError)
            // };
        }

        // println!("{:?}", self.layers.last().unwrap().get_activity());

        match self.layers.last() {
            Some(l) => Ok(l.get_activation()),
            None => Err(NNetError::PredictError)
        }
    }

    // fn fit(&mut self,  input: &Matrix2d, output: &Matrix2d, trainer: &T, cost: &CF) -> Result<(), NNetError> {
    //     let updated_weights  = try!(trainer.optimize(input, output, cost));
    //     self.weights = updated_weights;
    //     Ok(())
    // }

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
    // fn get_activities(&self) -> &[Matrix2d] {
    //     &self.activities[..]
    // }

    fn get_weights(&self) -> &[Matrix2d] {
        &self.weights[..]
    }

    fn get_layers(&self) -> &[Self::L] {
        &self.layers[..]
    }

    // fn get_cost(&mut self) -> &Self::C {
    //     &self.cost
    // }
    //
    // fn get_trainer(&mut self) -> &Self::T {
    //     &self.trainer
    // }
}
