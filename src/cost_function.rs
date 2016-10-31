use num_rust::Matrix2d;
use num_rust::utils::sum_vec;

use neural_net::{NeuralNet, NNetError};
use layer::Layer;

macro_rules! try_net {
    ( $expr : expr, $err:expr ) => (
        match $expr {
            Some(v) => v,
            None => return Err($err)
        }
    );
}

pub trait CostFunction {
    fn cost<NN: NeuralNet>(&self, net: &NN, actual: &Matrix2d, pred: &Matrix2d) -> Result<f64, NNetError>;
    fn cost_prime<NN: NeuralNet>(&mut self, net: &mut NN, input: &Matrix2d, actual: &Matrix2d, y_hat: &Matrix2d) -> Result<Vec<Matrix2d>, NNetError>;
}

pub struct MSE_Reg {
    lambda: f64
}

impl MSE_Reg {
    pub fn new(lambda: f64) -> MSE_Reg {
        MSE_Reg {
            lambda: lambda
        }
    }
}

impl CostFunction for MSE_Reg {
    fn cost<NN: NeuralNet>(&self, net: &NN, actual: &Matrix2d, pred: &Matrix2d) -> Result<f64, NNetError> {
        let cost =  try_net!( (*actual).clone() - (*pred).clone(), NNetError::CostError ).apply_fn(|x| x * x);

        let w_sum = net.get_weights().iter().fold(0f64, |acc, w| acc + sum_vec(&w.apply_fn(|x| x*x).get_matrix()[..]) );
        Ok(0.5f64 * sum_vec(&cost.get_matrix()[..]) / (pred.get_rows() as f64) + ( (self.lambda/2.0)* w_sum ))
    }

    fn cost_prime<NN: NeuralNet>(&mut self, net: &mut NN, input: &Matrix2d, actual: &Matrix2d, y_hat: &Matrix2d) -> Result<Vec<Matrix2d>, NNetError> {
        let mut deltas = Vec::new();
        let mut djdw = Vec::new();
        let cost_matrix = match actual.clone() - y_hat.clone() {
            Some(v) => -v,
            None => return Err(NNetError::GradientError)
        };
        let r_yhat = 1.0 / (y_hat.get_rows() as f64);

        // let activities = net.get_layers().iter().map(|l| l.get_activity().clone()).collect::<Vec<Matrix2d>>();
        let layer_len = net.get_layers().len();
        let activations = net.get_layers()[1..layer_len - 1].iter().map(|l| l.get_activation()).collect::<Vec<Matrix2d>>();
        let layer_gradients = net.get_layers()[1..].iter().map(|l| l.get_gradient()).collect::<Vec<Matrix2d>>();
        let weights = net.get_weights();
        let delta = try_net!(cost_matrix.mult(&layer_gradients.last().unwrap()), NNetError::GradientError);
        deltas.push(delta);
        for n in 0..(layer_len - 2) {
            let idx = (layer_len - 2) - n;
            let a_t = &activations[idx - 1].transpose();
            let prev_delta = deltas.last().unwrap().clone();
            let l_w = weights[idx].scale(self.lambda);
            // DJDW(idx) = activations(idx - 1).T ⊗ δ(idx - 1) * 1/m + W(idx) * λ
            djdw.push(a_t.dot(&prev_delta).expect("Dot product gone wrong a_t * prev_delta").scale(r_yhat)
                        .addition(&l_w).expect("Addition gone wrong a_t * prev_delta + lambda * W"));

            let w_t = &weights[idx].transpose();
            let z_prime = &layer_gradients[idx - 1];

            // δ(idx) = δ(idx - 1) ⊗ W(idx).T * activation_prime(activities(idx - 1))
            let delta = prev_delta.dot(w_t).expect("Dot product gone wrong prev_delta * w_t").mult(z_prime).expect("Mult gone wrong (prev_delta * w_t) x z_prime");
            deltas.push(delta);
        }

        // δ(0) = X.T ⊗ δ(1) * 1/m + W(0) * λ
        djdw.push(input.transpose().dot(&deltas.last().unwrap()).expect("Dot gone wrong X_t * delta_last").scale(r_yhat)
                    .addition(&weights[0].scale(self.lambda)).expect("Addition gone wrong X_t * delta_last + lambda * W(0)"));
        djdw.reverse();
        return Ok(djdw);
    }
}
