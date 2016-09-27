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
    fn cost(&self, actual: &Matrix2d, pred: &Matrix2d) -> Result<f64, NNetError>;
    fn cost_prime(&self, input: &Matrix2d, actual: &Matrix2d) -> Result<Vec<Matrix2d>, NNetError>;
}

pub struct MSE_Reg<'a, Net: 'a + NeuralNet> {
    net: &'a Net,
    lambda: f64
}

impl<'a, Net: 'a + NeuralNet> MSE_Reg<'a, Net> {
    pub fn new(net: &'a Net, lambda: f64) -> MSE_Reg<'a, Net> {
        MSE_Reg {
            net: net,
            lambda: lambda
        }
    }
}

impl<'a, Net: 'a + NeuralNet> CostFunction for MSE_Reg<'a, Net> {
    fn cost(&self, actual: &Matrix2d, pred: &Matrix2d) -> Result<f64, NNetError> {
        let cost =  try_net!( (*actual).clone() - (*pred).clone(), NNetError::CostError );

        let w_sum = self.net.get_weights().iter().fold(0f64, |acc, w| acc + sum_vec(&w.apply_fn(|x| x*x).get_matrix()[..]) );
        Ok(0.5f64 * sum_vec(&cost.get_matrix()[..]) / (pred.get_rows() as f64) + ( (self.lambda/2.0)* w_sum ))
    }

    fn cost_prime(&self, input: &Matrix2d, actual: &Matrix2d) -> Result<Vec<Matrix2d>, NNetError> {
        let mut deltas = Vec::new();
        let mut djdw = Vec::new();

        let y_hat = try!(self.net.predict(input));
        let z_last = match self.net.get_layers().last() {
            Some(v) => v.get_activity(),
            None => return Err(NNetError::GradientError)
        };

        let cost_matrix = match actual.clone() - y_hat.clone() {
            Some(v) => -v,
            None => return Err(NNetError::GradientError)
        };

        // δ(last) = -(y - ŷ) * activation_prime(activities(last))
        let layer = match self.net.get_layers().last() {
            Some(v) => v,
            None => return Err(NNetError::GradientError)
        };
        let delta = match cost_matrix.mult(&z_last.apply_fn( layer.get_activation_fn().activation_fn_prime )) {
            Some(tmp) =>  tmp,
            None => return Err(NNetError::GradientError)
        };
        deltas.push(delta);

        // just a compute of the reciprocal of y_hat.get_rows() so I don't recompute in the loop
        let r_yhat: f64 = 1.0 / (y_hat.get_rows() as f64);

        let layer_len = self.net.get_layers().len();

        for n in 0..(layer_len - 2) {
            let idx = (layer_len - 2) - n;
            let a_t = &self.net.get_layers()[idx - 1].get_activity().transpose();
            let prev_delta = try_net!(deltas.last(), NNetError::GradientError).clone();
            let l_w = self.net.get_weights()[idx].scale(self.lambda);
            // DJDW(idx) = activations(idx - 1).T ⊗ δ(idx - 1) * 1/m + W(idx) * λ
            let v = match a_t.dot(&prev_delta) {
                Some(tmp) => tmp,
                None => return Err(NNetError::GradientError)
            };

            let dw = match v.scale(r_yhat).addition(&l_w) {
                Some(tmp) => tmp,
                None => return Err(NNetError::GradientError)
            };

            djdw.push(dw);

            let w_t = &self.net.get_weights()[idx].transpose();
            let layer = self.net.get_layers()[idx - 1];
            let z_prime = &layer.get_activity().apply_fn( layer.get_activation_fn().activation_fn_prime );

            // δ(idx) = δ(idx - 1) ⊗ W(idx).T * activation_prime(activities(idx - 1))
            let v = match prev_delta.dot(w_t) {
                Some(tmp) =>  tmp,
                None => return Err(NNetError::GradientError)
            };

            let delta = match v.mult(z_prime) {
                Some(tmp) => tmp,
                None => return Err(NNetError::GradientError)
            };

            deltas.push(delta);
        }

        // δ(0) = X.T ⊗ δ(1) * 1/m + W(0) * λ
        let l_d = match deltas.last() {
            Some(tmp) => tmp,
            None => return Err(NNetError::GradientError)
        };

        let w = &self.net.weights[0].scale(self.lambda);
        let v = match input.transpose().dot(l_d) {
            Some(tmp) => tmp,
            None => return Err(NNetError::GradientError)
        };

        let dw = match v.scale(r_yhat).addition(w) {
            Some(tmp) => tmp,
            None => return Err(NNetError::GradientError)
        };

        djdw.push(dw);
        djdw.reverse();
        return Ok(djdw);
    }
}
