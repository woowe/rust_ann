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
    fn cost_prime<NN: NeuralNet>(&mut self, net: &mut NN, input: &Matrix2d, actual: &Matrix2d) -> Result<Vec<Matrix2d>, NNetError>;
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
        let cost =  try_net!( (*actual).clone() - (*pred).clone(), NNetError::CostError );

        let w_sum = net.get_weights().iter().fold(0f64, |acc, w| acc + sum_vec(&w.apply_fn(|x| x*x).get_matrix()[..]) );
        Ok(0.5f64 * sum_vec(&cost.get_matrix()[..]) / (pred.get_rows() as f64) + ( (self.lambda/2.0)* w_sum ))
    }

    fn cost_prime<NN: NeuralNet>(&mut self, net: &mut NN, input: &Matrix2d, actual: &Matrix2d) -> Result<Vec<Matrix2d>, NNetError> {
        let mut deltas = Vec::new();
        let mut djdw = Vec::new();
        let mut cost_matrix: Matrix2d;
        // just a compute of the reciprocal of y_hat.get_rows() so I don't recompute in the loop
        let mut r_yhat: f64;
        {
            let y_hat = try!(net.predict(input));
            r_yhat = 1.0 / (y_hat.get_rows() as f64);
            // let z_last = match net.get_layers().last() {
            //     Some(v) => v.get_activity(),
            //     None => return Err(NNetError::GradientError)
            // };

            println!("COST MATRIX");
            cost_matrix = match actual.clone() - y_hat.clone() {
                Some(v) => -v,
                None => return Err(NNetError::GradientError)
            };
        }

        // δ(last) = -(y - ŷ) * activation_prime(activities(last))
        println!("LAST LAYER");
        let layer = match net.get_layers().last() {
            Some(v) => v,
            None => return Err(NNetError::GradientError)
        };
        // let activity = match net.get_activities().last() {
        //     Some(v) => v,
        //     None => return Err(NNetError::GradientError)
        // };
        println!("DELTA");
        let delta = match cost_matrix.mult(&layer.get_gradient()) {
            Some(tmp) =>  tmp,
            None => return Err(NNetError::GradientError)
        };
        deltas.push(delta);

        let layer_len = net.get_layers().len();

        for n in 0..(layer_len - 2) {
            let idx = (layer_len - 2) - n;
            let a_t = &net.get_layers()[idx - 1].get_activity().transpose();
            let prev_delta = try_net!(deltas.last(), NNetError::GradientError).clone();
            let l_w = net.get_weights()[idx].scale(self.lambda);
            // DJDW(idx) = activations(idx - 1).T ⊗ δ(idx - 1) * 1/m + W(idx) * λ
            println!("LOOP V{}", n);
            let v = match a_t.dot(&prev_delta) {
                Some(tmp) => tmp,
                None => return Err(NNetError::GradientError)
            };

            println!("LOOP DW{}\nV: {:?}\n LW: {:?}", n, v, l_w);
            let dw = match v.scale(r_yhat).addition(&l_w) {
                Some(tmp) => tmp,
                None => return Err(NNetError::GradientError)
            };

            djdw.push(dw);

            let w_t = &net.get_weights()[idx].transpose();
            let layer = &net.get_layers()[idx - 1];
            // let activity = &layer.get_activity();
            let z_prime = layer.get_gradient();

            // δ(idx) = δ(idx - 1) ⊗ W(idx).T * activation_prime(activities(idx - 1))
            println!("LOOP V2{}", n);
            let v = match prev_delta.dot(w_t) {
                Some(tmp) =>  tmp,
                None => return Err(NNetError::GradientError)
            };

            println!("LOOP DELTA{}",n);
            let delta = match v.mult(&z_prime) {
                Some(tmp) => tmp,
                None => return Err(NNetError::GradientError)
            };

            deltas.push(delta);
        }

        // δ(0) = X.T ⊗ δ(1) * 1/m + W(0) * λ
        println!("L_D");
        let l_d = match deltas.last() {
            Some(tmp) => tmp,
            None => return Err(NNetError::GradientError)
        };

        let w = &net.get_weights()[0].scale(self.lambda);
        println!("AFTER LOOP V");
        let v = match input.transpose().dot(l_d) {
            Some(tmp) => tmp,
            None => return Err(NNetError::GradientError)
        };

        println!("AFTER LOOP DW");
        let dw = match v.scale(r_yhat).addition(w) {
            Some(tmp) => tmp,
            None => return Err(NNetError::GradientError)
        };

        djdw.push(dw);
        djdw.reverse();
        Ok(djdw)
    }
}
