use num_rust::Matrix2d;
use neural_net::{NNetError};
use activation_function::ActivationFunc;

pub trait Layer {
    fn set_activity(&mut self, x: &Matrix2d, w: &Matrix2d) -> Result<(), NNetError>;
    fn get_activity(&self) -> &Matrix2d;
    fn get_activation(&self) -> Matrix2d;
    fn len(&self) -> usize;
    fn get_gradient(&self) -> Matrix2d;
    fn set_input(&mut self, x: &Matrix2d) -> ();
}

pub struct Dense<AF: ActivationFunc> {
    activity: Matrix2d,
    activation_func: AF,
    len: usize
}

impl<AF: ActivationFunc> Dense<AF> {
    pub fn new(len: usize, activation_func: AF) -> Result<Dense<AF>, NNetError> {
        if len > 0 {
            return Ok(Dense {
                activity: Matrix2d::new(1, 1),
                activation_func: activation_func,
                len: len
            });
        }
        Err(NNetError::LayerSize)
    }
}

impl<AF: ActivationFunc> Layer for Dense<AF> {
    fn set_input(&mut self, x: &Matrix2d) -> () {
        self.activity = x.clone();
    }
    fn set_activity(&mut self, x: &Matrix2d, w: &Matrix2d) -> Result<(), NNetError> {
        // println!("X: {:?}\n W: {:?}", X, W);
        let new_activity = match x.dot(w) {
            Some(v) => v,
            None => return Err(NNetError::ActivityError)
        };

        self.activity = new_activity;
        Ok(())
    }

    fn get_activity(&self) -> &Matrix2d {
        &self.activity
    }

    fn get_activation(&self) -> Matrix2d {
        self.activity.apply_fn(|z| { self.activation_func.activation_fn(z) })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn get_gradient(&self) -> Matrix2d {
        self.activity.apply_fn(|z| { self.activation_func.activation_fn_prime(z) })
    }
}
