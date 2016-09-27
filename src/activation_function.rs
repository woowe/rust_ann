pub trait ActivationFunc {
    fn activation_fn(&self, x: f64) -> f64;
    fn activation_fn_prime(&self, x: f64) -> f64;
}

pub struct Sigmoid {}

impl ActivationFunc for Sigmoid {
    fn activation_fn(&self, z: f64) -> f64 {
        1. / (1. + (-z).exp())
    }

    fn activation_fn_prime(&self, z: f64) -> f64 {
        self.activation_fn(z) * (1. - self.activation_fn(z))
    }
}
