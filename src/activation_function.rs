use utils::sum_vec;

trait FastExp {
    fn fast_exp(&self) -> f64;
}

impl FastExp for f64 {
    fn fast_exp(&self) -> f64 {
        let x = 1. + *self/1024.;
        x.powi(1024)
    }
}

pub trait ActivationFunc {
    fn activation_fn(&self, x: f64) -> f64;
    fn activation_fn_prime(&self, x: f64) -> f64;
}

#[derive(Clone)]
pub struct Sigmoid;

impl ActivationFunc for Sigmoid {
    fn activation_fn(&self, z: f64) -> f64 {
        // 1. / (1. + (-z).exp())
        // fast version
        1. / (1. + (-z).fast_exp())
    }

    fn activation_fn_prime(&self, z: f64) -> f64 {
        self.activation_fn(z) * (1. - self.activation_fn(z))
    }
}

#[derive(Clone)]
pub struct Identity;

impl ActivationFunc for Identity {
    fn activation_fn(&self, z: f64) -> f64 {
        z
    }

    fn activation_fn_prime(&self, z: f64) -> f64 {
        1.
    }
}

#[derive(Clone)]
pub struct TanH;

impl ActivationFunc for TanH {
    fn activation_fn(&self, z: f64) -> f64 {
        2. / (1. + (-2. * z).exp()) - 1.
    }

    fn activation_fn_prime(&self, z: f64) -> f64 {
        1. - self.activation_fn(z) * self.activation_fn(z)
    }
}

#[derive(Clone)]
pub struct ReLU;

impl ActivationFunc for ReLU {
    fn activation_fn(&self, z: f64) -> f64 {
        if z < 0. {
            0.
        } else {
            z
        }
    }

    fn activation_fn_prime(&self, z: f64) -> f64 {
        if z < 0. {
            0.
        } else {
            1.
        }
    }
}

#[derive(Clone)]
pub struct SoftPlus;

impl ActivationFunc for SoftPlus {
    fn activation_fn(&self, z: f64) -> f64 {
        (1. + z.exp()).ln()
    }

    fn activation_fn_prime(&self, z: f64) -> f64 {
        1. / (1. + (-z).exp())
    }
}

// #[derive(Clone)]
// pub struct SoftMax<'a> {
//     activites: &'a Matrix2d
// }
//
// impl SoftMax<'a> {
//     new(activites: &'a Matrix2d) -> SoftMax<'a> {
//         SoftMax {
//             activites: activites
//         }
//     }
// }
//
// impl ActivationFunc for SoftMax {
//     fn activation_fn(&self, z: f64) -> f64 {
//         z.exp() / sum_vec(self.activites.apply_fn(|x| x.exp()).ravel())
//     }
//
//     fn activation_fn_prime(&self, z: f64) -> f64 {
//         1. / (1. + (-z).exp())
//     }
// }
