use utils::sum_vec;
use num_rust::Matrix2d;

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
    fn activation(&self, z: &Matrix2d) -> Matrix2d;
    fn activation_prime(&self, z: &Matrix2d) -> Matrix2d;
}

pub struct Sigmoid;

impl ActivationFunc for Sigmoid {
    fn activation(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn = |z: f64| { 1. / (1. + (-z).exp()) };
        z.par_apply_fn(&activ_fn)
    }

    fn activation_prime(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn = |z: f64| { 1. / (1. + (-z).exp()) };
        let activ_fn_prime = move |z: f64| { activ_fn(z) * (1. - activ_fn(z)) };
        z.par_apply_fn(&activ_fn_prime)
    }
}

pub struct Identity;

impl ActivationFunc for Identity {
    fn activation(&self, z: &Matrix2d) -> Matrix2d {
        z.clone()
    }

    fn activation_prime(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn_prime = |_: f64| { 1. };
        z.apply_fn(&activ_fn_prime)
    }
}

pub struct TanH;

impl ActivationFunc for TanH {
    fn activation(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn = |z: f64| { 2. / (1. + (-2. * z).exp()) - 1. };
        z.apply_fn(&activ_fn)
    }

    fn activation_prime(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn = |z: f64| { 2. / (1. + (-2. * z).exp()) - 1. };
        let activ_fn_prime = move |z: f64| {
            let out = activ_fn(z);
            1. - out * out
         };
        z.apply_fn(&activ_fn_prime)
    }
}

pub struct ReLU;

impl ActivationFunc for ReLU {
    fn activation(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn = |z: f64| { if z < 0. {
            0.
        } else {
            z
        } };
        z.apply_fn(&activ_fn)
    }

    fn activation_prime(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn_prime = |z: f64| { if z < 0. {
            0.
        } else {
            1.
        } };
        z.apply_fn(&activ_fn_prime)
    }
}

pub struct SoftPlus;

impl ActivationFunc for SoftPlus {
    fn activation(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn = |z: f64| { (1. + z.exp()).ln() };
        z.apply_fn(&activ_fn)
    }

    fn activation_prime(&self, z: &Matrix2d) -> Matrix2d {
        let activ_fn_prime = |z: f64| { 1. / (1. + (-z).exp()) };
        z.apply_fn(&activ_fn_prime)
    }
}

pub struct SoftMax;

impl ActivationFunc for SoftMax {
    fn activation(&self, z: &Matrix2d) -> Matrix2d {
        let mut max: f64 = 0.;
        for el in z.ravel().iter() {
            if *el > max {
                max = *el;
            }
        }
        let shift_z = z.apply_fn(|el| el - max);
        let m_exp = shift_z.apply_fn(&|x: f64| x.exp());
        let sum_exp = sum_vec( &m_exp.ravel() );
        let activ_fn = move |el: f64| {
            el / sum_exp
        };
        m_exp.apply_fn(&activ_fn)
    }

    fn activation_prime(&self, z: &Matrix2d) -> Matrix2d {
        let sf = self.activation(z);
        sf.mult(&sf.apply_fn(|x| 1. - x)).unwrap()
    }
}
