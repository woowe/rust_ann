use num_rust::Matrix2d;
use neural_net::{NeuralNet, NNetError};
use rand;

pub trait Trainer {
    fn optimize(&self, input: &Matrix2d, actual: &Matrix2d) -> Result<Vec<Matrix2d>, NNetError>;
}

pub struct MiniBatchSGD<'a, NN: 'a + NeuralNet> {
    epochs: usize,
    batch_size: usize,
    learn_rate: f64,
    net: &'a NN
}

impl<'a, NN: 'a + NeuralNet> MiniBatchSGD<'a, NN> {
    fn new(net: &'a NN, epochs: usize, batch_size: usize, learn_rate: f64) -> Result<MiniBatchSGD<'a, NN>, NNetError> {
        if epochs < 0 || batch_size < 0 {
            return Err(NNetError::MiniBatchSGDInitError)
        }
        Ok(MiniBatchSGD {
            epochs: epochs,
            batch_size: batch_size,
            learn_rate: learn_rate,
            net: net
        })
    }
}

impl<'a, NN: 'a + NeuralNet> Trainer for MiniBatchSGD<'a, NN> {
    fn optimize(&self, input: &Matrix2d, actual: &Matrix2d) -> Result<Vec<Matrix2d>, NNetError> {
        let mut net_weights = self.net.get_weights().clone();
        let seed: &[_] = &[rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>()];
        let shuffled_input = input.shuffle(seed).mini_batch(self.batch_size);
        let shuffled_actual = actual.shuffle(seed).mini_batch(self.batch_size);

        let mut training_data = shuffled_input.iter().map(|el| el.clone()).zip(shuffled_actual).collect::<Vec<(Matrix2d,Matrix2d)>>();

        let mut djdws: Vec<Matrix2d>;

        for i in 0..self.epochs {
            for &(ref s_input, ref s_output) in training_data.iter() {
                let mut djdws_iter = djdws.iter();
                for weight in net_weights.iter_mut() {
                    // gradient descent
                    // W(i) = W(i) - alhpa * DJDW
                    let djdw = match djdws_iter.next() {
                        Some(v) => *v.scale(self.learn_rate),
                        None => return Err(NNetError::OptimizeError)
                    };
                    let tmp_weight = match (*weight).clone() - djdw {
                        Some(v) => v,
                        None => return Err(NNetError::OptimizeError),
                    };
                    *weight = tmp_weight;
                }
                djdws = try!(self.net.get_cost().cost_prime(&s_input, &s_output));
            }

            let seed: &[_] = &[rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>()];
            let _ = training_data.iter_mut().map(|&mut (ref mut si,ref mut so)| {
                *si = si.shuffle(seed);
                *so = so.shuffle(seed);
            });
        }

        Ok(djdws)
    }
}
