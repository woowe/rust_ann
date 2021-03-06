use num_rust::Matrix2d;
use neural_net::{NeuralNet, NNetError};
use rand;
use cost_function::CostFunction;
// use utils::frobenius_norm;

pub trait Trainer {
    fn optimize(&mut self, input: &Matrix2d, actual: &Matrix2d) -> Result<(), NNetError>;
}

pub struct MiniBatchSGD<'a, NN: 'a + NeuralNet, C: 'a + CostFunction> {
    epochs: usize,
    batch_size: usize,
    learn_rate: f64,
    net: &'a mut NN,
    cost: &'a mut C
}

impl<'a, NN: 'a + NeuralNet, C: 'a + CostFunction> MiniBatchSGD<'a, NN, C> {
    pub fn new(net: &'a mut NN, cost: &'a mut C, epochs: usize, batch_size: usize, learn_rate: f64) -> Result<MiniBatchSGD<'a, NN, C>, NNetError> {
        Ok(MiniBatchSGD {
            epochs: epochs,
            batch_size: batch_size,
            learn_rate: learn_rate,
            net: net,
            cost: cost
        })
    }
}

impl<'a, NN: 'a + NeuralNet, C: 'a + CostFunction> Trainer for MiniBatchSGD<'a, NN, C> {
    fn optimize(&mut self, input: &Matrix2d, actual: &Matrix2d) -> Result<(), NNetError> {

        let seed: &[_] = &[rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>()];
        let shuffled_input = input.shuffle(seed).mini_batch(self.batch_size);
        let shuffled_actual = actual.shuffle(seed).mini_batch(self.batch_size);

        let mut training_data = shuffled_input.iter().map(|el| el.clone()).zip(shuffled_actual).collect::<Vec<(Matrix2d,Matrix2d)>>();

        let mut djdws: Vec<Matrix2d> = Vec::new();

        for i in 0..self.epochs {
            for &(ref s_input, ref s_output) in training_data.iter() {
                let pred = try!(self.net.predict(s_input));
                djdws = try!(self.cost.cost_prime(self.net, &s_input, &s_output, &pred));
                // println!("PRED: {:?}\nDJDWS: {:?}", pred, djdws);
                // println!("Optimizing... {}", self.epochs);
                let mut net_weights = self.net.get_weights().to_vec();
                let mut djdws_iter = djdws.iter();
                let i_f = i as f64;
                let lr = self.learn_rate / ( ( i_f / self.epochs as f64 ) * i_f + 1.);
                for weight in net_weights.iter_mut() {
                    // gradient descent
                    // W(i) = W(i) - alhpa * DJDW
                    let djdw = match djdws_iter.next() {
                        Some(v) => v.scale(lr),
                        None => return Err(NNetError::OptimizeError)
                    };
                    let tmp_weight = match (*weight).clone() - djdw {
                        Some(v) => v,
                        None => return Err(NNetError::OptimizeError),
                    };
                    *weight = tmp_weight;
                }
                let _ = self.net.set_weights(net_weights.clone());
            }

            if i >= 0 {
                let pred = self.net.predict(&input).unwrap();
                // let djdws = try!(self.cost.cost_prime(self.net, &s_input, &s_output, &pred));
                // println!("EPOCH {}", i);
                println!("{}.\t{}", i, self.cost.cost(self.net, actual, &pred).unwrap());
                // println!("ADDED GRADIENTS: {:?}", try!(self.cost.cost_prime(self.net, &s_input, &s_output, &pred)));
            }


            // let seed: &[_] = &[rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>(), rand::random::<usize>()];
            // let _ = training_data.iter_mut().map(|&mut (ref mut si,ref mut so)| {
            //     *si = si.shuffle(seed);
            //     *so = so.shuffle(seed);
            // });
            let shuffled_input = input.shuffle(seed).mini_batch(self.batch_size);
            let shuffled_actual = actual.shuffle(seed).mini_batch(self.batch_size);

            training_data = shuffled_input.iter().map(|el| el.clone()).zip(shuffled_actual).collect::<Vec<(Matrix2d,Matrix2d)>>();
        }


        Ok(())
    }
}
