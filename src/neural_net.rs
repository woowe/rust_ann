
pub trait NeuralNet {
    type Net;
    type Predict;
    type Error;

    fn new(&self, desc: Vec<Layer>, cost: CostFunction, trainer: Trainer) -> Result<self::Net, self::Error>;
    fn predict(&mut self, input: &Matrix2d) -> Result<self::Predict, self::Error>;
    fn fit(&mut self,  input: &Matrix2d, output: &Matrix2d) -> Result<(), self::Error>;
}

enum NNetError {
    DescSize
    GenericError
}

pub struct Sequential {
    layers: Vec<Layer>,
    weights: Vec<Matrix2d>
    cost: CostFunction,
    trainer: Trainer
}

impl NeuralNet for Sequential {
    type Net = Sequential;
    type Predict = Matrix2d;
    type Error = NNetError;
    fn new(&self, desc: Vec<Layer>, cost: CostFunction, trainer: Trainer) -> Result<self::Net, self::Error> {
        if desc.len() > 2 {
            return Ok(
                Sequential {
                    layers: desc.clone(),
                    weights: (0..desc.len() - 1).map(|idx| {
                        Matrix2d::fill_rng(desc[idx].len(), desc[idx + 1].len())
                    }),
                    cost: cost,
                    trainer: trainer
                }
            )
        }
        return Err(NNetError::DescSize)
    }

    fn predict(&mut self, input: &Matrix2d) -> Result<self::Predict, self::Error> {
        try!(self.layers[0].set_activity(&input, &self.weights[0]));

        for (idx, weight) in self.weights.iter().skip(1).enumerate() {
            let prev_activation = self.layers[idx - 1].get_activation();
            try!(self.layers[idx].set_activity(&prev_activation, weight));
        }

        return Ok(self.layers.last().get_activation())
    }


}
