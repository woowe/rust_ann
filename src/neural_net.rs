
pub trait NeuralNet {
    type Net;
    type Error;

    fn new(&self, desc: Vec<Layer>, cost: CostFunction, trainer: Trainer) -> Result<self::Net, self::Error>;
    fn fit(&mut self,  input: &Matrix2d, output: &Matrix2d) -> Result<(), self::Error>;
    fn predict(&mut self, input: &Matrix2d) -> Result<(), self::Error>;
}

enum NNetError {
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
    type Error = NNetError;
    fn new(&self, desc: Vec<Layer>, cost: CostFunction, trainer: Trainer) -> Result<self::Net, self::Error> {
        if desc.len() > 2 {
            Ok(
                Sequential {
                    layers: desc.clone(),
                    weights: (0..desc.len() - 1).map(|idx| {
                        Matrix2d::fill_rng(desc[idx].len(), desc[idx + 1].len())
                    })
                }
            )
        }
        return Err(NNetError::GenericError)
    }
}
