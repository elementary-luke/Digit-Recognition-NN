use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Bias
{
    pub val : f32,
    pub dCdB : Vec<f32> //List of changes wanted for each image of the batch that will be averaged
}

impl Bias
{
    pub fn new() -> Bias
    {
        Bias {val: 0.01, dCdB : Vec::new()}
    }
}