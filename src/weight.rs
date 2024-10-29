use serde::{Serialize, Deserialize};
use rand::Rng;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Weight
{
    pub val : f32,
    pub dCdW : Vec<f32> //List of changes wanted for each image of the batch that will be averaged
}

impl Weight
{
    pub fn new() -> Weight
    {
        Weight {val: rand::thread_rng().gen_range(-1.0..1.0), dCdW : Vec::new()}
    }
}