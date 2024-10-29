use serde::{Serialize, Deserialize};
use crate::weight::*;
use crate::bias::*;
use rand::Rng;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Neuron
{
    pub activation: f32, // sigmoid(z)
    pub z : f32, // weights * activations + bias
    pub weights: Vec<Weight>, // each neuron has a weight connecting each neuron in the previous layer
    pub bias: Bias,
    pub dCdA : f32
}

impl Neuron
{
    pub fn new() -> Neuron
    {
        Neuron{activation: rand::thread_rng().gen_range(-1.0..1.0), z : 0.01, weights: Vec::new(), bias: Bias::new(), dCdA: 0.0}
    }
}