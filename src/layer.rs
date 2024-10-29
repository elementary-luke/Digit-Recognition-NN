use serde::{Serialize, Deserialize};
use crate::neuron::*;

#[derive(Clone, Serialize, Deserialize, Debug)]

pub struct Layer
{
    pub neurons: Vec<Neuron>,
}

impl Layer
{
    pub fn new(number: usize) -> Layer
    {
        Layer{neurons: vec![Neuron::new(); number]}
    }
}