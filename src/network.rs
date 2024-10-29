use std::vec;
use std::fs;
use std::fs::DirEntry;
use serde::{Serialize, Deserialize};

use crate::rand::prelude::SliceRandom;
use crate::layer::*;
use crate::weight::*;
use crate::bias::*;

extern crate rand;



#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Network 
{
    pub layers : Vec<Layer>,
    pub nweights: i32,
    pub nbiases: i32,
}

impl Network 
{
    pub fn new() -> Network
    {
        return Network{layers: Vec::new(), nweights: 0, nbiases: 0};
    }

    pub fn add_layer(&mut self, number : usize)
    {
        self.layers.push(Layer::new(number));
    }

    // add weights to every neuron
    // the number of weights per neuron in each layer is the same as the number of neurons in the previous layer
    pub fn set_up_weights(&mut self)
    {
        for i in 1..self.layers.len()
        {
            for j in 0..self.layers[i].neurons.len()
            {
                for k in 0..self.layers[i - 1].neurons.len() 
                {
                    self.layers[i].neurons[j].weights.push(Weight::new());
                    self.nweights += 1;
                }
            }
        }
    }
    
    //make sure every neuron has a bias
    pub fn set_up_biases(&mut self)
    {
        for i in 1..self.layers.len()
        {
            for j in 0..self.layers[i].neurons.len()
            {
                self.layers[i].neurons[j].bias = Bias::new();
                self.nbiases += 1;
            }
        }
    }

    //in this case each input neuron represents a pixel in the original image
    pub fn set_input(&mut self, input: Vec<f32>)
    {
        for i in 0..self.layers[0].neurons.len()
        {
            self.layers[0].neurons[i].activation = input[i];
        }
    }

    // from input layer top output, calculate each neurons activation by sigmoid(all weights timesed by their neuron + bias)
    pub fn compute_output(&mut self)
    {
        for i in 1..self.layers.len()
        {
            for j in 0..self.layers[i].neurons.len()
            {
                let mut sum = 0.0;
                for k in 0..self.layers[i].neurons[j].weights.len()
                {
                    sum +=  self.layers[i].neurons[j].weights[k].val  * self.layers[i - 1].neurons[k].activation; // timesing the kth weight of the jth neuron of the current layer by the kth neuron of the previous layer
                }
                self.layers[i].neurons[j].z = sum + self.layers[i].neurons[j].bias.val;
                self.layers[i].neurons[j].activation = sigmoid(self.layers[i].neurons[j].z)
            }
        }
    }

    pub fn print_layer_activation(&mut self, nlayer: usize)
    {
        for i in 0..self.layers[nlayer].neurons.len()
        {
            println!{"{:?}", self.layers[nlayer].neurons[i].activation}
        }
    }

    // all the activations of the final layer
    pub fn get_output_activation(&mut self) -> Vec<f32>
    {
        let mut activations = vec![];
        for i in 0..self.layers[self.layers.len()-1].neurons.len()
        {
            activations.push(self.layers[self.layers.len()-1].neurons[i].activation);
        }
        return activations;
    }

    // cost is (the sum of the differences between the current output and what it's supposed to be) squared
    pub fn get_cost(&mut self, actual: usize) -> f32
    {
        let mut sum = 0.0;
        for i in 0..(self.layers[self.layers.len()-1].neurons.len())
        {
            let target = if i == actual {1.0} else {0.0};
            sum += (self.layers[self.layers.len()-1].neurons[i].activation - target).powf(2.0)
        }
        return sum;
    }

    //getting the negative gradient of the cost function
    pub fn get_neg_grad(&mut self, actual: usize)
    {
        
        //set all dCdAs in network to 0
        for i in 0..self.layers.len()
        {
            for j in 0..self.layers[i].neurons.len()
            {
                self.layers[i].neurons[j].dCdA = 0.0;
            }
        }

        let last_id = self.layers.len() - 1;
        //set dCdA for the output layer
        for i in 0..self.layers[last_id].neurons.len()
        {
            let target = if i == actual {1.0} else {0.0};
            self.layers[last_id].neurons[i].dCdA = 2.0 * (self.layers[last_id].neurons[i].activation - target);
        }

        //back propogation
        for k in (1..self.layers.len()).rev()
        {
            for i in 0..self.layers[k].neurons.len() 
            {
                // set dCdB for current neuron
                let curr_z = self.layers[k].neurons[i].z;
                let dCdB : f32 = dsigmoid(curr_z) * self.layers[k].neurons[i].dCdA as f32;
                self.layers[k].neurons[i].bias.dCdB.push(dCdB);

                // set dCdWs for each weight connected to the neuron
                for j in 0..self.layers[k].neurons[i].weights.len()
                {
                    let curr_a = self.layers[k].neurons[i].activation; //neuron connected to weight on this layer
                    let last_a = self.layers[k-1].neurons[j].activation; //neuron connected to weight on previous layer

                    let dCdW : f32 = last_a * dsigmoid(curr_z) * self.layers[k].neurons[i].dCdA as f32;
                    self.layers[k].neurons[i].weights[j].dCdW.push(dCdW);

                    //add dCdA contributed through this neuron to previous layers neuron connected by the weight
                    if k != 1
                    {
                        self.layers[k-1].neurons[j].dCdA += self.layers[k].neurons[i].weights[j].val  *  dsigmoid(curr_z)  *  self.layers[k].neurons[i].dCdA as f32;
                    }
                }
            }
        }
    }
    pub fn train(&mut self, files : &mut Vec<DirEntry>, batch_size : usize)
    {
            for _i in 0..batch_size
            {
                if files.len() == 0
                {
                    return;
                }
                let file = files.pop().unwrap();
                let digit : usize= file.path().parent().unwrap().file_name().unwrap().to_owned().into_string().unwrap().parse::<usize>().unwrap();
                let bytes = image::open(file.path()).unwrap().into_bytes().iter().map(|x| *x as f32 / 255.0).collect::<Vec<f32>>();
                //println!("{:?}", bytes.len());
                self.set_input(bytes);
                self.compute_output();

                self.get_neg_grad(digit);
            }

            //adjust weights
            for i in 1..self.layers.len()
            {
                for j in 0..self.layers[i].neurons.len()
                {
                    for k in 0..self.layers[i - 1].neurons.len() // number of weights per neuron in each layer is the same as the number of neurons in the previous layer
                    {
                        let change : f32 = self.layers[i].neurons[j].weights[k].dCdW.iter().sum::<f32>() / self.layers[i].neurons[j].weights[k].dCdW.len() as f32; // get average nudge
                        self.layers[i].neurons[j].weights[k].val -= change; //idrk if this is supposed t be a smaller multiple
                        self.layers[i].neurons[j].weights[k].dCdW.clear();
                    }
                }
            }

            //adjust biases
            for i in 1..self.layers.len()
            {
                for j in 0..self.layers[i].neurons.len()
                {
                    let change : f32 = self.layers[i].neurons[j].bias.dCdB.iter().sum::<f32>() / self.layers[i].neurons[j].bias.dCdB.len() as f32; // get average nudge
                    self.layers[i].neurons[j].bias.val -= change;
                    self.layers[i].neurons[j].bias.dCdB.clear()
                }
            }
    }

    pub fn get_highest_activation(&mut self) -> usize
    {
        let highest_act = self.layers[self.layers.len()-1].neurons.iter().map(|x| x.activation).fold(std::f32::MIN, |a,b| a.max(b));
        return self.layers[self.layers.len()-1].neurons.iter().position(|x| x.activation == highest_act).unwrap();
    }

    //test current network against the mnist testing images, getting an accuracy percentage and average cost
    pub fn test(&mut self) -> (usize, Vec<(String, usize)>, f32)
    {
        let paths = fs::read_dir("pngs/testing/0").unwrap()
            .chain(fs::read_dir("pngs/testing/1").unwrap())
            .chain(fs::read_dir("pngs/testing/2").unwrap())
            .chain(fs::read_dir("pngs/testing/3").unwrap())
            .chain(fs::read_dir("pngs/testing/4").unwrap())
            .chain(fs::read_dir("pngs/testing/5").unwrap())
            .chain(fs::read_dir("pngs/testing/6").unwrap())
            .chain(fs::read_dir("pngs/testing/7").unwrap())
            .chain(fs::read_dir("pngs/testing/8").unwrap())
            .chain(fs::read_dir("pngs/testing/9").unwrap());

        let mut files = paths.map(|x| x.unwrap()).collect::<Vec<DirEntry>>();
        files.shuffle(&mut rand::thread_rng());

        let mut numo_correct = 0;
        let mut mistakes : Vec<(String, usize)> = vec![];
        let mut costs : Vec<f32> = vec![];
        for file in files
        {
            let digit : usize= file.path().parent().unwrap().file_name().unwrap().to_owned().into_string().unwrap().parse::<usize>().unwrap();
            let bytes = image::open(file.path()).unwrap().into_bytes().iter().map(|x| *x as f32 / 255.0).collect::<Vec<f32>>();
            self.set_input(bytes);
            self.compute_output();
            if self.get_highest_activation() == digit
            {
                numo_correct += 1;
            }
            else 
            {
                mistakes.push((file.path().to_str().unwrap().to_owned(), self.get_highest_activation()));
            }
            
            costs.push(self.get_cost(digit));
        }

        //let percent_correct = numo_correct as f32 / numo_tested as f32 * 100.0;
        //println!("{numo_correct} / {numo_tested} correct! | {percent_correct}% accurate");
        return (numo_correct, mistakes, costs.iter().sum::<f32>() / costs.len() as f32);
        
        // let bytes = image::open("pngs/legoat.png").unwrap().into_bytes().iter().step_by(4).map(|x| *x as f32 / 255.0).collect::<Vec<f32>>();
        // println!("{:?}", bytes);
        // self.set_input(bytes);
        // self.compute_output();
        // self.print_layer_activation(3);
        
    }
}

//activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

//derivative of the activation function
fn dsigmoid(x: f32) -> f32 {
    (-x).exp() / (1.0 + (-x).exp()).powf(2.0)
}