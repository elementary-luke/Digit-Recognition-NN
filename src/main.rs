mod network;

use std::fs;
use std::fs::DirEntry;
use std::fs::File;
use std::path::Path;
use std::io::Write;
use std::io::Read;
use crate::network::*;
use egui::Align2;
use ::macroquad as mq;
use mq::prelude::*;
use egui_macroquad::*;
use egui::plot::*;
use macroquad::ui::Ui;
use rand::Rng;
use serde::de::Expected;
extern crate rand;
use crate::rand::prelude::SliceRandom;

fn window_conf() -> Conf {
    Conf {
        window_title: "digit recog".to_owned(),
        //fullscreen: true,
        window_width: 1920,
        window_height: 1080,
        ..Default::default()
    }
    
}

#[mq::main(window_conf())]
async fn main() 
{
    /* 
    let mut network = Network{layers: Vec::new(), nweights: 0, nbiases: 0};
    network.add_layer(784);
    network.add_layer(300);
    network.add_layer(100);
    network.add_layer(10);
    network.set_up_weights();
    network.set_up_biases();
    println!("number of weights: {:?}", network.nweights);
    println!("number of biases: {:?}", network.nbiases);
    
    
    
    
    //let mut network = load_network();
    
    
    for i in 0..10
    {
        if i % 5 == 0
        {
            network.test(false);
        }
        println!("{}th rerun", i);
        network.train(100, 600); //  these 2 parameters always multiply to 60 000, for best results set to (60 000, 1) and repeat a ridiculous amount of times i think idrk
    }
    save_network(network);
    */

    let mut screen = Screen::HOME;
    let mut network : Network = load_network("networks/64 32/network 200 100 600.txt".to_owned());
    let mut network_name : String = "network 200 100 600.txt".to_owned();
    //network.try_image();
    //network.test(true);
    let mut grid = Grid::new(28, 28);
    //network.test(true);
    let mut new_net_settings : Vec<i32> = vec![784, 10];
    let mut batch_size : usize = 100;
    let mut numobatches : usize = 600;
    let mut cbatch : usize = 0;
    let mut files : Vec<DirEntry> = vec![];
    let mut numo_correct = 0;
    let mut mistakes : Vec<(String, usize)> = vec![];
    let mut cost : f32 = 0.0;

    
    loop 
    {
        println!("{}", macroquad::time::get_fps());
        clear_background(RED);
       
        //network.print_layer_activation(3);
        //println!("{}",network.get_highest_activation());

        if screen == Screen::HOME
        {
            egui_macroquad::ui(|egui_ctx| {
                egui::Window::new("Home")                
                    .show(egui_ctx, |ui| {
                        ui.collapsing("Network", |ui| 
                        {
                            ui.label(network_name.clone());
                            ui.collapsing("Current structure", |ui| {
                                for i in 0..network.layers.len()
                                {
                                    ui.label(network.layers[i].neurons.len().to_string());
                                }
                            });

                            ui.collapsing("New", |ui| 
                            {
                                let mut to_delete : Option<usize> = None;
                                for i in 0..new_net_settings.len()
                                {
                                    if i == 0 || i == new_net_settings.len() - 1
                                    {
                                        ui.label(new_net_settings[i].to_string());
                                    }
                                    else 
                                    {
                                        ui.horizontal(|ui| {
                                            //ui.label(new_net_settings[i].to_string());
                                            ui.add(egui::DragValue::new(&mut new_net_settings[i]).speed(1).clamp_range(1..=3000));
        
                                            if ui.button("delete".to_owned()).clicked()
                                            {
                                                to_delete = Some(i)
                                            }
                                        });
                                    }

                                    if  i != new_net_settings.len() - 1
                                    {
                                        ui.horizontal(|ui| {
                                            if ui.button("add".to_owned()).clicked()
                                            {
                                                new_net_settings.insert(i+1, 1);
                                            }
                                        });
                                    }
                                    
                                }
                                if to_delete.is_some()
                                {
                                    new_net_settings.remove(to_delete.unwrap());
                                }
                                if ui.add(egui::Button::new("Create".to_string())).clicked() 
                                {
                                    network = Network::new();
                                    network_name = "Unsaved".to_owned();
                                    for i in 0..new_net_settings.len()
                                    {
                                        network.add_layer(new_net_settings[i] as usize);
                                    }
                                    network.set_up_weights();
                                    network.set_up_biases();
                                    new_net_settings = vec![784, 10];
                                }
                            });

                            
                            
                            if ui.add(egui::Button::new("Save".to_string())).clicked() 
                            {
                                if let Some(path) = rfd::FileDialog::new().add_filter("networks", &["ntwk", "txt"][0..2]).save_file()
                                {
                                    let path_name = path.into_os_string().into_string().unwrap();
                                    network_name = path_name.clone();
                                    save_network(network.clone(), path_name);
                                }
                            }
                            if ui.add(egui::Button::new("Load".to_string())).clicked() 
                            {
                                if let Some(path) = rfd::FileDialog::new().pick_file() 
                                {
                                    network_name = path.into_os_string().into_string().unwrap();
                                    network = load_network(network_name.clone());
                                }
                            }
                        });
                        ui.collapsing("Train", |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("batch size:");
                                    ui.add(egui::DragValue::new(&mut batch_size).speed(1).clamp_range(1..=60000));
                                });

                                ui.horizontal(|ui| {
                                    ui.label("number of batches:");
                                    ui.add(egui::DragValue::new(&mut numobatches).speed(1).clamp_range(1..=100000));
                                });
                                

                                if ui.button("Start").clicked()
                                {
                                    screen = Screen::TRAIN;
                                    cbatch = 0;
                                    network_name = "Unsaved".to_owned();
                                }
                        });
                        ui.collapsing("Test", |ui| {
                            if ui.button("draw").clicked()
                            {
                                screen = Screen::DRAW;
                            }
                            if ui.button("mnist test").clicked()
                            {
                                screen = Screen::TEST;
                                (numo_correct, mistakes, cost) = network.test();
                            }
                        })
                    });
            });
        }
        else if screen == Screen::DRAW
        {
            grid.draw();

            network.set_input(grid.data.clone());
            network.compute_output();
            egui_macroquad::ui(|egui_ctx| {
                egui::Window::new("Results")
                    .current_pos((560.0, 0.0))
                    .show(egui_ctx, |ui| {
                        if ui.button("Back").clicked()
                        {
                            screen = Screen::HOME;
                        }
                        if ui.button("Clear").clicked()
                        {
                            grid.data = vec![0.0; grid.width * grid.height];
                        }

                        let ys : Vec<f32> = network.get_output_activation();

                        let mut bars : Vec<Bar> = vec![];
                        for i in 0..ys.len()
                        {
                            bars.push(Bar::new(i as f64, ys[i] as f64).name(i))
                        }

                        let chart = BarChart::new(bars).width(0.3);

                        Plot::new("my_plot")
                        .legend(Legend::default()) 
                            .clamp_grid(false)
                            .x_grid_spacer(x_grid)
                            .y_grid_spacer(y_grid)
                            .allow_zoom(false)
                            .allow_drag(false)
                            .allow_scroll(false)
                            // .height(240.0)
                            // .width(500.0)
                            .include_y(1.1)
                            .allow_zoom(false)
                            .auto_bounds_y()
                            .show(ui, |plot_ui| plot_ui.bar_chart(chart));  
                        
                        let mut visuals = egui::Visuals::dark();
                        visuals.window_shadow.extrusion = 0.0;
            
                        let style = egui::Style {
                            visuals,
                            ..Default::default()
                        };
                        egui_ctx.set_style(style);
                    });
            });
        }
        else if screen == Screen::TRAIN
        {
            egui_macroquad::ui(|egui_ctx| {
                egui::Window::new("Training")
                    .show(egui_ctx, |ui| {
                        ui.label("Batch Size: ".to_owned() + &batch_size.to_string());
                        ui.label("Batch: ".to_owned() + &cbatch.to_string() + "/" + &numobatches.to_string());
                        let epoch = cbatch as f32 * batch_size as f32 / 60000.0;
                        let end_epoch = numobatches as f32 * batch_size as f32 / 60000.0;
                        ui.label("Epoch: ".to_owned() + &epoch.to_string() + "/" + &end_epoch.to_string());

                        if ui.button("Stop").clicked()
                        {
                            screen = Screen::HOME;
                        }
                });
            });
            if cbatch < numobatches
            {
                if files.len() == 0
                {
                    let paths = fs::read_dir("pngs/training/0").unwrap()
                    .chain(fs::read_dir("pngs/training/1").unwrap())
                    .chain(fs::read_dir("pngs/training/2").unwrap())
                    .chain(fs::read_dir("pngs/training/3").unwrap())
                    .chain(fs::read_dir("pngs/training/4").unwrap())
                    .chain(fs::read_dir("pngs/training/5").unwrap())
                    .chain(fs::read_dir("pngs/training/6").unwrap())
                    .chain(fs::read_dir("pngs/training/7").unwrap())
                    .chain(fs::read_dir("pngs/training/8").unwrap())
                    .chain(fs::read_dir("pngs/training/9").unwrap());

                    files = paths.map(|x| x.unwrap()).collect::<Vec<DirEntry>>();
                    files.shuffle(&mut rand::thread_rng());
                }
                network.train(&mut files, batch_size);
                cbatch += 1;
                
            }
            else
            {
                screen = Screen::HOME;
            }
        }
        else if screen == Screen::TEST
        {
            egui_macroquad::ui(|egui_ctx| {
                egui::Window::new("Test Results")
                    .show(egui_ctx, |ui| {
                        if ui.button("Back").clicked()
                        {
                            screen = Screen::HOME;
                        }
                        ui.label("Correct: ".to_owned() + &numo_correct.to_string() + " / " + "10000");
                        ui.label("Accuracy: ".to_owned() + &(numo_correct as f32 / 10000.0 * 100.0).to_string() + "%");
                        ui.label("Average Cost: ".to_owned() + &cost.to_string());
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for i in 0..mistakes.len()
                            {
                                ui.horizontal(|ui| {
                                    ui.label(format!("Thought {} was {}", mistakes[i].0, mistakes[i].1));
                                    if ui.button("show").clicked()
                                    {
                                        grid.data = image::open(mistakes[i].0.clone()).unwrap().into_bytes().iter().map(|x| *x as f32 / 255.0).collect::<Vec<f32>>();
                                        screen = Screen::DRAW;
                                    }
                                });
                                
                            }
                        });

                });
            });
        }

        egui_macroquad::draw();

        next_frame().await  
    }
    
    
    
}

fn x_grid(input: GridInput) -> Vec<GridMark> {
    // Note: this always fills all possible marks. For optimization, `input.bounds`
    // could be used to decide when the low-interval grids (minutes) should be added.

    let mut marks = vec![];

    for i in 0..10
    {
        marks.push(GridMark { value: i as f64 * 1.0, step_size: 100.0})
    }


    marks
}

fn y_grid(input: GridInput) -> Vec<GridMark> {
    // Note: this always fills all possible marks. For optimization, `input.bounds`
    // could be used to decide when the low-interval grids (minutes) should be added.

    let mut marks = vec![];

    for i in 0..11
    {
        marks.push(GridMark { value: i as f64 * 0.1, step_size: 30.0})
    }

    marks
}

fn save_network(network : Network, path : String)
{
    let serialised = serde_json::to_string(&network).unwrap();
    let mut data_file = File::create(path).expect("creation failed");

    // Write contents to the file
    data_file.write(serialised.as_bytes()).expect("write failed");
}

fn load_network(path : String) -> Network
{
    let mut data_file = File::open(path).unwrap(); //best so far is /64 32/network 200 100 600.txt
    let mut deserialised = String::new();
    data_file.read_to_string(&mut deserialised).unwrap();
    return serde_json::from_str(&deserialised).unwrap();
}

struct Grid
{
    width : usize,
    height : usize,
    data : Vec<f32>,
}

const unit_w : f32 = 20.0;
impl Grid
{
    pub fn new(width : usize, height : usize) -> Grid
    {
        return Grid {width, height, data : vec![0.0; width * height]}
    }

    pub fn draw(&mut self)
    {
        if mouse_position().0 < unit_w * self.width as f32 && mouse_position().1 < unit_w * self.height as f32
        {
            if is_key_pressed(KeyCode::R)
            {
                self.data = vec![0.0; self.width * self.height];
            }

            if is_mouse_button_down(MouseButton::Left)
            {
                let mut pos : usize = 0;
                self.data[mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize] = 1.0;

                pos = mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize - 1;
                if mouse_position().0 > unit_w as f32 && self.data[pos] < 0.7
                {
                    self.data[pos] = 0.7;
                }
                
                pos = mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize + 1;
                if mouse_position().0 < (self.width - 1) as f32 * unit_w as f32 && self.data[pos] < 0.7
                {
                    self.data[pos] = 0.7;
                }

                pos = mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize - 28;
                if mouse_position().1 > unit_w as f32 && self.data[pos] < 0.7
                {
                    self.data[pos] = 0.7;
                }

                pos = mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize + 28;
                if mouse_position().1 < unit_w * (self.height - 1) as f32 && self.data[pos] < 0.7
                {
                    self.data[pos] = 0.7;
                }

            }
            else if is_mouse_button_down(MouseButton::Right)
            {
                self.data[mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize] = 0.0;
                // println!("{:?}", self.data);
            }
        }

        for i in 0..self.width
        {
            for j in 0..self.height
            {
                let color = self.data[j as usize * self.height + i as usize];
                draw_rectangle(i as f32 *unit_w, j as f32 * unit_w, unit_w, unit_w, Color { r: (color), g: (color), b: (color), a: (1.0) });
            }
        }
    }
}
#[derive(PartialEq)]
pub enum Screen
{
    HOME,
    DRAW,
    TRAIN,
    TEST,
}