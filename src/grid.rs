use macroquad::prelude::*;
pub struct Grid
{
    pub width : usize,
    pub height : usize,
    pub data : Vec<f32>,
}

const unit_w : f32 = 20.0; //width in pixels of squares in the grida
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
                // functionality for drawing like brush. Pixel you are hovering over is brighter than it's 4 neighbours.
                let mut pos : usize;
                self.data[mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize] = 1.0;

                pos = mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize - 1;
                if mouse_position().0 > unit_w as f32 && self.data[pos] < 0.7 // make sure pixel is in grid and only make neighbour dimmer if it is not bright
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
                //erase pixel currently hovered over
                self.data[mouse_position().1 as usize / unit_w as usize * self.width + mouse_position().0 as usize / unit_w as usize] = 0.0;
            }
        }

        // draw grid
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