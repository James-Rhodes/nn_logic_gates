use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    tensor::Tensor,
};

mod dataset;
mod model;
mod training;

// type Backend = Autodiff<Wgpu>;
type Backend = Autodiff<Wgpu>;
fn main() {
    let device = WgpuDevice::default();
    let model = training::run::<Backend>(device.clone());

    let data = dataset::CURR_INPUT.to_vec();
    for d in data {
        let input = Tensor::<Backend, 1>::from_floats([d.a as f32, d.b as f32], &device);
        let input: Tensor<Backend, 2> = input.unsqueeze();
        let output = model.forward(input).flatten::<1>(0, 1).into_scalar();
        println!("{}, {} -> {}, Expected: {}", d.a, d.b, output, d.output)
    }
}
