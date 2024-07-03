use burn::{
    config::Config,
    module::Module,
    nn::{loss::CrossEntropyLoss, Linear, LinearConfig, Relu},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::dataset::LogicBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input: Linear<B>,
    // hidden: Linear<B>,
    output: Linear<B>,

    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_inputs: usize,
    num_hidden: usize,
    num_outputs: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            input: LinearConfig::new(self.num_inputs, self.num_hidden)
                // .with_bias(true)
                .init(device),
            // hidden: LinearConfig::new(self.num_hidden, self.num_hidden)
            //     .with_bias(true)
            //     .init(device),
            output: LinearConfig::new(self.num_hidden, self.num_outputs)
                // .with_bias(true)
                .init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    ///   - Input: [1, 0]
    ///   - Output [batch_size, 1]
    pub fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input.forward(inputs);
        let x = burn::tensor::activation::tanh(x);
        // let x = self.hidden.forward(x);
        // let x = burn::tensor::activation::tanh(x);
        let x = self.output.forward(x);
        burn::tensor::activation::sigmoid(x)
    }

    pub fn forward_step(&self, item: LogicBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output: Tensor<B, 2> = self.forward(item.inputs);

        let loss =
            CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}
impl<B: AutodiffBackend> TrainStep<LogicBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: LogicBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<LogicBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: LogicBatch<B>) -> ClassificationOutput<B> {
        self.forward_step(item)
    }
}
