use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{transform::ShuffledDataset, Dataset, InMemDataset},
    },
    prelude::*,
};

// Representing boolean and
pub const BOOL_AND: &[LogicItem] = &[
    LogicItem::new(0, 0, 0),
    LogicItem::new(0, 1, 0),
    LogicItem::new(1, 0, 0),
    LogicItem::new(1, 1, 1),
];

// Representing boolean or
pub const BOOL_OR: &[LogicItem] = &[
    LogicItem::new(0, 0, 0),
    LogicItem::new(0, 1, 1),
    LogicItem::new(1, 0, 1),
    LogicItem::new(1, 1, 1),
];

// Representing boolean xor
pub const BOOL_XOR: &[LogicItem] = &[
    LogicItem::new(0, 0, 0),
    LogicItem::new(0, 1, 1),
    LogicItem::new(1, 0, 1),
    LogicItem::new(1, 1, 0),
];

pub const CURR_INPUT: &[LogicItem] = BOOL_XOR;

// 0 = false, 1 = true
#[derive(Clone, Debug)]
pub struct LogicItem {
    pub a: f32,
    pub b: f32,
    pub output: f32,
}

impl LogicItem {
    pub const fn new(a: usize, b: usize, output: usize) -> Self {
        Self {
            a: a as f32,
            b: b as f32,
            output: output as f32,
        }
    }
}

type ShuffledData = ShuffledDataset<InMemDataset<LogicItem>, LogicItem>;

pub struct LogicDataset {
    dataset: ShuffledData,
}

impl Dataset<LogicItem> for LogicDataset {
    fn get(&self, index: usize) -> Option<LogicItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl LogicDataset {
    pub fn new(num_items: usize) -> Self {
        let data = (0..num_items)
            .map(|_| {
                let a: f32 = rand::random();
                let b: f32 = rand::random();

                let idx = 2 * (a > 0.5) as usize + (b > 0.5) as usize;

                let output = CURR_INPUT[idx].output as f32;

                LogicItem { a, b, output }
            })
            .collect();
        let dataset: InMemDataset<LogicItem> = InMemDataset::new(data);

        let dataset = ShuffledDataset::with_seed(dataset, 42);

        Self { dataset }
    }
}

#[derive(Clone, Debug)]
pub struct LogicBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct LogicBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> LogicBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<LogicItem, LogicBatch<B>> for LogicBatcher<B> {
    fn batch(&self, items: Vec<LogicItem>) -> LogicBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats([item.a, item.b], &self.device);

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0).to_device(&self.device);

        let targets: Vec<u32> = items
            .iter()
            .map(|item| if item.output > 0.5 { 1 } else { 0 })
            .collect();

        let targets: Tensor<B, 1, Int> = Tensor::from_data(
            Data::new(targets, [items.len()].into()).convert(),
            &self.device,
        )
        .to_device(&self.device);

        LogicBatch { inputs, targets }
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;

    #[test]
    fn batching() {
        type Backend = Wgpu;
        let device = burn::backend::wgpu::WgpuDevice::default();

        let amount = 4;
        let dataset = LogicDataset::new(amount);

        let mut items = vec![];
        for i in 0..amount {
            items.push(dataset.get(i).unwrap());
        }

        let batcher = LogicBatcher::<Backend>::new(device);

        let batch = batcher.batch(items);

        println!("batch input dims: {:?}", batch.inputs.dims());
        assert_eq!(batch.inputs.dims(), [4, 2]);
        println!("batch output dims: {:?}", batch.targets.dims());
        assert_eq!(batch.targets.dims(), [4]);
        println!("batch input: {:?}", batch.inputs.into_data());
        println!("batch output: {:?}", batch.targets.into_data());
    }
}
