use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{
            transform::{PartialDataset, ShuffledDataset},
            Dataset, InMemDataset,
        },
    },
    prelude::*,
};

// Representing boolean and
#[allow(unused)]
pub const BOOL_AND: &[LogicItem] = &[
    LogicItem::new(0, 0, 0),
    LogicItem::new(0, 1, 0),
    LogicItem::new(1, 0, 0),
    LogicItem::new(1, 1, 1),
];

// Representing boolean or
#[allow(unused)]
pub const BOOL_OR: &[LogicItem] = &[
    LogicItem::new(0, 0, 0),
    LogicItem::new(0, 1, 1),
    LogicItem::new(1, 0, 1),
    LogicItem::new(1, 1, 1),
];

// Representing boolean xor
#[allow(unused)]
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
    pub a: usize,
    pub b: usize,
    pub output: usize,
}

impl LogicItem {
    pub const fn new(a: usize, b: usize, output: usize) -> Self {
        Self { a, b, output }
    }
}

type ShuffledData = ShuffledDataset<InMemDataset<LogicItem>, LogicItem>;
type PartialData = PartialDataset<ShuffledData, LogicItem>;

pub struct LogicDataset {
    dataset: PartialData,
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
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let data = CURR_INPUT.to_vec();
        let amount = 100;
        let data = data
            .iter()
            .cloned()
            .cycle()
            .take(data.len() * amount)
            .collect();
        let dataset: InMemDataset<LogicItem> = InMemDataset::new(data);

        let dataset = ShuffledDataset::with_seed(dataset, 100);

        let len = dataset.len();
        let filtered_dataset = match split {
            "train" => PartialData::new(dataset, 0, len * 8 / 10),
            "test" => PartialData::new(dataset, len * 8 / 10, len),
            _ => panic!("Invalid split type"), // Handle unexpected split types
        };

        Self {
            dataset: filtered_dataset,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LogicBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct LogicBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
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
            let input_tensor =
                Tensor::<B, 1>::from_floats([item.a as f32, item.b as f32], &self.device);

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.output as f32], &self.device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        LogicBatch { inputs, targets }
    }
}
