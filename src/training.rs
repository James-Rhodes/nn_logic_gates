use crate::dataset::{LogicBatcher, LogicDataset};
use crate::model::{Model, ModelConfig};
use burn::optim::AdamConfig;
use burn::train::metric::AccuracyMetric;
use burn::train::LearnerBuilder;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::metric::LossMetric,
};

static ARTIFACT_DIR: &str = "/tmp/nn_logic";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 40)]
    pub num_epochs: usize,
    #[config(default = 1024)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-1)]
    pub learning_rate: f64,
}

pub fn run<B: AutodiffBackend>(device: B::Device) -> Model<B> {
    // Config
    let optimizer = AdamConfig::new();

    let model_config = ModelConfig::new(2, 4, 2);
    let config = TrainingConfig::new(model_config, optimizer);
    B::seed(config.seed);

    // Define train/test datasets and dataloaders

    let train_dataset = LogicDataset::new(config.batch_size * 10);
    let test_dataset = LogicDataset::new(config.batch_size * 2);

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Test Dataset Size: {}", test_dataset.len());

    let batcher_train = LogicBatcher::<B>::new(device.clone());

    let batcher_test = LogicBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(train_dataset.len())
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(test_dataset.len())
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .clone()
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");

    model_trained
}
