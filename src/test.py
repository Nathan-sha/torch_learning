from src.dl.data.vision import make_mnist_datasets



train_dataset, test_dataset = make_mnist_datasets()

print(train_dataset, test_dataset)

print(train_dataset[0].shape, test_dataset[0].shape)