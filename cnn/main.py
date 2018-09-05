from pipeline import Pipeline
import torchvision.transforms as transforms


def main():
    pipe = Pipeline(
        input_size=28 * 28, hidden_size=500, output_size=10,
        data_dir='../data', batch_size=100, transform=transforms.ToTensor(),
        log_interval=50, epochs=10, load_model=True
        )
    pipe.run()


if __name__ == '__main__':
    main()
