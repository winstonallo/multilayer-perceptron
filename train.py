from data import TrainingData


class Trainer:

    def __init__(self):
        self.inputs, self.targets = TrainingData().get_data()


if __name__ == "__main__":
    trainer = Trainer()
    print(trainer.inputs)
    print(trainer.targets)
