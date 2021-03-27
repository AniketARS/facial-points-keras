from models.model import get_model


if __name__ == '__main__':
    model = get_model()
    print(model.summary())
