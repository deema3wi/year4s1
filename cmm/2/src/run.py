import main

MODE = 'predict' # 'train' | 'predict'

EPOCHS = 8
BATCH_SIZE = 128
LR = 0.001 # Learning Rate

CHECKPOINT = 'artifacts/mnist_cnn.pt'
IMAGES_DIR = 'img'

DEVICE = 'auto' # 'cuda', 'cpu' або 'auto'


if __name__ == '__main__':
    if MODE == 'train':
        main.run_training(
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            lr=LR, 
            device_name=DEVICE, 
            save_path=CHECKPOINT
        )
        
        print("\nАвтоматичний запуск перевірки на зображеннях:")
        main.run_prediction(CHECKPOINT, IMAGES_DIR, DEVICE)

    elif MODE == 'predict':
        main.run_prediction(CHECKPOINT, IMAGES_DIR, DEVICE)
    else:
        print("Невідомий режим. Виберіть 'train' або 'predict'.")
