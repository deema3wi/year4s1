python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py train --epochs 5 --batch-size 128 --lr 0.001
python main.py eval --checkpoint artifacts\mnist_cnn.pt
python main.py predict --checkpoint artifacts\mnist_cnn.pt --samples 8
python main.py predict --checkpoint artifacts\mnist_cnn.pt --image path\to\digit.png