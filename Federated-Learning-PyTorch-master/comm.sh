python src/federated_main.py --model=cnn --dataset=mnist --gpu=cuda:0 --iid=0 --epochs=100 --select_edges=3 --num_users=50
python src/federated_main.py --model=cnn --dataset=cifar --gpu=cuda:0 --iid=0 --epochs=300 --select_edges=3 --num_users=50 --lr=0.02
python src/federated_main.py --model=lstm --dataset=shakespeare --gpu=cuda:0 --iid=0 --epochs=100 --select_edges=3 --lr=1.2

