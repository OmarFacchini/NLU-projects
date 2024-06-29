# used to simplify the running of the tests
# --epochs is the number of epochs you want to train your model on
# --dropout is a flag used to say if we want to use the dropout layers or not
# --lr is the learning rate used for the training of the model
# --exp_name is the name that will be used for weights & bias and also the name of the best model that will be stored from the training
# note: ideally the exp_name format would be: optimDroplr if using dropout or optimlr if not
# so in case of SGD with dropout and lr 0.01 exp_name = SGDDrop001
# in case of same situation without dropout  exp_name = SGD001
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 100 --dropout 0 --lr 0.1 --exp_name a