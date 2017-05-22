import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a model')
    parser.add_argument('model', action="store", type=str, dest='model')
    args = parser.parse_args()

    if args.model == 'LSTM':
        from lib_models.char_lstm import *
        network = LSTM()
        network.build()
        network.train()
        
    elif args.model == 'bidirectional_LSTM':
        from lib_models.bidirectional_lstm import *
        network = LSTM()
        network.build()
        network.train()
        
