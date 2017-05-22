import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a model')
    parser.add_argument('model', action="store", type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sentences', nargs='*', default=None)

    args = parser.parse_args()

    print('Using model: %s' % args.model)
    print('Training: %s' % args.train)
    if args.sentences is not None:
        for value in args.sentences:
            print('processing sentence: %s' % value)

    if args.model == 'lstm':
        from lib_model.char_lstm import *

        network = LSTM()
        network.build()
        if args.train == True:
            network.train()
        if args.sentences is not None:
            network.predict_sentences(args.sentences)

    elif args.model == 'bidirectional_lstm':
        from lib_model.bidirectional_lstm import *

        network = LSTM()
        network.build()
        if args.train == True:
            network.train()
        if args.sentences is not None:
            network.predict_sentences(args.sentences)

