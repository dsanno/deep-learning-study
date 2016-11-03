import argparse
import json
import numpy as np
import os
import six

pickle = six.moves.cPickle

def count_words(sentences):
    word_counts = {}
    for tokens in sentences:
        for token in tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
    return word_counts

def tokens_to_ids(tokens, word_ids, unknown_id):
    return [word_ids[token] if token in word_ids else unknown_id for token in tokens]

def make(data_dir, train_file, test_file, min_count):
    with open(os.path.join(data_dir, train_file)) as f:
        sentences = [tokens.split() for tokens in f]
    word_counts = count_words(sentences)
    words = [w for w, c in word_counts.items() if c >= min_count]
    begin_id = 0
    end_id = 1
    unknown_id = 2
    words = ['<S>', '</S>', '<UNK>'] + words
    word_ids = {w: i for i, w in enumerate(words)}
    train_token_ids = [tokens_to_ids(tokens, word_ids, unknown_id) for tokens in sentences]
    with open(os.path.join(data_dir, test_file)) as f:
        sentences = [tokens.split() for tokens in f]
    test_token_ids = [tokens_to_ids(tokens, word_ids, unknown_id) for tokens in sentences]
    return {
        'words': words,
        'train': train_token_ids,
        'test': test_token_ids,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make translation dataset')
    parser.add_argument('input_path', type=str, help='Input json file path')
    parser.add_argument('output_path', type=str, help='Output file path')
    parser.add_argument('--data-dir', '-d', type=str, default='.', help='Data file directory')
    parser.add_argument('--min-count', '-m', type=int, default=1, help='Minimum count for training words')
    args = parser.parse_args()

    dataset = {}
    with open(args.input_path) as f:
        config = json.load(f)
    for lang, lang_config in config.items():
        train_file = lang_config['train']
        test_file = lang_config['test']
        dataset[lang] = make(args.data_dir, train_file, test_file, args.min_count)
    with open(args.output_path, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
