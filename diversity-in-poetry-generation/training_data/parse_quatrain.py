from uniformers.datasets import load_dataset
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str)
    args = parser.parse_args()

    output_dir="quatrain_data/" + args.lang 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    quatrains = load_dataset('quatrain', lang=args.lang, split="train")
    quatrains = quatrains['text']

    with open(output_dir + '/QuaTrain.txt', 'w') as f:
        for quatrain in quatrains:
            f.write('\n'.join(quatrain))
            f.write('\n\n')

