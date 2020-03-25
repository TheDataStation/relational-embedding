import argparse
from textification import textify_relation as tr

if __name__ == "__main__":
    print("Textify relation")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to collection of relations')
    parser.add_argument('--relation_strategy', default='row_and_col', type=str, choices=tr.RELATION_STRATEGY,
                        help='Strategy to capture relationships, row, col, or row_and_col')
    parser.add_argument('--integer_strategy', default='skip', type=str, choices=tr.INTEGER_STRATEGY,
                        help='strategy to determine how to deal with integers')
    parser.add_argument('--grain', default='cell', type=str, choices=tr.TEXTIFICATION_GRAIN,
                        help='choose whether to process individual tokens (token), or entire cells (cell)')
    parser.add_argument('--output', type=str, default='textified.txt', help='path where to write the output file')
    parser.add_argument('--output_format', default='sequence_text', choices=tr.OUTPUT_FORMAT,
                        help='sequence_text or windowed_text')
    parser.add_argument('--debug', default=False, help='whether to run program in debug mode or not')

    args = parser.parse_args()

    tr.main(args)
