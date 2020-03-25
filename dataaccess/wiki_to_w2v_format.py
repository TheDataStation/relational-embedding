from os.path import join
from os import listdir
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def read_files_format_join(path_to_files, output_file):

    all_files = [join(path_to_files, f) for f in listdir(path_to_files)]
    all_chunks = []

    for f in tqdm(all_files):
        with open(f, 'r', encoding='latin1') as g:
            lines = g.readlines()

        filtered_lines = []
        for l in lines:
            l = l.strip()
            if l == "":
                continue  # filter empty lines
            if l[0:2] == "==":
                if l[0:13] == "== References":
                    break
                elif l[0:17] == "== External links":
                    break
            else:
                filtered_lines.append(l)
        chunk = ' '.join(filtered_lines)

        chunk = chunk.replace(',', '')
        chunk = chunk.replace('.', ' ')
        chunk = chunk.replace('"', '')
        analyzed = nlp(chunk)
        tokens = [el.text for el in analyzed]

        token_chunk = ' '.join(tokens)
        all_chunks.append(token_chunk)
        # print(token_chunk)
        # exit()

    text_chunk = ' '.join(all_chunks)
    with open(output_file, 'w') as g:
        g.write(text_chunk)


if __name__ == "__main__":
    print("Wiki to w2v format")

    read_files_format_join("/Users/ra-mit/data/fabric/nba/text_files/", "/Users/ra-mit/data/fabric/nba/all_text.txt")
