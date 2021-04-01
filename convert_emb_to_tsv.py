
import os
import sys
import logging
import argparse

import gensim
from gensim import utils
from tqdm import tqdm
from token_dict import TokenDict


logger = logging.getLogger(__name__)


def word2vec2tensor(task, suffix, binary=False):
    suffix = "" if suffix == "" else "_" + suffix
    word2vec_model_path = "./{}{}.emb".format(task, suffix)
    tensor_path = "./{}{}".format(task, suffix)
    outfiletsv = tensor_path + '_tensor.tsv'
    outfiletsvmeta = tensor_path + '_metadata.tsv'
    
    if "_sparse" in suffix or "_spectral" in suffix or "_restart" in suffix:
        suffix = "_".join(suffix.split("_")[:-1])
    dictionary_path = "../../graph/{}/{}{}.dict".format(task, task, suffix)
    cc = TokenDict()
    cc.load(dictionary_path)
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=binary)
    logger.info("Dictionary and Model Loaded")
    
    with utils.open(outfiletsv, 'wb') as file_vector, utils.open(outfiletsvmeta, 'wb') as file_metadata:
        for word in tqdm(model.index2word):
            word_meta = cc.getTokenForNum(word)
            file_metadata.write(gensim.utils.to_utf8(word_meta) + gensim.utils.to_utf8('\n'))
            vector_row = '\t'.join(str(x) for x in model[word])
            file_vector.write(gensim.utils.to_utf8(vector_row) + gensim.utils.to_utf8('\n'))

    logger.info("2D tensor file saved to %s", outfiletsv)
    logger.info("Tensor metadata file saved to %s", outfiletsvmeta)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='sample',
                        help='task to generate relation from')

    parser.add_argument('--suffix',
                        type=str,
                        default='',
                        help='a suffix to identify')

    args = parser.parse_args()

    logger.info("running %s", ' '.join(sys.argv))
    word2vec2tensor(args.task, args.suffix) 
    logger.info("finished running %s", os.path.basename(sys.argv[0]))
