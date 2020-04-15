./word2vec -train ../textified.txt -output word2vec_neg_20_i50.bin -cbow 1 -size 200 -window 10 -negative 20 -hs 0 -sample 1e-3 -threads 32 -binary 1 -iter 50
./word2vec -train ../textified.txt -output word2vec_neg_10_i30.bin -cbow 1 -size 200 -window 10 -negative 10 -hs 0 -sample 1e-3 -threads 20 -binary 1 -iter 30
