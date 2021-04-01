for i in {1..20}
do
    a=$(($i*5))
    echo $a
    python generate_sample.py --times $a
    cd .. 
    start=`date +%s`
    python data/preprocess_data.py --task sample 
    end_preprocess=`date +%s` 
    python generate_graph.py --task sample --suffix $a
    end_graph=`date +%s` 
    cd ProNE 
    python proNE.py --task sample --suffix $a
    end_train=`date +%s` 
    cd ..
    cd data 
    runtime=$(($end_train-$end_graph))
    runtime2=$(($end_graph-$end_preprocess))
    runtime3=$(($end_preprocess-$start))
    echo $runtime
    echo $runtime2 
    echo $runtime3
done 