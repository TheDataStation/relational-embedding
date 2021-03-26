for i in {1..2}
do
    a=$(($i))
    echo $a
    python generate_sample.py --times $a
    cd .. 
    start=`date +%s`
    python data/preprocess_data.py --task sample 
    end=`date +%s` 
    python generate_graph.py --task sample --suffix $a
    end2=`date +%s` 
    cd node2vec 
    python src/main.py --task sample --suffix $a
    end3=`date +%s` 
    cd ..
    cd evaluation 
    runtime=$(($end3-$end2))
    runtime2=$(($end2-$end))
    runtime3=$(($end-$start))
    echo $runtime
    echo $runtime2 
    echo $runtime3
done 