filenames=("cannonMM-cuda/mappings" "pummaMM-cuda/mappings" "summaMM-cuda/mappings" \
    "johnsonMM-cuda/mappings" "solomonikMM-cuda/mappings" "cosma-cuda/mappings")
for f in ${filenames[@]}; do
    cp $f $f.py
    echo $f
    cloc $f.py | tail -n 2 | head -n 1 | tr -s ' ' | cut -d' ' -f5
done

cmappers=("../legion/include/shard.h" "../legion/include/taco_mapper.h" "../legion/src/taco_mapper.cpp")
for f in ${cmappers[@]}; do
    echo $f
    cloc $f | tail -n 2 | head -n 1 | tr -s ' ' | cut -d' ' -f5
done
