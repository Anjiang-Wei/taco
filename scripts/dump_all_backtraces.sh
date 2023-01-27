#!/bin/bash

set -x

#                                     $path_to_jobfile                $binary_name  $output_dir
# bash scripts/dump_all_backtraces.sh build/cannonMM-cuda/XXXXXXX.out cannonMM-cuda /g/g92/wei8/release_exp/taco/hanglog

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

job="$1"
binary="$2"
# job is the path to the job file (*.out)
scratch_dir="$3"
mkdir -p "$scratch_dir"

i=0
if command -v bjobs &> /dev/null
then
    # hosts="$(bjobs -o exec_host $job | tail -n -1 | tr ':' '\n' | sort -u | tr '\n' ' ')"
    # hosts="$(grep "All hosts" ${job} | tr ' ' '\n' | tail -n +4 | sort -u | tr '\n' ' ')" # does not work for higher node counts, echo fail
    hosts="$(python3 scripts/find_lassen_nodes.py ${job})"
    echo $hosts
else
    hosts="$(sacct --json -j $job | jq -r '.jobs[0]|.steps[1]|.nodes|.list|@sh' | tr -d \')"
fi

for host in $hosts; do
    ssh $host bash "$root_dir/dump_node_backtraces.sh" "$binary" "$scratch_dir" &
    if [[ $(( i % 10 )) == 0 ]]; then
        wait
    fi
    let i++
done
echo "Trying to dump for ${i} nodes"
wait

