#!/bin/bash

set -x

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

job="$1"
# job is the path to the job file (*.out)
scratch_dir="$2"
mkdir -p "$scratch_dir"

i=0
if command -v bjobs &> /dev/null
then
    # hosts="$(bjobs -o exec_host $job | tail -n -1 | tr ':' '\n' | sort -u | tr '\n' ' ')"
    hosts="$(grep "All hosts" ${job} | tr ' ' '\n' | tail -n +4 | sort -u | tr '\n' ' ')"
else
    hosts="$(sacct --json -j $job | jq -r '.jobs[0]|.steps[1]|.nodes|.list|@sh' | tr -d \')"

fi
for host in $hosts; do
    ssh $host bash "$root_dir/dump_node_backtraces.sh" "$scratch_dir" &
    if [[ $(( i % 200 )) == 0 ]]; then
        wait
    fi
    let i++
done

wait

