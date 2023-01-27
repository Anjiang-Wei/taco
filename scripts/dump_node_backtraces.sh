#!/bin/bash

BINARY="$1"
out_dir="$2"

pstree -u $(whoami) -p > "$out_dir/pstree_$(hostname).log"
for pid in $(pgrep -u $(whoami) ${BINARY}); do
    gdb -p $pid -batch -quiet -ex "thread apply all bt" 2>&1 > "$out_dir/bt_$(hostname)_$pid.log"
    if grep realm_freeze "$out_dir/bt_$(hostname)_$pid.log"; then
        gdb -p $pid -batch -quiet -ex "gcore $(hostname)_$pid.core"
    fi
done
