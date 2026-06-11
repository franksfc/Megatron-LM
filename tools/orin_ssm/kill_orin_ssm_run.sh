#!/bin/bash
set -euo pipefail

NODE_IPS=${NODE_IPS:-10.119.6.252,10.119.7.0}
RUN_PATTERN=${RUN_PATTERN:-}

base_pattern='[p]retrain_orin_ssm_mindspeed.py|[t]orchrun'
if [ -n "${RUN_PATTERN}" ]; then
    pattern="${RUN_PATTERN}|${base_pattern}"
else
    pattern="${base_pattern}"
fi

kill_on_host() {
    local host="$1"
    local cmd="pids=\$(pgrep -f '${pattern}' || true); if [ -n \"\$pids\" ]; then echo \"\$pids\" | xargs -r kill -9; fi; pgrep -fc '${pattern}' || true"
    if [ "${host}" = "local" ]; then
        bash -lc "${cmd}"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=5 "${host}" "${cmd}" || true
    fi
}

IFS=',' read -r -a nodes <<< "${NODE_IPS}"
for host in "${nodes[@]}"; do
    printf '%s ' "${host}"
    kill_on_host "${host}"
done
