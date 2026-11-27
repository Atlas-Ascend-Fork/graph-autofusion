# High-accuracy VF profiling

This directory contains the Ascend 950 VF profiling probe and fitting helper.

After loading the CANN environment, run the following commands from this directory. The commands create a temporary experiment directory.

    tool=$(pwd)
    out=$(mktemp -d)
    cmake -S . -B "$out/build" -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH"
    cmake --build "$out/build" -j 4
    cd "$out"
    msprof --application="$out/build/att_vf_dependency" --output="$out/profile-v2" \
      --aic-mode=task-based --aic-metrics=PipeUtilization > "$out/profile-v2.log" 2>&1
    python3 "$tool/fit.py"

fit.py reads profile-v2 and profile-v2.log from the current directory and writes fit.json.
