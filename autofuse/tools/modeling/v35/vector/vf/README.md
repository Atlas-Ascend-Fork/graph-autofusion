# v35 高精度 VF 采集

在 Ascend 950 上加载 CANN 环境后，从本目录执行。`out` 指向新的实验目录。

```bash
tool=$(pwd)
out=$(mktemp -d)
cmake -S . -B "$out/build" -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH"
cmake --build "$out/build" -j 4
cd "$out"
msprof --application="$out/build/att_vf_dependency" --output="$out/profile-v2" \
  --aic-mode=task-based --aic-metrics=PipeUtilization > "$out/profile-v2.log" 2>&1
python3 "$tool/fit.py"
```

`fit.py` 读取当前目录的 `profile-v2` 和 `profile-v2.log`，并生成 `fit.json`。
