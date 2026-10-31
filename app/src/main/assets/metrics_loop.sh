#!/system/bin/sh
# 端末の負荷・温度・電力を一定間隔で出力する（PC 側の devtool.py monitor が読む）。使い方: metrics_loop.sh [間隔秒=0.5]  ※ root（su）で実行
INT=${1:-0.5}
TZ=/sys/class/thermal
ZHVX=""; ZHMX=""; ZCPU=""; ZGPU=""
for z in $TZ/thermal_zone*; do
  t=$(cat $z/type 2>/dev/null)
  case "$t" in
    nsphvx-*) ZHVX="$ZHVX $z/temp";;
    nsphmx-*) ZHMX="$ZHMX $z/temp";;
    cpu-*)    ZCPU="$ZCPU $z/temp";;
    gpuss-*)  ZGPU="$ZGPU $z/temp";;
  esac
done
avg() { s=0; n=0; for v in $(cat "$@" 2>/dev/null); do s=$((s+v)); n=$((n+1)); done; [ $n -gt 0 ] && echo $((s/n)) || echo 0; }
mx() { m=0; for v in $(cat "$@" 2>/dev/null); do [ $v -lt 100000 ] && [ $v -gt $m ] && m=$v; done; echo $m; }
while true; do
  echo "S $(date +%s%N | cut -c1-13)"
  grep '^cpu[0-9]' /proc/stat
  echo "F $(for c in 0 1 2 3 4 5 6 7; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq; done | tr '\n' ' ')"
  echo "G $(cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage | tr -d '% ') $(cat /sys/class/kgsl/kgsl-3d0/gpuclk)"
  echo "M $(grep MemAvailable /proc/meminfo | tr -s ' ' | cut -d' ' -f2) $(grep MemTotal /proc/meminfo | tr -s ' ' | cut -d' ' -f2)"
  echo "B $(cat /sys/class/power_supply/battery/current_now) $(cat /sys/class/power_supply/battery/voltage_now) $(cat /sys/class/power_supply/battery/status)"
  echo "T $(avg $ZHVX) $(avg $ZHMX) $(mx $ZCPU) $(avg $ZGPU)"
  echo "E"
  sleep $INT
done
