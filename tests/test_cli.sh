#!/bin/sh
# End-to-end CLI checks. Usage: QM_BIN=build/quillm QM_FIXTURES=build/fixtures sh tests/test_cli.sh
set -u
BIN=${QM_BIN:-build/quillm}
FX=${QM_FIXTURES:-build/fixtures}
pass=0 fail=0
ok() { pass=$((pass+1)); }
bad() { fail=$((fail+1)); echo "FAIL: $1"; }

out1=$($BIN run $FX/small_q8_0.qlm -p "Once upon a time" -n 30 --seed 7 --temp 0.9 --top-k 40 --top-p 0.9 --no-stats)
out2=$($BIN run $FX/small_q8_0.qlm -p "Once upon a time" -n 30 --seed 7 --temp 0.9 --top-k 40 --top-p 0.9 --no-stats -t 1)
[ "$out1" = "$out2" ] && ok || bad "same seed, different thread count -> different text"
case "$out1" in "Once upon a time"*) ok ;; *) bad "output does not start with prompt" ;; esac
g_scalar=$($BIN run $FX/small_f32.qlm -p "the fox" -n 25 --temp 0 --isa scalar --no-stats)
for isa in avx2 avx512; do
  if g=$($BIN run $FX/small_f32.qlm -p "the fox" -n 25 --temp 0 --isa $isa --no-stats 2>/dev/null); then
    [ "$g" = "$g_scalar" ] && ok || bad "greedy output differs between scalar and $isa"
  fi
done
$BIN info $FX/odd_q4_0.qlm | grep -q "layers.0.wq *q4_0 *\[72, 72\]" && ok || bad "info output"
$BIN run /nonexistent.qlm -p x >/dev/null 2>&1; [ $? -eq 1 ] && ok || bad "missing file exit code"
$BIN bogus >/dev/null 2>&1; [ $? -eq 2 ] && ok || bad "usage exit code"
$BIN run $FX/small_f32.qlm -p x --isa nosuch >/dev/null 2>&1; [ $? -eq 1 ] && ok || bad "bad isa exit code"
stats=$($BIN run $FX/small_f32.qlm -p "hi" -n 5 2>&1 >/dev/null)
echo "$stats" | grep -q "tok/s" && ok || bad "stats line"
echo "tests/test_cli.sh: $pass passed, $fail failed"
[ $fail -eq 0 ]
