#!/bin/sh
#CC=clang ./signalsmith-mathfun-test.sh
cd "$(dirname "$0")"
CC="${CC:-cc}" # Assumes GCC-like
OUT="${TMPDIR:-/tmp}/signalsmith-mathfun-test"

run() {
	echo "Running: $*"
	if ! $CC -std=c99 -O2 "$@" signalsmith-mathfun-test.c -o "$OUT" -lm; then
		exit 1
	elif ! "$OUT" > "$OUT.log" 2>&1; then
		cat "$OUT.log"
		exit 1
	fi
}

# Routes, at the default accuracy
for ROUTE in "" "-DTEST_ROUND" "-DTEST_ROUND_INT32" "-DTEST_FLOOR" "-DTEST_BARRIER" "-DTEST_FMA" "-DTEST_SQRT" "-DTEST_LOGEXP" "-DTEST_LOGEXP2" "-DTEST_BARRIER -DTEST_ROUND -DTEST_SQRT" "-DTEST_SMALL_INPUTS"; do
	for FAST in "" "-ffast-math"; do
		run $ROUTE $FAST
	done
done

# Accuracy levels, default route only
ACCURACY=1
while [ $ACCURACY -le 50 ]; do
	for FAST in "" "-ffast-math"; do
		run -DTEST_ACCURACY=$ACCURACY $FAST
	done
	ACCURACY=$((ACCURACY + 1))
done

rm -f "$OUT" "$OUT.log"
echo "\nAll passed\n"
