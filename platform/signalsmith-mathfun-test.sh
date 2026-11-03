#!/bin/sh
#CC=clang ./signalsmith-mathfun-test.sh
cd "$(dirname "$0")"
CC="${CC:-cc}" # Assumes GCC-like
OUT="${TMPDIR:-/tmp}/signalsmith-mathfun-test"
FAILED=""

run() {
	echo "Running: $*"
	if ! $CC -std=c99 -O2 "$@" signalsmith-mathfun-test.c -o "$OUT" -lm; then
		FAILED="$FAILED\n  (compile) $*"
	elif ! "$OUT" > "$OUT.log" 2>&1; then
		FAILED="$FAILED\n  $*"
		cat "$OUT.log"
		exit 1
	fi
}

# Routes, at the default accuracy
for ROUTE in "" "-DTEST_ROUND" "-DTEST_ROUND_INT32" "-DTEST_FLOOR" "-DTEST_BARRIER" "-DTEST_FMA" "-DTEST_LOGEXP" "-DTEST_LOGEXP2" "-DTEST_BARRIER -DTEST_ROUND"; do
	for FAST in "" "-ffast-math"; do
		run $ROUTE $FAST
	done
done

# Accuracy levels, default route only
ACCURACY=1
while [ $ACCURACY -le 50 ]; do
	for FAST in "" "-ffast-math"; do
		run -DTEST_ACCURACY=$ACCURACY $FAST
		ACCURACY=$((ACCURACY + 1))
	done
done

rm -f "$OUT" "$OUT.log"
if [ -n "$FAILED" ]; then
	echo "\nFailed:$FAILED\n"
	exit 1
fi
echo "\nAll passed\n"
