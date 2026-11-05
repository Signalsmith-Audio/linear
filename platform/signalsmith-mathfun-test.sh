#!/bin/sh
#CC=clang ./signalsmith-mathfun-test.sh
cd "$(dirname "$0")"
CC="${CC:-cc}" # Assumes GCC-like
OUT="${TMPDIR:-/tmp}/signalsmith-mathfun-test"
rm -f "$OUT.log"

run() {
	echo "Running: $*"
	if ! $CC -std=c99 -O2 "$@" signalsmith-mathfun-test.c -o "$OUT" -lm; then
		exit 1
	elif ! "$OUT" > "$OUT.log" 2>&1; then
		cat "$OUT.log"
		exit 1
	fi
}

# There are quite a few routing options, so we try and hit various combinations via a pseudo-random counter
ROUTE_INDEX=0
while [ $ROUTE_INDEX -le 20 ]; do
	for FAST in "" "-ffast-math"; do # fast-math alternates off/on
		run -DTEST_ROUTE_INDEX=$ROUTE_INDEX $FAST
		ROUTE_INDEX=$((ROUTE_INDEX + 1))
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
