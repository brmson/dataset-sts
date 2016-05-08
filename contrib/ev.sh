#!/bin/sh
# ev.sh - process train logs, showing average val set performance
#
# Usage: ev.sh [--transfer] ID [CRIT]
#
# By default, MRR is used as a performance measure.  E.g. for hypev,
# pass ``QAccuracy`` as CRIT.
#
# Logs with filenames matching *.ID are scanned.  You MUST use --transfer
# in case of transfer learning, and make sure all your runs are complete,
# otherwise the pre-training performance info may get mixed in.

if [ x"$1" = x"--transfer" ]; then
	# skip odd samples, which are "pre-training" stats
	skip_odd=1
	shift
else
	skip_odd=
fi
id=$1
crit=$2
[ -n "$crit" ] || crit=MRR
list=$(echo -n '[';
cat *."$id" | grep -v '' | sed -rne 's/.*(\bval|_val|dev|trial).*'"$crit"': (raw |real )*([^ ,]*).*/\3/p' | sed 's/ *$//' |
		if [ "$skip_odd" = "1" ]; then awk 'NR % 2 == 0'; else cat; fi |
		tr '\n' ',' | sed 's/,/, /g';
		echo ']')
python -c"
import numpy as np
import scipy.stats as ss
def stat(r):
    bar = ss.t.isf((1-0.95)/2, len(r)-1) * np.std(r)/np.sqrt(len(r))
    print('%dx $id - %f (95%% [%f, %f]):' % (len(r), np.mean(r), np.mean(r) - bar, np.mean(r) + bar))
stat($list)
"
echo
echo '```'
echo $(ls *."$id" | head -n1) etc.
echo "$list"
echo '```'
