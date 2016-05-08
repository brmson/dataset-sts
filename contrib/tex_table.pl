#!/usr/bin/perl

while (<>) {
	chomp;
	if (/^\|--/) {
		print("\\hline\n");
		next;
	}
	s/^\| //;
	s/ *\|$//;
	s/±/ \\pm/g;
	@a = split(/ *\| */, $_, -1);
	my $m = shift @a;
	@b = ();
	for $r (@a) {
		if ($r eq '') {
			push @b, '';
		} elsif ($r =~ s/^\\pm//) {
			push @b, sprintf('$\quad^{\\pm%.3f}$', $r);
		} else {
			push @b, sprintf('$%.3f$', $r);
		}
	}
	print(join(' & ', $m, @b) . "\\\\\n");
}
