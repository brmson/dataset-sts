#!/usr/bin/perl

while (<>) {
	if (/^\|--/) {
		print("\\hline\n");
		next;
	}
	s/^\| //;
	s/±/ \\pm/g;
	s/ \|$/\\\\/;
	@a = split(/ *\| */, $_);
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
