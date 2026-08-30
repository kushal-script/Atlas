# The optical preset, and a physical mismatch that turned out not to matter

## The mismatch was real

The optical generator blurs the search capture by 1.8 to 4.0 pixels at 10 nm per
pixel, so 18 to 40 nm. The shipped optical preset searched a nominal pose bank of
2 and 8 nm and a pose grid bank of 4, 9, 16 and 25 nm, the latter never set for
optical at all but inherited from the secondary electron preset. The pose search
was therefore hunting blur levels the optical search image does not contain, and
the grid's ceiling sat below the range where the signal actually lives.

## And correcting it changes nothing

| configuration | credit | within 5 px | median error |
| --- | --- | --- | --- |
| shipped, psf 2,8 and wide 4,9,16,25 | 0.430 | 43 percent | 100.44 px |
| wide extended to 40 | 0.430 | 43 | 96.28 |
| wide matched, 16 to 40 | 0.430 | 43 | 96.51 |
| psf matched, 16 to 32 | 0.427 | 43 | 100.89 |
| both matched | 0.430 | 43 | 96.40 |
| both widened further | 0.423 | 43 | 100.97 |

Six configurations, credit flat to three decimals and two of them marginally
worse. The median error moves a little, so the bank change is altering which
hypotheses get evaluated; it simply never moves a pair across a credit boundary.
The failures are not marginal cases sitting just outside the five pixel tier,
they are the same wholesale misses, unmoved.

## The headroom is real too, which is what makes this worth recording

An oracle probe handing the matcher the true zoom and rotation finds the true
site winning the correlation on 62 percent of the sixty pair suite and 74 percent
of the seventy two pair suite, against 43 and 64 percent achieved. So roughly
nineteen points of the weaker suite are reachable in principle by a better pose
search, and the blur bank is not the way to reach them. That is a different
finding from the degraded secondary electron set, where the oracle ceiling is
34 percent against 20 achieved and most of the remaining loss is information
that was never captured.

Recording the negative matters because the mismatch is the obvious thing to try:
anyone reading the preset will notice that the banks do not span the optical
blur, and this is the measurement showing that fixing it is not where the Set D
points are.
