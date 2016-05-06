import subprocess
import itertools

Ls = [15, 30, 60, 120]
degs = [1, 2, 3, 4, 5]
patchmattypes = ['dense', 'seqaij']
optfile = 'elasticity_opts.txt'

for L, deg, mt in itertools.product(Ls, degs, patchmattypes):
    subprocess.check_call(
        ['python',
         'elasticity2d.py',
         str(L),
         str(deg),
         '-options_file',
         'elasticity_opts.txt',
         '-scp_mat_type',
         mt,
         '-log_view',
         'ascii:elasticity-%d-%d-%s.py:ascii_info_detail' % (L, deg, mt)
         ]
    )
