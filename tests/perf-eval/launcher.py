import pyop2.ir.ast_plan as ap

import sys
import cProfile
import pstats
import os
import StringIO

opts = ['NORMAL', 'LICM', 'LICM_AP', 'LICM_AP_TILE', 'LICM_AP_VECT', 'LICM_AP_VECT_EXT']
problems = ['HELMHOLTZ_2D', 'HELMHOLTZ_3D', 'BURGERS']


if len(sys.argv) in [2, 3]:
    if len(sys.argv) == 3 and sys.argv[2] in problems:
        problem = sys.argv[2]
    else:
        problem = problems[0]

    if sys.argv[1] == '--help':
        _opts = "\n".join(["- %s" % i for i in opts])
        print "Possible optimisations are:\n" + _opts
        sys.exit(0)
    else:
        opt = sys.argv[1] if sys.argv[1] in opts else 'ALL'
else:
    opt = 'ALL'
    problem = problems[0]

if problem == 'HELMHOLTZ_2D':
    from helmholtz_2d import run_helmholtz as run_prob
    print "Running Helmholtz 2D problem"
elif problem == 'HELMHOLTZ_3D':
    from helmholtz_3d import run_helmholtz as run_prob
    print "Running Helmholtz 3D problem"
elif problem == 'BURGERS':
    from burgers import run_burgers as run_prob
    print "Running Burgers problem"


problem = problem.lower()


### RUN PROBLEM ###

mesh_size = 4
poly_orders = [1, 2, 3, 4 ,5]

for poly_order in poly_orders:
    results = []
    digest = open ("digest_%s_p%d.txt" % (problem, poly_order),"w")

    print "*****************************************"

    if opt in ['ALL', 'NORMAL']:
        print "Run NORMAL %s p%d" % (problem, poly_order)
        cProfile.run("results.append(run_prob(mesh_size, poly_order))", 'cprof.NORMAL.dat')
        digest.write("*****************************************\n")
        p = pstats.Stats('cprof.NORMAL.dat')
        stat_parser = StringIO.StringIO()
        p.stream = stat_parser
        p.sort_stats('time').print_stats(10)
        digest.write(stat_parser.getvalue())
        digest.write("*****************************************\n\n")
        os.remove('cprof.NORMAL.dat')


    if opt in ['ALL', 'LICM']:
        print "Run LICM %s p%d" % (problem, poly_order)
        os.environ['PYOP2_IR_LICM'] = 'True'
        os.environ['PYOP2_IR_AP'] = 'False'
        os.environ['PYOP2_IR_TILE'] = 'False'
        os.environ['PYOP2_IR_VECT'] = 'None'
        cProfile.run("results.append(run_prob(mesh_size, poly_order))", 'cprof.LICM.dat')
        digest.write("*****************************************\n")
        p = pstats.Stats('cprof.LICM.dat')
        stat_parser = StringIO.StringIO()
        p.stream = stat_parser
        p.sort_stats('time').print_stats(10)
        digest.write(stat_parser.getvalue())
        digest.write("*****************************************\n\n")
        os.remove('cprof.LICM.dat')


    if opt in ['ALL', 'LICM_AP']:
        print "Run LICM+ALIGN+PADDING %s p%d" % (problem, poly_order)
        os.environ['PYOP2_IR_LICM'] = 'True'
        os.environ['PYOP2_IR_AP'] = 'True'
        os.environ['PYOP2_IR_TILE'] = 'False'
        os.environ['PYOP2_IR_VECT'] = '((%s, 3), "avx", "intel")' % ap.AUTOVECT
        cProfile.run("results.append(run_prob(mesh_size, poly_order))", 'cprof.LICM_AP.dat')
        digest.write("*****************************************\n")
        p = pstats.Stats('cprof.LICM_AP.dat')
        stat_parser = StringIO.StringIO()
        p.stream = stat_parser
        p.sort_stats('time').print_stats(10)
        digest.write(stat_parser.getvalue())
        digest.write("*****************************************\n\n")
        os.remove('cprof.LICM_AP.dat')


    if opt in ['ALL', 'LICM_AP_VECT']:
        print "Run LICM+ALIGN+PADDING+VECT %s p%d" % (problem, poly_order)
        os.environ['PYOP2_IR_LICM'] = 'True'
        os.environ['PYOP2_IR_AP'] = 'True'
        os.environ['PYOP2_IR_TILE'] = 'False'
        os.environ['PYOP2_IR_VECT'] = '((%s, 2), "avx", "intel")' % ap.V_OP_UAJ
        cProfile.run("results.append(run_prob(mesh_size, poly_order))", 'cprof.LICM_AP_VECT.dat')
        digest.write("*****************************************\n")
        p = pstats.Stats('cprof.LICM_AP_VECT.dat')
        stat_parser = StringIO.StringIO()
        p.stream = stat_parser
        p.sort_stats('time').print_stats(10)
        digest.write(stat_parser.getvalue())
        digest.write("*****************************************\n\n")
        os.remove('cprof.LICM_AP_VECT.dat')


    if opt in ['ALL', 'LICM_AP_VECT_EXT']:
        print "Run LICM+ALIGN+PADDING+VECT+EXTRA %s p%d" % (problem, poly_order)
        os.environ['PYOP2_IR_LICM'] = 'True'
        os.environ['PYOP2_IR_AP'] = 'True'
        os.environ['PYOP2_IR_TILE'] = 'False'
        os.environ['PYOP2_IR_VECT'] = '((%s, 3), "avx", "intel")' % ap.V_OP_UAJ_EXTRA
        cProfile.run("results.append(run_prob(mesh_size, poly_order))", 'cprof.LICM_AP_VECT_EXT.dat')
        digest.write("*****************************************\n")
        p = pstats.Stats('cprof.LICM_AP_VECT_EXT.dat')
        stat_parser = StringIO.StringIO()
        p.stream = stat_parser
        p.sort_stats('time').print_stats(10)
        digest.write(stat_parser.getvalue())
        digest.write("*****************************************\n\n")
        os.remove('cprof.LICM_AP_VECT_EXT.dat')


    digest.close()
