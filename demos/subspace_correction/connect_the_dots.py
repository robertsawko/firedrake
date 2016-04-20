from pytools import generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam

def ref_ltg_2d(deg, eids):
    ts = gnitstam(deg, 2)
    t_to_i = dict((t, i) for (i, t) in enumerate(ts))

    # This triangulates the reference element.
    subtris = []
    for i in range(deg):
        for j in range(deg-i-1):
            sw = t_to_i[(i, j)]
            se = t_to_i[(i+1, j)]
            nw = t_to_i[(i, j+1)]
            ne = t_to_i[(i+1, j+1)]
            t1 = sorted([sw, se, nw])
            t2 = sorted([se, nw, ne])
            subtris.append(t1)
            subtris.append(t2)
        j=deg-i-1
        sw = t_to_i[(i, j)]
        se = t_to_i[(i+1, j)]
        nw = t_to_i[(i, j+1)]
        t1 = sorted([sw, se, nw])
        subtris.append(t1)

    # then, given an array ltgc giving the dofs on a cell,
    # one can create a new list of triangles/P1 dof map by
    # something like
    # dofmap_patch = np.zeros((len(subtris)*len(cells), sdim+1), np.int)
    # ntcur = 0
    # for c in cells:
    #     ltg_cur = ltg[c]
    #     for (i, st) in enumerate(subtris):
    #         dofmap_patch[i, :] = np.tage(ltg_c, st)
    #         ntcur += 1
    #
    # and can do this either on a local patch (e.g. CTD on patches
    # or for all of the cells in a mesh.


    # also need to get the element node locations and do that stuff
