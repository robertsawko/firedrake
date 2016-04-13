from firedrake import *
import numpy
import collections
import itertools
import functools


class SubspaceCorrectionPrec(object):
    """Given a bilinear form, constructs a subspace correction preconditioner
    for it.  Currently, this is intended to approximate the solution
    of high-order Lagrange (eventually Bernstein as well)
    discretization by the solution of local problems on each vertex
    patch together with a global low-order discretization.

    :arg a:  A bilinear form defined in UFL
    :arg bcs: Optional strongly enforced boundary conditions
    """
    def __init__(self, a, bcs=None):
        self.a = a
        if bcs is None:
            bcs = ()
        try:
            bcs = tuple(bcs)
        except TypeError:
            bcs = (bcs, )
        nodes = set()
        for bc in bcs:
            nodes.update(bc.nodes)
        self.bc_nodes = nodes

        # one phase of the preconditioner involves restricting the problem
        # to the patch around each vertex and solving it.  So, we'll need
        # to grab that information from the mesh's plex object
        mesh = a.ufl_domain()
        self.mesh = mesh

        test, trial = a.arguments()

        V = test.function_space()
        assert V == trial.function_space()

        dof_section = V._dm.getDefaultSection()
        dm = mesh._plex

        # This includes halo vertices, we might need to filter some out
        vstart, vend = dm.getDepthStratum(0)

        # range for cells
        cstart, cend = dm.getHeightStratum(0)

        patches = []

        patch_faces = []
        # section from plex cells to firedrake cell numbers
        cell_numbering = mesh._cell_numbering
        # section for plex vertices to firedrake vertices
        vtx_numbering = mesh._vertex_numbering
        for v in range(vstart, vend):
            closure, orientation = dm.getTransitiveClosure(v, useCone=False)
            cells = closure[numpy.logical_and(cstart <= closure, closure < cend)]
            # find faces that are on boundary of cell patch
            boundary_faces = []
            for c in cells:
                faces = dm.getCone(c)
                for f in faces:
                    # Only select faces if they are not on the domain boundary
                    if dm.getLabelValue("exterior_facets", f) == 1:
                        continue
                    closure, _ = dm.getTransitiveClosure(f, useCone=True)
                    if v not in closure:
                        boundary_faces.append(f)
            patch_faces.append(boundary_faces)
            # Both of the vertices and cells are in plex numbering,
            patches.append(numpy.array([cell_numbering.getOffset(c)
                                        for c in cells]))

        # Have a functionspace V
        cell_node_map = V.cell_node_map().values
        # shape (ncell, ndof_per_cell)

        dof_patches = []
        glob_patches = []
        bc_masks = []
        from functools import partial
        for patch, faces in zip(patches, patch_faces):
            local = collections.defaultdict(partial(next, itertools.count()))
            dof_patch = numpy.empty((len(patch), cell_node_map.shape[-1]),
                                    dtype=int)
            bc_mask = numpy.zeros(dof_patch.shape, dtype=bool)
            for i, c in enumerate(patch):
                for j, dof in enumerate(cell_node_map[c, :]):
                    dof_patch[i, j] = local[dof]
                    # Mask out global dirichlet bcs
                    if dof in self.bc_nodes:
                        bc_mask[i, j] = True
            glob_patch = numpy.empty(dof_patch.max() + 1, dtype=int)
            for i, j in numpy.ndindex(dof_patch.shape):
                glob_patch[dof_patch[i, j]] = cell_node_map[patch[i], j]

            # Mask out dofs on boundary of patch
            for f in faces:
                closure, _ = dm.getTransitiveClosure(f, useCone=True)
                for p in closure:
                    off = dof_section.getOffset(p)
                    for j in range(dof_section.getDof(p)):
                        bc_mask[numpy.where(dof_patch == local[off + j])] = True

            glob_patches.append(glob_patch)
            dof_patches.append(dof_patch)
            bc_masks.append(bc_mask)

        for p, d, g, bcs in zip(patches, dof_patches, glob_patches, bc_masks):
            print p
            print d
            print g
            print bcs
            print
        
        
        
M = RectangleMesh(2, 2, 1.0, 1.0)
V = FunctionSpace(M, "CG", 2)
bcs = DirichletBC(V, 0, (1, 2, 3, 4))
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx

SCP = SubspaceCorrectionPrec(a, bcs=None)
