cimport petsc4py.PETSc as PETSc

cdef extern from "petsc.h":
    ctypedef long PetscInt
    ctypedef enum PetscBool:
        PETSC_TRUE, PETSC_FALSE
    int PetscMalloc1(PetscInt,void*)
    int PetscFree(void*)
    int PetscSortInt(PetscInt,PetscInt[])

cdef extern from "../src/sys/utils/hash.h":
    struct _PetscHashI
    ctypedef _PetscHashI* PetscHashI "PetscHashI"
    ctypedef long PetscHashIIter
    void PetscHashICreate(PetscHashI)
    void PetscHashIClear(PetscHashI)
    void PetscHashIDestroy(PetscHashI)
    void PetscHashISize(PetscHashI, PetscInt)
    void PetscHashIAdd(PetscHashI, PetscInt, PetscInt)
    void PetscHashIPut(PetscHashI, PetscInt, PetscHashIIter, PetscHashIIter)
    void PetscHashIMap(PetscHashI, PetscInt, PetscHashIIter)
    void PetscHashIHasKey(PetscHashI, PetscInt, PetscBool)
    void PetscHashIIterBegin(PetscHashI, PetscHashIIter)
    int PetscHashIIterAtEnd(PetscHashI, PetscHashIIter)
    void PetscHashIIterGetKeyVal(PetscHashI, PetscHashIIter, PetscInt, PetscInt)
    int PetscHashIGetKeys(PetscHashI, PetscInt *, PetscInt[])
    void PetscHashIIterNext(PetscHashI, PetscHashIIter)

cdef extern from "petscdmplex.h":
    int DMPlexGetHeightStratum(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt*)
    int DMPlexGetDepthStratum(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt*)

    int DMPlexGetCone(PETSc.PetscDM,PetscInt,PetscInt*[])
    int DMPlexGetConeSize(PETSc.PetscDM,PetscInt,PetscInt*)
    int DMPlexGetSupport(PETSc.PetscDM,PetscInt,PetscInt*[])

    int DMPlexGetTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])
    int DMPlexRestoreTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])

cdef extern from "petscdmlabel.h":
    struct _n_DMLabel
    ctypedef _n_DMLabel* DMLabel "DMLabel"
    int DMLabelCreateIndex(DMLabel, PetscInt, PetscInt)
    int DMLabelHasPoint(DMLabel, PetscInt, PetscBool*)

cdef extern from "petscdm.h":
    int DMGetLabel(PETSc.PetscDM,char[],DMLabel*)

cdef extern from "petscis.h":
    int PetscSectionGetOffset(PETSc.PetscSection,PetscInt,PetscInt*)
    int PetscSectionGetDof(PETSc.PetscSection,PetscInt,PetscInt*)
