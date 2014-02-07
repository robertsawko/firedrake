declare -a PROBLEMS=(MASS_2D MASS_3D HELMHOLTZ_2D HELMHOLTZ_3D HELMHOLTZ_XTR ADVDIFF_2D ADVDIFF_3D ADVDIFF_XTR BURGERS_2D BURGERS_3D BURGERS_XTR)

if [ $# -eq 0 ]
then
    echo "No arguments supplied. Specify input problem or ALL. Exiting..."
    exit
fi

if [ "$1" == "ALL" ]
then
    RUN_PROBLEMS=(${PROBLEMS[@]})
else
    RUN_PROBLEMS=("$1")
fi

if [ "$2" == "--time-kernel" ]
then
    TYPE=$2
else
    TYPE=""
fi

for p in "${RUN_PROBLEMS[@]}"
do
    for i in 1 2 3 4
    do
        python launcher.py ALL $p $i $TYPE
        python launcher.py LICM_AP_VECT_SPLIT $p $i $TYPE
        python launcher.py LICM_AP_TILE_SPLIT $p $i $TYPE
    done
done
