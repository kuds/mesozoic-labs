# F2 preparation environment (sourced by every command; paths read from the pinned tools, never typed)
export F=/home/user/work/pc_study/f2prep
export ST=/home/user/work/pc_study
export S2=$(sed -n 's/^S2 = "\(.*\)"$/\1/p' $ST/tools/pc/r10round_common.py)
export S1=$(sed -n 's/^S1 = Path("\(.*\)")$/\1/p' $ST/tools/pc/pc_common.py)
export PY=$S2/venv/bin/python
export TMPDIR=$F/tmp MPLCONFIGDIR=$F/mpl MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GIT_OPTIONAL_LOCKS=0
export HEADSHA=1dfe2327f773007fa7fd68460dc0c53740936ff9
cd $F/work
