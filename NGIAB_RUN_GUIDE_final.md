# Running NGIAB (ngen + t-route) fast on the cluster

A step-by-step guide to running the NextGen In A Box (NGIAB) container as a Slurm batch job on the
`amd` partition, based on a series of timed test runs with `gage-02361000` (227 catchments,
2020-01-01 to 2022-12-31, hourly).

**Bottom line:** with one node, one MPI process per catchment (227), a small polling fix and
Open MPI's built-in shared-memory transport (Step 5), a full 3-year run takes **about 26 seconds**,
down from **more than 2.5 hours** in serial mode.
Every optimization was checked to give **byte-identical outputs**.

---

## Contents

1. [What we learned](#1-what-we-learned)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 – One-time setup](#step-1--one-time-setup)
4. [Step 2 – Check the gage's input data](#step-2--check-the-gages-input-data)
5. [Step 3 – Choose the number of MPI processes](#step-3--choose-the-number-of-mpi-processes)
6. [Step 4 – Basic batch run](#step-4--basic-batch-run)
7. [Step 5 – Fast run with the polling fix (recommended)](#step-5--fast-run-with-the-polling-fix-recommended)
8. [Step 6 – Check the run and its outputs](#step-6--check-the-run-and-its-outputs)
9. [Step 7 – Optional settings](#step-7--optional-settings)
10. [Step 8 – Optional: multi-node runs](#step-8--optional-multi-node-runs)
11. [Step 9 – Optional: profiling](#step-9--optional-profiling)
12. [Benchmark results](#benchmark-results)
13. [Output comparison](#output-comparison)
14. [Troubleshooting](#troubleshooting)
15. [File reference](#file-reference)

---

## 1. What we learned

| Question | Answer |
|---|---|
| Is `OMP_NUM_THREADS=1` the reason the run is slow? | **No.** The models run in parallel with MPI processes, not OpenMP threads. Keep it at 1. |
| Why was the original run slow? | The interactive container menu was used in **serial mode**: one process for all 227 catchments. |
| Is t-route (routing) the bottleneck? | **No.** Routing takes about 20 s. |
| Is file I/O the bottleneck? | **No.** Running from a RAM disk saved about 2 s, and writing only `Q_OUT` saved nothing. I/O is about 4% of the run. |
| Does vol-async or C++ async I/O help? | **No.** There is almost no I/O time to hide. |
| What actually limited the speed? | ngen's processes were **idle about 94% of the model run**. While waiting for flows from another process, ngen checks, sleeps a fixed **1 ms**, then checks again. Real work is about 0.23 ms per simulated hour. |
| What fixed it? | A tiny `LD_PRELOAD` library that shortens that 1 ms sleep. The model run went from **91 s to 6 s**, with **byte-identical outputs**. |
| Why did 227 processes on one node fail? | Open MPI's default communication library, **UCX**, could not connect all process pairs once there were more than 128 on a node (`Destination is unreachable`), with one container or many. Switching to Open MPI's own shared-memory transport (`ob1`/`vader`) fixes it, and the run drops to **26 s** with identical outputs (Step 5.6). |
| Do more nodes help? | **Not for this gage.** The fix also works on 2 and 4 nodes (about 50 s instead of 2–3 min, identical outputs), but one node with the container's own `mpirun` is faster (35 s with 128 processes, 26 s with 227 and the shared-memory transport): routing stays ~20 s on one process, and starting one container per process adds overhead. |

---

## 2. Prerequisites

- Cluster account with access to the `amd` partition. Nodes have 2 × 128 cores (512 hardware threads),
  about 1.5 TB RAM, a 7 TB local disk (`/local`) and a 756 GB RAM disk (`/dev/shm`).
- The container image: `~/ngiab_openmpi.sif` (contains ngen 0.3.0, t-route, Open MPI 4.1.1, HDF5 1.12.1).
- Preprocessed gage data, laid out like this:

```
~/ngiab_preprocess_output/<GAGE_ID>/
├── config/
│   ├── <GAGE_ID>_subset.gpkg        # hydrofabric (catchments + nexuses)
│   ├── realization.json             # model setup, time period, forcing path
│   ├── troute.yaml                  # routing setup
│   └── cat_config/                  # per-catchment configs (CFE, NOAH-OWP-M)
├── forcings/forcings.nc             # meteorological forcing
├── partitions_<N>.json              # MPI partition files (created automatically)
└── outputs/                         # ngen/ (CSV) and troute/ (NetCDF) results
```

> **Important:** the container's start script picks up **every** `*.gpkg` file and every
> `*realization*.json` file under `config/`. Keep exactly one of each there; put backups elsewhere.

---

## Step 1 – One-time setup

### 1.1 Create the log folder

All job scripts write logs to `logs/` relative to where you submit, so always submit from your home directory.

```bash
cd ~
mkdir -p logs
```

### 1.2 The job scripts

These scripts live in your home directory:

| Script | Use it for |
|---|---|
| `run_ngiab.sh` | Basic single-node run, writes to the gage's `outputs/` folder |
| `run_ngiab_fastio.sh` | **Recommended.** Single-node run on fast local storage, optional polling fix |
| `run_ngiab_multinode.sh` | Runs across several nodes (not needed for small gages) |
| `run_ngiab_perf.sh` | Profiling with `perf` |
| `run_ngiab_waitprof.sh` + `ngiab_wait_sampler.py` | Finding what processes wait on |

Make them executable:

```bash
chmod +x ~/run_ngiab*.sh
```

### 1.3 Build the polling fix (`fastpoll.so`)

This tiny library shortens the 1 ms sleep ngen uses while waiting for flows from other processes.
Build it **inside the container** so it matches the container's C library (glibc 2.34).

Save as `~/ngiab_tools/fastpoll.c`:

```c
/* LD_PRELOAD shim for ngen-parallel: shorten the fixed 1 ms polling sleep that
 * HY_PointHydroNexusRemote uses while waiting for flows from other MPI ranks.
 * Only sleeps of exactly 1 ms are changed; all other sleeps pass through unchanged.
 *   NGEN_POLL_SLEEP_NS = new sleep in nanoseconds (default 20000 = 20 us; 0 = just yield the CPU)
 * Build: gcc -O2 -shared -fPIC -o fastpoll.so fastpoll.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sched.h>
#include <stdlib.h>
#include <time.h>

static int (*real_nanosleep)(const struct timespec *, struct timespec *);
static long poll_ns = -1;

int nanosleep(const struct timespec *req, struct timespec *rem)
{
    if (!real_nanosleep)
        real_nanosleep = (int (*)(const struct timespec *, struct timespec *))dlsym(RTLD_NEXT, "nanosleep");
    if (poll_ns < 0) {
        const char *e = getenv("NGEN_POLL_SLEEP_NS");
        poll_ns = e ? atol(e) : 20000;
    }
    if (req && req->tv_sec == 0 && req->tv_nsec == 1000000) {
        if (poll_ns == 0) {
            sched_yield();
            return 0;
        }
        struct timespec t = {0, poll_ns};
        return real_nanosleep(&t, rem);
    }
    return real_nanosleep(req, rem);
}
```

Build it:

```bash
module load Apptainer squashfuse gocryptfs
mkdir -p ~/ngiab_tools
cd ~/ngiab_tools
singularity exec ~/ngiab_openmpi.sif gcc -O2 -Wall -shared -fPIC -o fastpoll.so fastpoll.c -ldl
singularity exec ~/ngiab_openmpi.sif nm -D fastpoll.so | grep nanosleep   # should show "T nanosleep"
```

> Rebuild it if you switch to a different container image.

---

## Step 2 – Check the gage's input data

```bash
GAGE=gage-02361000
cd ~/ngiab_preprocess_output/$GAGE

# Exactly one hydrofabric and one realization file
ls config/*.gpkg config/*realization*.json

# Number of catchments = the maximum number of MPI processes you can use
python3 -c "import sqlite3; print(sqlite3.connect('config/${GAGE}_subset.gpkg').execute('select count(*) from divides').fetchone()[0])"

# Simulation period
python3 -c "import json; print(json.load(open('config/realization.json'))['time'])"
```

For `gage-02361000` this gives **227** catchments and 2020-01-01 → 2022-12-31 (26,280 hourly steps).

---

## Step 3 – Choose the number of MPI processes

- **On one node, use one process per catchment (227 for this gage) with the shared-memory
  transport from Step 5.6.** It was the fastest setting tested (26 s). With Open MPI's default
  transport (UCX), use **at most 128 processes per node**; more fails during `MPI_Init`.
- A node has 256 physical cores, so up to 256 processes fit on one node with `--hint=nomultithread`.
- The number of processes **cannot exceed the number of catchments.** Asking for 228 with 227
  catchments makes the partition step fail and every process crash.
- Use `--ntasks` (not `--cpus-per-task`) to set the process count. Open MPI takes its slot count from
  Slurm's task count.
- Partition files (`partitions_<N>.json`) are created automatically the first time you use a given `N`
  and reused afterwards. Check that a partition file is balanced:

```bash
python3 -c "
import json
p = json.load(open('partitions_128.json'))['partitions']
c = [len(x['cat-ids']) for x in p]
print(len(c), 'processes; catchments per process min', min(c), 'max', max(c))"
```

A good file has nearly equal counts (for example 1–2 per process). The container's interactive
"parallel mode" creates **unbalanced** partition files (1–10 catchments per process); the `auto` mode used
by the scripts below creates balanced ones.

---

## Step 4 – Basic batch run

`run_ngiab.sh` runs the container's start script in non-interactive `auto` mode on one node.
Results go to the gage's own `outputs/` folder, **overwriting the previous run**.

Key settings inside the script:

```bash
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=64              # number of MPI processes (override with --ntasks)
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread     # one process per physical core
#SBATCH --mem=64G                # 16G ran out of memory with 128 processes

module load Apptainer squashfuse gocryptfs
export OMP_NUM_THREADS=1

# Slurm sets TMPDIR=/local/job_<id>, which does not exist inside the container; bind it
TMPDIR=${TMPDIR:-/tmp}
time singularity run \
    --bind "$DATA_DIR":/ngen/ngen/data \
    --bind "$TMPDIR" \
    "$IMAGE" /ngen/ngen/data auto "$PROCS"
```

Run it:

```bash
cd ~
sbatch --ntasks=128 run_ngiab.sh gage-02361000
```

Expected time for `gage-02361000`: about **2 minutes** (without the polling fix).

> Never use the interactive menu's **serial mode** for real runs. It runs one process for all
> catchments and took more than 2.5 hours for this gage.

---

## Step 5 – Fast run with the polling fix (recommended)

`run_ngiab_fastio.sh` copies the inputs to fast local storage, runs there, and copies the
results back to a **new folder**, so your original `config/` and `outputs/` are never modified.

### 5.1 Arguments

```
sbatch run_ngiab_fastio.sh [GAGE_ID] [STAGE] [OUTVARS] [POLL_NS]
```

| Argument | Values | Default | Meaning |
|---|---|---|---|
| `GAGE_ID` | folder name | `gage-02361000` | Gage to run |
| `STAGE` | `shm`, `local`, `gpfs` | `shm` | Where to run: RAM disk, node-local disk, or a copy in your home directory |
| `OUTVARS` | `all`, `qout` | `all` | Write all model variables, or only `Q_OUT` |
| `POLL_NS` | number, or leave out | (off) | Turn on the polling fix with this sleep in nanoseconds. **`0` = fastest** |

### 5.2 Recommended command

```bash
cd ~
OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader \
  sbatch --exclusive --ntasks=227 run_ngiab_fastio.sh gage-02361000 shm all 0
```

- `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` switches Open MPI to its built-in shared-memory
  transport (Step 5.6). The job and the container inherit these variables from your shell.
- `--ntasks=227` = one process per catchment. For another gage, use the number of catchments from
  Step 2, up to 256 (the physical cores on one node).
- Without the two transport variables, stay at `--ntasks=128` (the script's default); more fails
  with UCX.

`--exclusive` gives the job the whole node. With `POLL_NS=0`, waiting processes keep their core busy,
so they should not share a node with other jobs. On a shared node, use `20000` (20 µs) instead.

### 5.3 What the script does

1. Copies `config/` (only `realization.json`, `troute.yaml`, the `.gpkg` and `cat_config/`),
   `forcings/forcings.nc` and `partitions_<N>.json` to `/dev/shm/ngiab_<jobid>/<GAGE_ID>/`.
2. Rewrites the forcing path in the **copied** `realization.json` to point at the copied forcing file
   (the original uses a full path into your home directory). With `qout`, it also adds
   `"output_variables": ["Q_OUT"]`.
3. If `POLL_NS` is set, turns on the polling fix:
   ```bash
   export SINGULARITYENV_LD_PRELOAD=$HOME/ngiab_tools/fastpoll.so
   export SINGULARITYENV_NGEN_POLL_SLEEP_NS=$POLL_NS
   ```
   Singularity passes any `SINGULARITYENV_X` variable into the container as `X`.
4. Runs `singularity run ... auto <N>` exactly like Step 4.
5. Copies `outputs/` to `~/ngiab_preprocess_output/<GAGE_ID>/outputs_<STAGE>_<OUTVARS>[_poll<NS>]_<jobid>/`.
6. Deletes the staged copy, even if the run fails.

### 5.4 Expected time for `gage-02361000`

| Phase | 128 processes, UCX (default) | 227 processes, `vader` (5.2) |
|---|---|---|
| Init (`NGen::init`) | ~7 s | **~2.5 s** |
| Model run (`NGen::simulation`) | ~6 s (was ~91 s without the fix) | **~2.8 s** |
| Routing (`NGen::routing`) | ~20 s | ~20 s |
| **Total** | **~35 s** | **~26 s** |

### 5.5 Using the polling fix in other scripts or interactively

Set the two variables before any `singularity` command:

```bash
module load Apptainer squashfuse gocryptfs
export SINGULARITYENV_LD_PRELOAD=$HOME/ngiab_tools/fastpoll.so
export SINGULARITYENV_NGEN_POLL_SLEEP_NS=0
singularity run --bind $HOME/ngiab_preprocess_output/gage-02361000:/ngen/ngen/data \
    --bind "$TMPDIR" ~/ngiab_openmpi.sif /ngen/ngen/data auto 128
```

Unset them to return to ngen's original behavior:

```bash
unset SINGULARITYENV_LD_PRELOAD SINGULARITYENV_NGEN_POLL_SLEEP_NS
```

> **How the fix works:** `LD_PRELOAD` loads `fastpoll.so` before the C library, so ngen's calls to
> `nanosleep` go to the shim first. Requests of exactly 1 ms (the polling sleep in
> `HY_PointHydroNexusRemote::get_downstream_flow` / `add_upstream_flow`) are shortened; every other
> sleep is passed through unchanged. It only changes how often a waiting process checks for its data,
> not what ngen computes.
>
> **Limitation:** it is a workaround. If a future ngen version changes this sleep, the shim silently
> stops having an effect. The proper fix belongs in ngen's source code.

### 5.6 MPI transport: more than 128 processes on one node

By default the container's Open MPI uses the **UCX** communication library. With more than
128 processes on one node, MPI start-up fails:

```
pml_ucx.c:419  Error: ucp_ep_create(proc=219) failed: Destination is unreachable
*** An error occurred in MPI_Init
```

This happened with 227 processes in one container (job 26507) and in separate containers
(26503, 26506), and allowing TCP as a fallback (`UCX_TLS=self,sm,tcp`, job 26508) did not help.
Slurm allocated the CPUs correctly (454 of 512), and per-process limits were not the cause
(open files 131,072; locked memory unlimited; max processes 255,086). The exact reason UCX fails
above 128 processes was not determined.

**Fix for single-node runs:** use Open MPI's built-in shared-memory transport instead of UCX:

```bash
export OMPI_MCA_pml=ob1          # Open MPI's own point-to-point layer instead of UCX
export OMPI_MCA_btl=self,vader   # "self" (same process) + "vader" (shared memory within a node)
```

Set them in your shell before `sbatch` (as in 5.2) or before `singularity run`; the container
and its `mpirun` inherit them.

| Job | Setup (1 node, polling fix 0, RAM disk) | Init | Model run | Routing | Run total | Outputs |
|---|---|---|---|---|---|---|
| 26494 | 128 processes, UCX | 7.1 s | 5.8 s | 20.2 s | 35 s | identical |
| 26507 | 227 processes, UCX | – | – | – | failed in `MPI_Init` | – |
| **26509** | **227 processes, `ob1`/`vader`** | **2.5 s** | **2.8 s** | 20.1 s | **26 s** | identical |

Initialization also dropped from about 7 s to 2.5 s. A likely reason is that UCX's start-up work
(for example probing the InfiniBand card) is skipped, but this was not measured separately, and the
process count changed at the same time.

> **Single node only.** `vader` is shared memory, so it cannot connect processes on different nodes.
> Do **not** set these variables for `run_ngiab_multinode.sh` on 2 or more nodes.

---

## Step 6 – Check the run and its outputs

### 6.1 Job status and run time

```bash
JOB=<jobid>
sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList -P
```

`State` should be `COMPLETED` and `ExitCode` `0:0`.

### 6.2 Phase timings

ngen's timing summary goes to the `.out` log; t-route's summary goes to the `.err` log
(Python logging writes to stderr):

```bash
grep -a "NGen::" logs/ngiab_*_$JOB.out
grep -aE "^(Stage|Outputs|Poll sleep|Run|Total|Results):" logs/ngiab_*_$JOB.out    # fastio script only
grep -a "Total execution time" logs/ngiab_*_$JOB.err
```

The `real` line from the `time` command also appears in the `.err` log, because `time` writes to stderr.

### 6.3 Output completeness

```bash
D=~/ngiab_preprocess_output/gage-02361000/outputs_shm_all_poll0_<jobid>   # or .../outputs

ls $D/ngen/nex-*.csv | wc -l      # 94 nexus files for gage-02361000
ls $D/ngen/cat-*.csv | wc -l      # 227 catchment files
for f in $D/ngen/nex-*.csv; do wc -l < $f; done | sort | uniq -c   # every file: 26281 lines (header + 26,280 hours)
ls -la $D/troute/                 # troute_output_202001010000.nc (~96 MB)
du -sh --apparent-size $D         # ~1.1 GB with all variables
```

> Plain `du -sh` on the shared filesystem shows the **compressed** size (about 33 MB here).
> Use `--apparent-size` for the real size.

### 6.4 Compare two runs

Byte-for-byte check:

```bash
A=~/ngiab_preprocess_output/gage-02361000/outputs_shm_all_26488
B=~/ngiab_preprocess_output/gage-02361000/outputs_shm_all_poll0_26494
diff -rq $A/ngen $B/ngen && echo "ngen CSV outputs identical"
cmp -s $A/troute/troute_output_202001010000.nc $B/troute/troute_output_202001010000.nc \
  && echo "troute .nc identical" || echo "troute .nc differs"
```

For a numeric comparison, use the container's Python (the cluster's Python has no `pandas`):

```bash
singularity exec ~/ngiab_openmpi.sif python -c "
import pandas as pd
a = pd.read_csv('$A/ngen/nex-501803_output.csv', header=None)
b = pd.read_csv('$B/ngen/nex-501803_output.csv', header=None)
print('max abs diff:', (a[2] - b[2]).abs().max())"
```

---

## Step 7 – Optional settings

### 7.1 Write only `Q_OUT` (saves disk space, not time)

Routing only needs `Q_OUT` from the nexus files. Writing only `Q_OUT` cuts the output from
1.1 GB to about 394 MB, but did **not** make the run faster.

- With the fastio script: use `qout` as the third argument.
- To change the gage's own config (back it up **outside** `config/` first):

```bash
cd ~/ngiab_preprocess_output/gage-02361000
cp config/realization.json ~/realization_full_outputs.json
python3 - <<'EOF'
import json
p = 'config/realization.json'
r = json.load(open(p))
r['global']['formulations'][0]['params']['output_variables'] = ['Q_OUT']
json.dump(r, open(p, 'w'), indent=4)
EOF
```

Restore with `cp ~/realization_full_outputs.json config/realization.json`.

> With only `Q_OUT`, ngen writes more decimal places (`0.000006283` instead of `0.000006`).
> The values are the same; the full-output files are just rounded to 6 decimals.

### 7.2 Hourly routing (routing ~18 s → ~4 s)

Routing is now the largest part of the run (~20 s of ~26 s). Routing once an hour instead of every
5 minutes was measured at **4.1 s instead of 17.7 s** for t-route alone. In `config/troute.yaml`:

```yaml
forcing_parameters:
    qts_subdivisions: 1      # was 12
    dt: 3600                 # was 300
    nts: 26280               # was 315360 (number of routing steps for 2020-2022)
    max_loop_size: 26280     # was 315360
```

> This changes the routing results. NWM routes every 5 minutes; check whether hourly routing is
> accurate enough for your study before using it. Switching t-route from parallel to serial
> made no difference (17.7 s vs 18.8 s).

---

## Step 8 – Optional: multi-node runs

`run_ngiab_multinode.sh` launches one container per MPI process with `srun` (the container's own
`mpirun` can only use one node). It supports the polling fix and gives byte-identical outputs, but
for `gage-02361000` it was **slower overall than one node** (47–53 s vs 35 s for 128 processes and 26 s for 227 processes on one node, see 8.4 and 5.6). Use it only
for basins too large for one node.

### 8.1 Arguments

```
sbatch [--nodes=N --ntasks=T --ntasks-per-node=P] run_ngiab_multinode.sh [GAGE_ID] [POLL_NS] [STAGE]
```

| Argument | Values | Default | Meaning |
|---|---|---|---|
| `GAGE_ID` | folder name | `gage-02361000` | Gage to run |
| `POLL_NS` | number, `off`, or leave out | off | Polling fix sleep in nanoseconds (`0` = yield, fastest) |
| `STAGE` | `inplace`, `copy` | `inplace` | `inplace` writes to the gage's `outputs/`. `copy` runs on a copy in `$HOME` and saves results to `outputs_mn<T>[_poll<NS>]_<jobid>/`, leaving the gage folder untouched |

The default layout in the script is 2 nodes × 64 processes. When you change the node count, also set
`--ntasks-per-node` so that `nodes × ntasks-per-node ≥ ntasks`.

The `copy` folder must be on the **shared** filesystem (not node-local storage): rank 0 runs routing
and reads the nexus files written by every node.

### 8.2 Commands

```bash
cd ~
# 2 nodes x 64 = 128 processes, polling fix, gage folder untouched
sbatch run_ngiab_multinode.sh gage-02361000 0 copy

# 4 nodes, 227 processes (one per catchment), polling fix
sbatch --nodes=4 --ntasks=227 --ntasks-per-node=57 run_ngiab_multinode.sh gage-02361000 0 copy

# Without the polling fix, writing to the gage's outputs/
sbatch run_ngiab_multinode.sh gage-02361000
```

> **Do not use `--exclusive` with 2 or more nodes.** Your QOS (`normal`) allows at most **825 CPUs**
> per user at a time. `--exclusive` reserves whole nodes (512 CPUs each), so a 2- or 4-node exclusive job
> can never start; it stays `PENDING` with reason `QOSMaxCpuPerUserLimit`. Without `--exclusive` a job
> counts only its own cores: 2 CPUs per process with `--hint=nomultithread` (128 processes = 256 CPUs,
> 227 = 454). Running jobs (including interactive sessions) count toward the same limit; check with
> `squeue -u $USER -o "%.8i %.12j %.10T %.6D %.6C %R"`.

### 8.3 What the script sets (already included)

```bash
unset SLURM_TRES_PER_TASK SLURM_EXPORT_ENV   # leak in when submitting from an interactive job and break srun
export PMIX_MCA_gds=hash                     # avoid PMIx shared-memory lock errors between containers
export UCX_POSIX_USE_PROC_LINK=n             # processes are in separate containers
export OMPI_MCA_btl=^openib                  # use UCX for InfiniBand

# Polling fix: Singularity passes SINGULARITYENV_X into every container as X;
# srun --export=ALL passes these to every node
export SINGULARITYENV_LD_PRELOAD=$HOME/ngiab_tools/fastpoll.so
export SINGULARITYENV_NGEN_POLL_SLEEP_NS=$POLL_NS

srun --mpi=pmix --export=ALL singularity exec ... /dmod/bin/ngen-parallel ...
```

The script also checks that the process count does not exceed the number of catchments, generates the
partition file if needed (and with `copy`, saves a newly generated one back to the gage folder),
and deletes the staged copy when the job ends.

### 8.4 1 vs 2 vs 4 nodes (with the polling fix)

All runs: polling fix `0`, `copy` mode on the shared filesystem, not exclusive, run one after another.
For comparison, the first row is the recommended single-node run from Step 5.

| Job | Launch method | Nodes | Processes | Init | Model run | Routing | Run total | Outputs |
|---|---|---|---|---|---|---|---|---|
| **26494** | `mpirun` in 1 container (`run_ngiab_fastio.sh`) | 1 | 128 | 7.1 s | 5.8 s | 20.2 s | **35 s** | identical |
| 26500 | `srun`, 1 container per process | 1 | 128 | 5.5 s | 19.3 s | 20.9 s | 50 s | identical |
| 26501 | `srun` | 2 | 128 | 3.8 s | 14.2 s | 19.6 s | 50 s | identical |
| 26502 | `srun` | 4 | 128 | 2.9 s | 10.4 s | 18.6 s | 47 s | identical |
| 26503, 26506 | `srun` | 1 | 227 | – | – | – | **failed** | – |
| 26504 | `srun` | 2 | 227 | 5.2 s | 3.6 s | 19.8 s | 47 s | identical |
| 26505 | `srun` | 4 | 227 | 3.4 s | 3.7 s | 19.2 s | 53 s | identical |

Without the fix, the same multi-node setups took about 2–3 minutes (26476: ~178 s; 26482: ~2 min).

**What this shows**

1. **The fix works on any number of nodes.** Multi-node runs dropped from about 2–3 minutes to about
   50 s, and all five successful runs are byte-identical to the reference run (26488).
2. **More nodes speed up the model run itself.** With 128 processes: 19.3 s (1 node) → 14.2 s
   (2 nodes) → 10.4 s (4 nodes). With 227 processes: about 3.6 s on 2 or 4 nodes.
3. **The total barely changes (47–53 s)**, for two reasons:
   - **Routing stays at about 20 s.** It runs on rank 0 alone, so more nodes do not help it.
   - **Container start-up and shut-down cost time.** Run total minus the three phases is about 4–5 s
     for 128 processes, but about 27 s for 227 processes on 4 nodes (227 containers to start and stop).
     With the container's own `mpirun` (26494) it is about 2 s.
4. **On one node, one shared container beats one container per process.** With 128 processes, the
   model run took 5.8 s with `mpirun` in one container versus 19.3 s with `srun` and a container per
   process. The likely reason is slower communication between processes in separate containers
   (not measured directly).
5. **227 processes on one node with `srun` fails** (twice): `pml_ucx.c ... ucp_ep_create ... Destination
   is unreachable` during `MPI_Init`. Later tests showed this is a UCX limit, not a container issue:
   227 processes in a single container fail the same way. On one node, the fix is Open MPI's
   shared-memory transport (Step 5.6), which ran 227 processes in 26 s. With UCX, use at most 128
   processes per node.

**Recommendation:** for gages that fit on one node (up to 256 catchments per node with one process
each), use Step 5.2: `run_ngiab_fastio.sh ... shm all 0` with the shared-memory transport, about
26 s. Use the multi-node script only for basins too large for one node, with `POLL_NS=0`
(or `20000` on nodes shared with other jobs), at most about 128 processes per node, and without
`--exclusive`.

---

## Step 9 – Optional: profiling

### 9.1 Install `perf` (one time)

`perf` is not installed on the cluster or in the container. Install it from conda-forge into its own
environment (the package is `linux-perf`; the package called `perf` is unrelated):

```bash
conda create -y -p $HOME/envs/perf -c conda-forge --override-channels linux-perf
$HOME/envs/perf/bin/perf --version
```

The cluster allows user-space profiling only (`perf_event_paranoid = 2`).

### 9.2 Where CPU time goes: `run_ngiab_perf.sh`

```bash
cd ~
sbatch run_ngiab_perf.sh gage-02361000
```

Profiles ranks 0 and 64 with `perf record` and ranks 1 and 65 with `perf stat`.
Results go to `~/perf_results/<jobid>/`:

| File | Content |
|---|---|
| `stat_rank*.txt` | CPU time vs elapsed time |
| `report_rank*_by_library.txt` | CPU time per library |
| `report_rank*_by_function.txt` | CPU time per function |
| `record_rank*.data` | Raw `perf` data |

> `perf` puts its own conda environment first on `PATH` for the program it starts, which makes ngen's
> embedded Python pick up the wrong Python (`No module named 'numpy'`). The script restores the
> original `PATH` with `-- env PATH="$PATH"`.

### 9.3 What processes wait on: `run_ngiab_waitprof.sh`

```bash
cd ~
sbatch run_ngiab_waitprof.sh gage-02361000
cat ~/perf_results/<jobid>/wait_profile.txt
```

Samples 16 processes 20 times per second from `/proc/<pid>/task/*/{stat,syscall,wchan}`.
No extra software or permissions are needed. This is how the 1 ms sleep was found
(92.5% of main-thread time in `clock_nanosleep`).

### 9.4 Profiling results without the fix (128 processes)

| Finding | Value |
|---|---|
| CPU use per process during the model run | ~6% (busy ~0.6 s per 10 s) |
| Main-thread time in `clock_nanosleep` | 92.5% |
| HDF5/NetCDF (forcing reads) | ~1–2% of run time |
| Busy CPU time in libc / HDF5 / gfortran string handling | 29% / 23% / 17% (variable lookups by name, not math) |
| Model math (SLOTH, Noah-OWP, CFE) | ~10% of busy time |

---

## Benchmark results

All runs: `gage-02361000`, 227 catchments, 2020-01-01 → 2022-12-31 (26,280 hourly steps).

| Job / run | Setup | Init | Model run | Routing | Total |
|---|---|---|---|---|---|
| (interactive) | Serial, 1 process | – | – | – | > 2.5 h (stopped at 64%) |
| (interactive) | 16 processes, local partition | – | ~10 min | – | – |
| (interactive) | 32 processes, balanced | 3.0 s | 250 s | 18.7 s | ~4.5 min |
| (interactive) | 64 processes, unbalanced partition | – | ~182 s | – | – |
| (interactive) | 64 processes, balanced | 4.5 s | 157 s | 20.8 s | ~3 min |
| 26476 | 2 nodes, 128 processes | – | – | 27.6 s | ~178 s |
| 26482 | 4 nodes, 227 processes | – | – | 23.3 s | 2 min 1 s |
| 26484 | 1 node, 128, shared filesystem | 7.4 s | 92.4 s | 20.2 s | 122 s |
| 26488 | 1 node, 128, RAM disk | 7.5 s | 90.8 s | 20.2 s | 120 s |
| 26489 | 1 node, 128, RAM disk, `Q_OUT` only | 7.6 s | 91.0 s | 20.3 s | 121 s |
| 26493 | 1 node, 128, RAM disk, **fix 20 µs** | 7.1 s | **11.9 s** | 19.9 s | **41 s** |
| **26494** | 1 node, 128, RAM disk, **fix 0 (yield)** | 7.1 s | **5.8 s** | 20.2 s | **35 s** |
| 26495 | Same as 26494 (repeat) | – | – | – | 35 s |
| 26500 | 1 node, 128, `srun` (multi-node script), fix 0 | 5.5 s | 19.3 s | 20.9 s | 50 s |
| 26501 | 2 nodes, 128, `srun`, fix 0 | 3.8 s | 14.2 s | 19.6 s | 50 s |
| 26502 | 4 nodes, 128, `srun`, fix 0 | 2.9 s | 10.4 s | 18.6 s | 47 s |
| 26503, 26506 | 1 node, 227, `srun`, fix 0 | – | – | – | failed (`MPI_Init`) |
| 26504 | 2 nodes, 227, `srun`, fix 0 | 5.2 s | 3.6 s | 19.8 s | 47 s |
| 26505 | 4 nodes, 227, `srun`, fix 0 | 3.4 s | 3.7 s | 19.2 s | 53 s |
| 26507 | 1 node, 227, `mpirun` in 1 container, UCX, fix 0 | – | – | – | failed (`MPI_Init`) |
| 26508 | 1 node, 227, `srun`, UCX with TCP fallback, fix 0 | – | – | – | failed (`MPI_Init`) |
| **26509** | **1 node, 227, `mpirun` in 1 container, `ob1`/`vader`, fix 0** | **2.5 s** | **2.8 s** | 20.1 s | **26 s** |

t-route alone (same inputs, outside ngen): 17.7 s as configured, 18.8 s in serial mode, **4.1 s** with hourly routing.

---

## Output comparison

Every output folder was compared against job 26488 (original code, RAM disk):

| Folder | Run | Nexus (94) | Catchment (227) | Routing `.nc` |
|---|---|---|---|---|
| `outputs/` | 26482, 4 nodes, 227 processes | identical | identical | identical |
| `outputs1/` | Earlier run (August) | identical | identical | identical |
| `outputs_gpfs_all_26484/` | Shared filesystem | identical | identical | identical |
| `outputs_shm_qout_26489/` | `Q_OUT` only | identical | same values, more decimals | identical |
| `outputs_shm_all_poll20000_26493/` | Fix, 20 µs | identical | identical | identical |
| `outputs_shm_all_poll0_26494/` | Fix, yield | identical | identical | identical |
| `outputs_shm_all_poll0_26495/` | Fix, yield (repeat) | identical | identical | identical |
| `outputs_mn128_poll0_26500/` | 1 node, 128, `srun`, fix | identical | identical | identical |
| `outputs_mn128_poll0_26501/` | 2 nodes, 128, `srun`, fix | identical | identical | identical |
| `outputs_mn128_poll0_26502/` | 4 nodes, 128, `srun`, fix | identical | identical | identical |
| `outputs_mn227_poll0_26504/` | 2 nodes, 227, `srun`, fix | identical | identical | identical |
| `outputs_mn227_poll0_26505/` | 4 nodes, 227, `srun`, fix | identical | identical | identical |
| `outputs_shm_all_poll0_26509/` | 1 node, 227, `vader` transport, fix | identical | identical | identical |

"Identical" means byte-for-byte. Results do not depend on the number of processes or nodes,
the launch method (`mpirun` or `srun`), the storage location, or the polling fix.

For the multi-node folders, the check used `diff -rq` on `ngen/` and `cmp` on the routing `.nc` file
(Step 6.4).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Run takes hours | Interactive menu, serial mode | Use the batch scripts (`auto` mode) |
| `mkdir ... /local: Read-only file system`, `orte_init failed` | Slurm's `TMPDIR=/local/job_<id>` does not exist in the container | `--bind "$TMPDIR"` |
| `There are not enough slots available ... 128 slots` | Job requested 1 task with many CPUs | Use `--ntasks=N --cpus-per-task=1` |
| Job state `OUT_OF_MEMORY`, rank 0 killed during routing | 16 GB too small for 128 processes + routing | `--mem=64G` |
| `srun: fatal: cpus-per-task set by two different environment variables` | `SLURM_TRES_PER_TASK` leaked from the interactive job | `unset SLURM_TRES_PER_TASK SLURM_EXPORT_ENV` |
| `execve(): singularity: No such file or directory` under `srun` | `SLURM_EXPORT_ENV=NONE` leaked in | Unset it and use `srun --export=ALL` |
| `PMIX ERROR ... gds_ds12_lock_pthread.c` | PMIx shared memory across containers | `export PMIX_MCA_gds=hash` |
| `UCX ERROR open(/proc/<pid>/fd/...) Permission denied` | Processes in separate containers | `export UCX_POSIX_USE_PROC_LINK=n` |
| `partitions_228.json(1): expected value`, all processes abort | More processes than catchments; empty partition file | Use ≤ number of catchments; delete the empty file |
| `ucp_disconnect_nb ... Connection reset by remote peer` at the end | Processes on other nodes already exited | Harmless; check exit code `0:0` |
| `OpenFabrics device` / `mlx5_0` warnings | Old `openib` driver probing InfiniBand | Harmless; `export OMPI_MCA_btl=^openib` |
| `INFO: gocryptfs not found` | Apptainer looks for an optional tool | `module load gocryptfs` (or `singularity -q`) |
| `Definition of "au" ... overrides prefixed-unit` | Unit library notice from every process | Harmless |
| `ModuleNotFoundError: No module named 'numpy'` under `perf` | `perf` changes `PATH` | Launch with `perf ... -- env PATH="$PATH" ...` |
| `du -sh` shows 33 MB for a 1.1 GB output | Shared filesystem compresses files | `du -sh --apparent-size` |
| `outputs/ngen` files end early (e.g. 2021-11-26) | An earlier run was stopped partway | Rerun; check line counts (Step 6.3) |
| Multi-node job stays `PENDING` with `QOSMaxCpuPerUserLimit` | `--exclusive` reserves whole nodes (512 CPUs each); QOS limit is 825 CPUs per user | Drop `--exclusive` for multi-node jobs (Step 8.2) |
| `pml_ucx.c ... ucp_ep_create ... Destination is unreachable`, `MPI_Init` fails | More than 128 processes on one node with UCX (one container or many) | On one node: `export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader` (Step 5.6). Across nodes: ≤ 128 processes per node |
| `SINGULARITYENV_... is set, but APPTAINERENV_... is preferred` (once per process) | Apptainer prefers the newer variable prefix | Harmless; renaming the variables to `APPTAINERENV_` removes it |

---

## File reference

| Path | What it is |
|---|---|
| `~/ngiab_openmpi.sif` | NGIAB container |
| `~/run_ngiab.sh` | Basic single-node run (Step 4) |
| `~/run_ngiab_fastio.sh` | Fast run on local storage with optional polling fix (Step 5) |
| `~/run_ngiab_multinode.sh` | Multi-node run (Step 8) |
| `~/run_ngiab_perf.sh` | `perf` profiling (Step 9.2) |
| `~/run_ngiab_waitprof.sh`, `~/ngiab_wait_sampler.py` | Wait profiling (Step 9.3) |
| `~/ngiab_tools/fastpoll.c`, `~/ngiab_tools/fastpoll.so` | Polling fix source and library (Step 1.3) |
| `~/envs/perf/` | conda environment with `perf` |
| `~/perf_results/<jobid>/` | Profiling results |
| `~/logs/` | Job logs (`ngiab_<jobname>_<jobid>.out` / `.err`) |
| `~/ngiab_preprocess_output/<GAGE_ID>/outputs_*_<jobid>/` | Results from `run_ngiab_fastio.sh` |
| `~/ngiab_preprocess_output/<GAGE_ID>/outputs_mn*_<jobid>/` | Results from `run_ngiab_multinode.sh ... copy` |
| `~/mpitest/` | 2-node MPI test program (can be deleted) |
