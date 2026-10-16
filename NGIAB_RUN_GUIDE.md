# Running NGIAB (ngen + t-route) fast on the cluster

A step-by-step guide to running the NextGen In A Box (NGIAB) container as a Slurm batch job on the
`amd` partition, based on a series of timed test runs with `gage-02361000` (227 catchments,
2020-01-01 to 2022-12-31, hourly).

**Bottom line:** with one node, 128 MPI processes and a small polling fix (Step 5), a full
3-year run takes **about 35 seconds**, down from **more than 2.5 hours** in serial mode.
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

- **Use 128 processes on one node.** It was the fastest and most efficient setting tested.
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
sbatch --exclusive run_ngiab_fastio.sh gage-02361000 shm all 0
```

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

| Phase | Time |
|---|---|
| Init (`NGen::init`) | ~7 s |
| Model run (`NGen::simulation`) | **~6 s** (was ~91 s without the fix) |
| Routing (`NGen::routing`) | ~20 s |
| **Total** | **~35 s** |

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

Routing is now the largest part of the run (~20 s of ~35 s). Routing once an hour instead of every
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
`mpirun` can only use one node). It works, but for this gage **one node with 128 processes was as fast
as four nodes with 227** (about 2 minutes each, both without the polling fix), so it is only worth it
for much larger basins.

```bash
cd ~
sbatch run_ngiab_multinode.sh gage-02361000                        # 2 nodes x 64 = 128 processes
sbatch --nodes=4 --ntasks=227 run_ngiab_multinode.sh gage-02361000 # one process per catchment
```

Settings the script needs (already included):

```bash
unset SLURM_TRES_PER_TASK SLURM_EXPORT_ENV   # leak in when submitting from an interactive job and break srun
export PMIX_MCA_gds=hash                     # avoid PMIx shared-memory lock errors between containers
export UCX_POSIX_USE_PROC_LINK=n             # processes are in separate containers
export OMPI_MCA_btl=^openib                  # use UCX for InfiniBand
srun --mpi=pmix --export=ALL singularity exec ... /dmod/bin/ngen-parallel ...
```

The script checks that the process count does not exceed the number of catchments, and generates
the partition file itself.

> The polling fix (Step 5) was **not tested** in multi-node runs. `srun --export=ALL` should pass the
> `SINGULARITYENV_*` variables through.

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

"Identical" means byte-for-byte. Results do not depend on the number of processes or nodes,
the storage location, or the polling fix.

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
| `~/mpitest/` | 2-node MPI test program (can be deleted) |
