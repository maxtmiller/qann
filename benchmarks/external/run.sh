#!/usr/bin/env bash
# Benchmark qann against other libraries with ann-benchmarks and VIBE.
#
#   benchmarks/external/run.sh --setup            # once, on a fresh Ubuntu 24.04 machine (uses sudo)
#   benchmarks/external/run.sh                    # both suites, default datasets, vs hnswlib and faiss
#
# Options:
#   --suite ann|vibe|both     which harness to run (default both)
#   --ann-datasets a,b        ann-benchmarks datasets (default: the six on ann-benchmarks.com)
#   --vibe-datasets a,b       VIBE datasets (default: in-distribution sets under 2M vectors, plus the
#                             yi-128-ip and llama-128-ip inner product sets; add
#                             cqadupstack-lemur-2048-ip and cqadupstack-muvera-5120-ip for the rest)
#   --compare a,b|all|none    other libraries to build and run (default hnswlib,faiss)
#   --qann SPEC               pip requirement, path to a Linux wheel or sdist, or "local" to build an
#                             sdist of this checkout (default: qann==<version in pyproject.toml>)
#   --parallelism N           indexes benchmarked at once (default: min(cores - 1, RAM GB / 16))
#   --batch                   ann-benchmarks batch mode (all queries at once, all cores)
#   --workdir DIR             where repos, images, datasets and results go (default ~/qann-bench)
#
# Plots land in <workdir>/out. Results already computed are reused; rerunning resumes.
set -euo pipefail

AB_URL=https://github.com/erikbern/ann-benchmarks
AB_REF=2e081ad32c1eccab72dcb739ad886c310b90f715
VIBE_URL=https://github.com/vector-index-bench/vibe
VIBE_REF=7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

suite=both
ann_datasets=glove-25-angular,glove-100-angular,sift-128-euclidean,nytimes-256-angular,fashion-mnist-784-euclidean,gist-960-euclidean
vibe_datasets=agnews-mxbai-1024-euclidean,arxiv-nomic-768-normalized,glove-200-cosine,gooaq-distilroberta-768-normalized,imagenet-clip-512-normalized,inaturalist-resnet-2048-cosine,landmark-dino-768-cosine,landmark-nomic-768-normalized,yahoo-minilm-384-normalized,yi-128-ip,llama-128-ip
compare=hnswlib,faiss
qann_spec="qann==$(grep -m1 '^version' "$REPO/pyproject.toml" | cut -d'"' -f2)"
parallelism=""
batch=""
workdir="$HOME/qann-bench"
setup=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    --suite) suite="$2"; shift 2 ;;
    --ann-datasets) ann_datasets="$2"; shift 2 ;;
    --vibe-datasets) vibe_datasets="$2"; shift 2 ;;
    --compare) compare="$2"; shift 2 ;;
    --qann) qann_spec="$2"; shift 2 ;;
    --parallelism) parallelism="$2"; shift 2 ;;
    --batch) batch="--batch"; shift ;;
    --workdir) workdir="$2"; shift 2 ;;
    --setup) setup=1; shift ;;
    -h | --help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

log() { printf '\n== %s\n' "$*"; }

if [[ -n "$setup" ]]; then
    log "Installing Docker, Apptainer and Python tooling"
    sudo apt-get update
    sudo apt-get install -y git python3-venv python3-pip docker.io software-properties-common acl \
        uidmap squashfuse fuse2fs gocryptfs
    sudo add-apt-repository -y ppa:apptainer/ppa
    sudo apt-get update
    sudo apt-get install -y apptainer
    # Apptainer builds need unprivileged user namespaces, which Ubuntu 24.04's AppArmor blocks.
    sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
    sudo systemctl enable --now docker
    # Lets this user reach Docker now, without logging out to pick up the docker group.
    sudo setfacl -m "user:$USER:rw" /var/run/docker.sock
    log "Setup done. Run again without --setup to benchmark."
    exit 0
fi

for tool in git docker python3; do
    command -v "$tool" >/dev/null || { echo "$tool not found; run with --setup first" >&2; exit 1; }
done
if [[ "$suite" != ann ]] && ! command -v singularity >/dev/null; then
    echo "singularity/apptainer not found (VIBE needs Linux); run with --setup or use --suite ann" >&2
    exit 1
fi

if [[ -z "$parallelism" ]]; then
    cores=$(nproc)
    mem_gb=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
    parallelism=$(( cores - 1 < mem_gb / 16 ? cores - 1 : mem_gb / 16 ))
    (( parallelism >= 1 )) || parallelism=1
fi
[[ -n "$batch" ]] && parallelism=1

mkdir -p "$workdir/out"
workdir="$(cd "$workdir" && pwd)"

# Writes <dir>/qann-dist/requirement.txt (plus the package file, if any) for the container
# build to pip install. Paths are as seen inside the container, under $2.
stage_qann() {
    local dist="$1/qann-dist" mount="$2"
    rm -rf "$dist"
    mkdir -p "$dist"
    local spec="$qann_spec"
    if [[ "$spec" == local ]]; then
        if [[ ! -f "$workdir/qann-local.tar.gz" ]]; then
            log "Building an sdist of $REPO"
            python3 -m venv "$workdir/build-venv"
            "$workdir/build-venv/bin/pip" install -q build
            rm -rf "$workdir/sdist"
            "$workdir/build-venv/bin/python" -m build --sdist -o "$workdir/sdist" "$REPO"
            cp "$workdir"/sdist/*.tar.gz "$workdir/qann-local.tar.gz"
        fi
        spec="$workdir/qann-local.tar.gz"
    fi
    if [[ -f "$spec" ]]; then
        cp "$spec" "$dist/"
        echo "$mount/$(basename "$spec")" >"$dist/requirement.txt"
    else
        echo "$spec" >"$dist/requirement.txt"
    fi
}

checkout() {
    local url="$1" ref="$2" dir="$3"
    [[ -d "$dir/.git" ]] || git clone -q "$url" "$dir"
    git -C "$dir" fetch -q origin "$ref"
    git -C "$dir" checkout -q "$ref"
}

install_adapter() {
    local dest="$1" files="$2"
    rm -rf "$dest"
    mkdir -p "$dest"
    cp "$HERE/module.py" "$HERE/$files"/* "$dest/"
}

run_ann() {
    local ab="$workdir/ann-benchmarks"
    log "ann-benchmarks @ ${AB_REF:0:7}"
    checkout "$AB_URL" "$AB_REF" "$ab"
    cd "$ab"
    [[ -d venv ]] || python3 -m venv venv
    venv/bin/pip install -q -r requirements.txt
    install_adapter ann_benchmarks/algorithms/qann ann-benchmarks
    stage_qann "$ab" /tmp/qann-dist

    log "Building images: qann ${compare}"
    venv/bin/python install.py --algorithm qann
    case "$compare" in
    none) ;;
    all) venv/bin/python install.py --proc "$(nproc)" || echo "some images failed to build; continuing" ;;
    *) for lib in ${compare//,/ }; do venv/bin/python install.py --algorithm "$lib"; done ;;
    esac

    mkdir -p "$workdir/out/ann-benchmarks"
    for ds in ${ann_datasets//,/ }; do
        log "ann-benchmarks: $ds"
        venv/bin/python run.py --dataset "$ds" --parallelism "$parallelism" $batch
        venv/bin/python plot.py --dataset "$ds" --x-scale logit --y-scale log $batch \
            -o "$workdir/out/ann-benchmarks/$ds${batch:+-batch}.png"
    done
}

run_vibe() {
    local vibe="$workdir/vibe"
    log "VIBE @ ${VIBE_REF:0:7}"
    checkout "$VIBE_URL" "$VIBE_REF" "$vibe"
    cd "$vibe"
    install_adapter vibe/algorithms/qann vibe
    mkdir -p images
    # install.sh builds from ./images, so image.def's %files path resolves there.
    stage_qann "$vibe/images" /opt/qann-dist

    log "Building images: qann ${compare}"
    ./install.sh --algorithm qann --force
    case "$compare" in
    none) ;;
    all) ./install.sh --skip-gpu || echo "some images failed to build; continuing" ;;
    *) for lib in ${compare//,/ }; do ./install.sh --algorithm "$lib"; done ;;
    esac
    [[ -f plot.sif ]] || singularity build plot.sif plot.def

    for ds in ${vibe_datasets//,/ }; do
        log "VIBE: $ds"
        python3 run.py --dataset "$ds" --parallelism "$parallelism"
    done
    ./export_results.sh --parallelism "$parallelism"
    for ds in ${vibe_datasets//,/ }; do
        ./plot.sh --dataset "$ds" --output "plots/$ds"
    done
    rm -rf "$workdir/out/vibe"
    cp -r plots "$workdir/out/vibe"
}

case "$suite" in
ann) run_ann ;;
vibe) run_vibe ;;
both) run_ann; run_vibe ;;
*) echo "--suite must be ann, vibe or both" >&2; exit 1 ;;
esac

log "Done. Plots in $workdir/out"
