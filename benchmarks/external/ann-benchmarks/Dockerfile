FROM ann-benchmarks

# OpenBLAS for k-means and batch coarse search: without it qann builds with the scalar
# fallback and its index builds take about 3x as long (#170)
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas-dev && rm -rf /var/lib/apt/lists/*
COPY qann-dist /tmp/qann-dist
# Built from source for this CPU (AVX-512 where it has it), the way hnswlib builds itself here.
RUN CMAKE_ARGS="-DVECENGINE_NATIVE=ON" pip install --no-cache-dir --no-binary qann -r /tmp/qann-dist/requirement.txt
RUN python -c 'import qann; print(qann.__version__)'
RUN ldd "$(python -c 'import qann; print(qann.__file__)')" | grep -q openblas || { echo "qann built without OpenBLAS" >&2; exit 1; }
