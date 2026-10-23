FROM ann-benchmarks

COPY qann-dist /tmp/qann-dist
# Built from source for this CPU (AVX-512 where it has it), the way hnswlib builds itself here.
RUN CMAKE_ARGS="-DVECENGINE_NATIVE=ON" pip install --no-cache-dir --no-binary qann -r /tmp/qann-dist/requirement.txt
RUN python -c 'import qann; print(qann.__version__)'
