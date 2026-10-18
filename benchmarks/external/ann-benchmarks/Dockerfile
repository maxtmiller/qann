FROM ann-benchmarks

COPY qann-dist /tmp/qann-dist
RUN pip install --no-cache-dir -r /tmp/qann-dist/requirement.txt
RUN python -c 'import qann; print(qann.__version__)'
