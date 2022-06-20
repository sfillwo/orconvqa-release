git clone https://github.com/sfillwo/orconvqa-release.git
cd orconvqa-release || return

wget --no-parent --content-disposition -r -nc -R "index.html*" https://ciir.cs.umass.edu/downloads/ORConvQA/
cd ..

conda env update -n base -f env_cpu.yml
