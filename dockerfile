FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl git bzip2 ca-certificates sudo libglib2.0-0 libxext6 libsm6 libxrender1 \
    libgl1-mesa-glx libglib2.0-dev libx11-dev build-essential \
    gdal-bin libgdal-dev python3-gdal && \
    rm -rf /var/lib/apt/lists/*
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean --all --yes
ENV PATH=$CONDA_DIR/bin:$PATH

# Install mamba
RUN conda install -y -c conda-forge mamba

# Create environment and install base packages

# Create requirements file
RUN cat <<EOF > /tmp/spfeas_env_copy.yml
name: spfeas
channels:
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - backcall=0.2.0=pyh9f0ad1d_0
  - backports=1.0=pyhd8ed1ab_4
  - backports.functools_lru_cache=2.0.0=pyhd8ed1ab_0
  - blosc=1.21.1=hd32f23e_0
  - boost-cpp=1.72.0=h8e57a91_0
  - bzip2=1.0.8=h4bc722e_7
  - c-ares=1.33.0=ha66036c_0
  - ca-certificates=2025.4.26=hbd8a1cb_0
  - cairo=1.16.0=hcf35c78_1003
  - cfitsio=3.470=hb418390_7
  - curl=7.79.1=h2574ce0_1
  - dbus=1.13.6=hfdff14a_1
  - entrypoints=0.4=pyhd8ed1ab_0
  - expat=2.6.2=h59595ed_0
  - ffmpeg=4.1.3=h167e202_0
  - fontconfig=2.14.0=h8e229c2_0
  - freetype=2.10.4=h0708190_1
  - freexl=1.0.6=h166bdaf_1
  - gdal=2.4.4=py36hbb8311d_1
  - geos=3.8.1=he1b5a44_0
  - geotiff=1.5.1=h05acad5_10
  - gettext=0.22.5=he02047a_3
  - gettext-tools=0.22.5=he02047a_3
  - giflib=5.2.2=hd590300_0
  - glib=2.66.3=h58526e2_0
  - gmp=6.3.0=hac33072_2
  - gnutls=3.6.13=h85f3911_1
  - graphite2=1.3.13=h59595ed_1003
  - gst-plugins-base=1.14.5=h0935bb2_2
  - gstreamer=1.14.5=h36ae1b5_2
  - harfbuzz=2.4.0=h9f30f68_3
  - hdf4=4.2.15=h10796ff_3
  - hdf5=1.10.5=nompi_h5b725eb_1114
  - icu=64.2=he1b5a44_1
  - ipykernel=5.5.5=py36hcb3619a_0
  - ipython_genutils=0.2.0=pyhd8ed1ab_1
  - jasper=1.900.1=h07fcdf6_1006
  - jpeg=9e=h0b41bf4_3
  - json-c=0.13.1=hbfbb72e_1002
  - jupyter_client=7.1.2=pyhd8ed1ab_0
  - jupyter_core=4.8.1=py36h5fab9bb_0
  - kealib=1.4.13=hec59c27_0
  - keyutils=1.6.1=h166bdaf_0
  - krb5=1.19.3=h3790be6_0
  - lame=3.100=h166bdaf_1003
  - ld_impl_linux-64=2.40=h12ee557_0
  - libasprintf=0.22.5=he8f35ee_3
  - libasprintf-devel=0.22.5=he8f35ee_3
  - libblas=3.9.0=24_linux64_openblas
  - libcblas=3.9.0=24_linux64_openblas
  - libclang=9.0.1=default_hb4e5071_5
  - libcurl=7.79.1=h2574ce0_1
  - libdap4=3.20.6=hd7c4107_2
  - libedit=3.1.20191231=he28a2e2_2
  - libev=4.33=hd590300_2
  - libexpat=2.6.2=h59595ed_0
  - libffi=3.2.1=he1b5a44_1007
  - libgcc=12.4.0=h767d61c_2
  - libgcc-ng=12.4.0=h69a702a_2
  - libgdal=2.4.4=h5439ffd_1
  - libgettextpo=0.22.5=he02047a_3
  - libgettextpo-devel=0.22.5=he02047a_3
  - libgfortran-ng=13.2.0=h69a702a_0
  - libgfortran5=13.2.0=ha4646dd_0
  - libglib=2.66.3=hbe7bbb4_0
  - libgomp=12.4.0=h767d61c_2
  - libiconv=1.17=hd590300_2
  - libkml=1.3.0=hd79254b_1012
  - liblapack=3.9.0=24_linux64_openblas
  - liblapacke=3.9.0=24_linux64_openblas
  - libllvm9=9.0.1=default_hc23dcda_4
  - libnetcdf=4.7.4=nompi_h9f9fd6a_101
  - libnghttp2=1.43.0=h812cca2_1
  - libopenblas=0.3.27=pthreads_hac2b453_1
  - libopencv=4.2.0=py36_3
  - libpng=1.6.37=h21135ba_2
  - libpq=12.9=h16c4e8d_3
  - libsodium=1.0.18=h36c2ea0_1
  - libspatialite=4.3.0a=h2482549_1038
  - libssh2=1.10.0=ha56f1ee_2
  - libstdcxx=12.4.0=h8f9b012_2
  - libstdcxx-ng=12.4.0=h4852527_2
  - libtiff=4.1.0=hc3755c2_3
  - libuuid=2.38.1=h0b41bf4_0
  - libwebp=1.0.2=h56121f0_5
  - libxcb=1.16=hd590300_0
  - libxkbcommon=0.10.0=he1b5a44_0
  - libxml2=2.9.10=hee79883_0
  - lz4-c=1.9.3=h9c3ff4c_1
  - lzo=2.10=hd590300_1001
  - mock=5.1.0=pyhd8ed1ab_0
  - ncurses=6.4=h6a678d5_0
  - nest-asyncio=1.6.0=pyhd8ed1ab_0
  - nettle=3.6=he412f7d_0
  - nspr=4.35=h27087fc_0
  - nss=3.69=hb5efdd6_1
  - numexpr=2.7.3=py36h0cdc3f0_0
  - numpy=1.19.5=py36hfc0c790_2
  - opencv=4.2.0=py36_3
  - openh264=1.8.0=hdbcaa40_1000
  - openjpeg=2.3.1=hf7af979_3
  - openssl=1.1.1w=h7f8727e_0
  - packaging=21.3=pyhd8ed1ab_0
  - parso=0.7.1=pyh9f0ad1d_0
  - pcre=8.45=h9c3ff4c_0
  - pip=21.2.2=py36h06a4308_0
  - pixman=0.38.0=h516909a_1003
  - poppler=0.67.0=h14e79db_8
  - poppler-data=0.4.12=hd8ed1ab_0
  - postgresql=12.9=h16c4e8d_3
  - proj=7.0.0=h966b41f_5
  - pthread-stubs=0.4=h36c2ea0_1001
  - py-opencv=4.2.0=py36h95af2a2_3
  - pygments=2.14.0=pyhd8ed1ab_0
  - pyparsing=3.1.4=pyhd8ed1ab_0
  - pyqt=5.12.3=py36haa643ae_3
  - python=3.6.11=h4d41432_2_cpython
  - python_abi=3.6=2_cp36m
  - qt=5.12.5=hd8c4c69_1
  - qtpy=2.0.1=pyhd8ed1ab_0
  - readline=8.2=h5eee18b_0
  - setuptools=58.0.4=py36h06a4308_0
  - six=1.16.0=pyh6c4a22f_0
  - snappy=1.1.10=hdb0a2a9_1
  - sqlite=3.45.3=h5eee18b_0
  - tk=8.6.14=h39e8969_0
  - tornado=6.1=py36h8f6f2f9_1
  - traitlets=4.3.3=pyhd8ed1ab_2
  - wcwidth=0.2.10=pyhd8ed1ab_0
  - wheel=0.37.1=pyhd3eb1b0_0
  - x264=1!152.20180806=h14c3975_0
  - xerces-c=3.2.2=h8412b87_1004
  - xorg-kbproto=1.0.7=h7f98852_1002
  - xorg-libice=1.1.1=hd590300_0
  - xorg-libsm=1.2.4=h7391055_0
  - xorg-libx11=1.8.9=hb711507_1
  - xorg-libxau=1.0.11=hd590300_0
  - xorg-libxdmcp=1.1.3=h7f98852_0
  - xorg-libxext=1.3.4=h0b41bf4_2
  - xorg-libxrender=0.9.11=hd590300_0
  - xorg-renderproto=0.11.1=h7f98852_1002
  - xorg-xextproto=7.3.0=h0b41bf4_1003
  - xorg-xproto=7.0.31=h7f98852_1007
  - xz=5.6.4=h5eee18b_1
  - zeromq=4.3.5=h59595ed_1
  - zlib=1.2.13=h5eee18b_1
  - zstd=1.4.9=ha95c52a_0
  - pip:
      - access==1.1.9
      - affine==2.3.1
      - alabaster==0.7.13
      - astroid==1.4.9
      - astropy==4.1
      - attrs==22.2.0
      - babel==2.11.0
      - backports-shutil-get-terminal-size==1.0.0
      - beautifulsoup4==4.12.3
      - bitarray==0.8.1
      - blaze==0.10.1
      - bleach==4.1.0
      - bokeh==0.12.5
      - boto==2.46.1
      - bottleneck==1.2.1
      - cerberus==1.1
      - certifi==2018.4.16
      - cffi==1.15.1
      - chardet==3.0.3
      - charset-normalizer==2.0.12
      - click==8.0.4
      - click-plugins==1.1.1
      - cligj==0.7.2
      - cloudpickle==2.2.1
      - colorama==0.3.9
      - contextlib2==0.5.5
      - coverage==6.2
      - cryptography==40.0.2
      - cycler==0.11.0
      - cython==0.25.2
      - cytoolz==0.8.2
      - dask==2021.3.0
      - dataclasses==0.8
      - datashape==0.5.2
      - decorator==4.4.2
      - deprecation==2.1.0
      - distributed==1.16.3
      - docutils==0.18.1
      - esda==2.4.3
      - et-xmlfile==1.1.0
      - fastcache==1.0.2
      - fiona==1.8.22
      - flask==2.0.3
      - flask-cors==5.0.0
      - future==1.0.0
      - geojson==2.3.0
      - geopandas==0.9.0
      - gevent==1.2.1
      - giddy==2.3.4
      - greenlet==2.0.2
      - h5py==2.7.0
      - heapdict==1.0.1
      - humanize==0.5.1
      - idna==3.10
      - imageio==2.15.0
      - imagesize==1.4.1
      - importlib-metadata==4.8.3
      - importlib-resources==5.4.0
      - inequality==1.0.0
      - iniconfig==1.1.1
      - ipython==6.2.1
      - isort==4.2.5
      - itsdangerous==2.0.1
      - jdcal==1.4.1
      - jedi==0.10.2
      - jinja2==3.0.3
      - joblib==1.1.1
      - jsonschema==2.6.0
      - kiwisolver==1.3.1
      - lazy-object-proxy==1.7.1
      - libpysal==4.5.0
      - llvmlite==0.36.0
      - locket==1.0.0
      - lxml==3.7.3
      - mapbox-vector-tile==1.2.0
      - mapclassify==2.6.0
      - markupsafe==2.0.1
      - matplotlib==3.3.4
      - mbutil==0.3.0
      - mccabe==0.7.0
      - mgwr==2.2.1
      - mistune==2.0.5
      - mpmath==0.19
      - msgpack-python==0.5.6
      - multipledispatch==1.0.0
      - munch==4.0.0
      - nbconvert==5.1.1
      - nbformat==5.1.3
      - networkx==2.5.1
      - nltk==3.2.3
      - nose==1.3.7
      - notebook==5.2.2
      - numba==0.53.1
      - numpydoc==0.6.0
      - odo==0.5.0
      #- opencv-contrib-python==4.11.0.86
      - opencv-python==3.4.1.15
      - openpyxl==2.4.7
      - pandas==1.1.5
      - pandocfilters==1.5.1
      - partd==0.3.8
      - pathlib2==2.2.1
      - patsy==1.0.1
      - pep8==1.7.0
      - pexpect==4.2.1
      - pickleshare==0.7.4
      - pillow==8.4.0
      - pluggy==1.0.0
      - ply==3.10
      - pointpats==2.2.0
      - prompt-toolkit==1.0.14
      - protobuf==3.19.6
      - psutil==7.0.0
      - ptyprocess==0.5.1
      - py==1.11.0
      - pyclipper==1.3.0.post6
      - pycparser==2.21
      - pycrypto==2.6.1
      - pycurl==7.43.0
      - pyflakes==1.5.0
      - pygeos==0.13
      - pylint==1.6.4
      - pyodbc==4.0.16
      - pyopenssl==17.0.0
      - pyproj==3.0.1
      - pyqt5-sip==4.19.18
      - pyqtchart==5.12
      - pyqtwebengine==5.12.1
      - pysal==2.3.0
      - pytest==7.0.1
      - pytest-cov==4.0.0
      - python-dateutil==2.8.0
      - pytz==2025.2
      - pywavelets==1.1.1
      - pyyaml==5.1.1
      - pyzmq==16.0.2
      - qtawesome==0.4.4
      - qtconsole==4.3.0
      - quantecon==0.5.2
      - rasterio==1.2.10
      - rasterstats==0.17.1
      - requests==2.27.1
      - retrying==1.3.4
      - rope-py3k==0.9.4.post1
      - rtree==0.9.7
      - scikit-image==0.17.2
      - scikit-learn==0.22
      - scipy==1.5.4
      - seaborn==0.11.2
      - segregation==2.0.0
      - shapely==1.6.3
      - simplegeneric==0.8.1
      - simplejson==3.20.1
      - singledispatch==3.4.0.3
      - sklearn==0.0
      - snowballstemmer==2.2.0
      - snuggs==1.4.7
      - sortedcollections==2.1.0
      - sortedcontainers==2.4.0
      - soupsieve==2.3.2.post1
      - spaghetti==1.5.9
      - spglm==1.0.8
      - sphinx==1.5.6
      - spint==1.0.7
      - splot==1.1.7
      - spreg==1.3.2
      - spvcm==0.3.0
      - sqlalchemy==1.4.54
      - statsmodels==0.12.2
      - sympy==1.0
      - tables==3.4.4
      - tblib==1.7.0
      - terminado==0.6
      - testpath==0.6.0
      - tifffile==2020.9.3
      - tobler==0.9.0
      - tomli==1.2.3
      - toolz==0.12.0
      - tqdm==4.64.1
      - typing-extensions==4.1.1
      - unicodecsv==0.14.1
      - urllib3==1.24.3
      - webencodings==0.5.1
      - werkzeug==2.0.3
      - widgetsnbextension==2.0.0
      - wrapt==1.16.0
      - xlrd==1.0.0
      - xlsxwriter==0.9.6
      - xlwt==1.2.0
      - xmltodict==0.14.2
      - zict==2.1.0
      - zipp==3.6.0
EOF



RUN mamba env create -f /tmp/spfeas_env_copy.yml -y --verbose && \
    conda clean --all --yes


# Verify environment without activating
RUN /opt/conda/bin/conda run -n spfeas python -c 'import cython; print(cython.__version__)'

 
# Download and install mpglue
RUN wget https://github.com/jgrss/mpglue/archive/refs/tags/0.2.14.tar.gz -O mpglue-0.2.14.tar.gz
#print cython version

RUN /opt/conda/bin/conda run -n spfeas pip install mpglue-0.2.14.tar.gz


# # Clone and install spfeas
RUN git clone https://github.com/jgrss/spfeas.git /opt/spfeas && \
    /opt/conda/bin/conda run -n spfeas pip install /opt/spfeas
# initialize mamba for bash
RUN echo 'eval "$(mamba shell hook --shell bash)"' >> ~/.bashrc 
RUN bash -c "source ~/.bashrc"

# Activate the spfeas environment by default when the container starts
CMD ["bash", "-c", "source activate spfeas && exec bash"]