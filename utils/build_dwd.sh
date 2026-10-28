# 1. Load the module
module load singularity

# 2. Set your paths
export PROJECT_DIR="/home/nr_dodb/nr_dose"
export BUILD_DIR="/tmp/${USER}_sif_build"

# 3. Create the local build folder and clear out any old attempts
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

# 4. Tell Singularity to use this local drive for temporary cache
export SINGULARITY_TMPDIR=$BUILD_DIR
export SINGULARITY_CACHEDIR=$BUILD_DIR

# 5. Copy your blueprint to the local drive and move there
cp $PROJECT_DIR/dwd.def $BUILD_DIR/
cd $BUILD_DIR

# 6. Run the build (This bypasses the network security block!)
singularity build --fakeroot --fix-perms dwd.sif dwd.def

# 7. Copy the finished container back home and clean up the node

cp dwd.sif $PROJECT_DIR/


cd $PROJECT_DIR
rm -rf $BUILD_DIR