set -e
ROOT=`dirname "$(realpath $0)"`
BUILD_DIR="${ROOT}/build-test"
ANDROID_NDK_HOME="$ANDROID_SDK_ROOT/ndk/21.4.7075529"

make_dir()
{
    if [ ! -d $1 ]
    then
        mkdir $1
    fi
}

clean()
{
    rm -rf ${BUILD_DIR}
}

print_usage()
{
    echo
    echo "SNN Build Script..."
    echo
    echo "Usage: Linux Build: `basename $0` linux"
    echo
    echo "Usage: Android Build: Default(64 bit, debug): `basename $0` android"
    echo
    echo "Usage: Clean: `basename $0` clean"
    echo
}

build_android()
{
    if [[ -z "$ANDROID_SDK_ROOT" ]]; then
        echo "Please set ANDROID_SDK_ROOT"
        echo "export ANDROID_SDK_ROOT=/path/to/sdk"
        exit 1
    elif [ ! -d "$ANDROID_NDK_HOME" ]; then
        echo "NDK version doesn't match"
        echo "Required version: 21.4.7075529"
        exit 1
    else
        echo "ANDROID_SDK_ROOT = $ANDROID_SDK_ROOT"
        echo "ANDROID_NDK_HOME = ${ANDROID_NDK_HOME}"
    fi

    build_type="Debug"
    if [ "$1" == "r" ]; then
        echo "Build Type: Release"
        build_type="Release"
    elif [ "$1" == "p" ]; then
        echo "Build Type: RelWithDebInfo"
        build_type="RelWithDebInfo"
    else
        echo "Build Type: Debug"
    fi
    
    make_dir ${BUILD_DIR}
    
    cd ${BUILD_DIR} && \
    cmake ../ -DCMAKE_BUILD_TYPE=${build_type} \
            -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
            -DANDROID_ABI=arm64-v8a \
            -DANDROID_NATIVE_API_LEVEL=29
    
    cd ${BUILD_DIR} && make -j8
}

build_linux()
{
    build_type="Debug"
    if [ "$1" == "r" ]; then
        echo "Build Type: Release"
        build_type="Release"
    elif [ "$1" == "p" ]; then
        echo "Build Type: RelWithDebInfo"
        build_type="RelWithDebInfo"
    else
        echo "Build Type: Debug"
    fi

    make_dir ${BUILD_DIR}
    cd ${BUILD_DIR} && cmake ../ -DCMAKE_BUILD_TYPE=${build_type}  -DOpenGL_GL_PREFERENCE=GLVND
    cd ${BUILD_DIR} && make -j8
}


if [ "$1" == "clean" ]; then
    clean
    exit 0
elif [ "$1" == "android" ]; then
    build_android $2
    exit 0
elif [ "$1" == "linux" ]; then
    build_linux $2
    exit 0
else 
    print_usage
    exit 1
fi
