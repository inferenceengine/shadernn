[1/4/2020] OpenCV 4.2.0 pre-built libs for Windows and Android

[2/27/2020] We are now using a custom compiled version for the Android version:
- It enables us to only compile what we use and remove some dependencies
- It can be integrated as part of the Android system, no high-level android calls

To compile it:
```
mkdir opencv
cd opencv
git clone https://github.com/opencv/opencv.git src-github
cd src-github
git checkout -b opencv_innopeak 43a907dddaa552331c7f18785641fabef8701759
cd ..
mkdir opencv-build
mkdir -p android-build/armeabi-v7a
mkdir -p android-build/arm64-v8a
cd opencv-build
# Cleaning directory 
rm -rf ../opencv-build/*
# Compiling arm
cmake ../src-github/ -DENABLE_NEON=ON \
-DENABLE_VFPV3=ON \
-DWITH_PTHREADS_PF=OFF \
-DCV_TRACE=OFF \
-DCV_ENABLE_INTRINSICS=ON \
-DBUILD_opencv_ittnotify=OFF \
-DWITH_PNG=OFF \
-DWITH_TIFF=OFF \
-DWITH_JPEG=OFF \
-DBUILD_ITT=OFF \
-DBUILD_TESTING=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DBUILD_EXAMPLES=OFF \
-DBUILD_DOCS=OFF \
-DBUILD_opencv_apps=OFF \
-DBUILD_SHARED_LIBS=OFF \
-DOpenCV_STATIC=ON \
-DWITH_1394=OFF \
-DWITH_CUBLAS=OFF \
-DWITH_CUFFT=OFF \
-DWITH_FFMPEG=OFF \
-DWITH_GDAL=OFF \
-DWITH_GSTREAMER=OFF \
-DWITH_GTK=OFF \
-DWITH_HALIDE=OFF \
-DWITH_JASPER=OFF \
-DWITH_NVCUVID=OFF \
-DWITH_OPENEXR=OFF \
-DWITH_PROTOBUF=OFF \
-DWITH_QUIRC=OFF \
-DWITH_V4L=OFF \
-DWITH_WEBP=OFF \
-DBUILD_LIST=core,imgproc \
-DANDROID_NDK=/home/achanot/Android/Sdk/ndk/21.0.6113669 \
-DCMAKE_TOOLCHAIN_FILE=/home/achanot/Android/Sdk/ndk/21.0.6113669/build/cmake/android.toolchain.cmake \
-DANDROID_NATIVE_API_LEVEL=android-28 \
-DBUILD_JAVA=OFF \
-DBUILD_ANDROID_EXAMPLES=OFF \
-DBUILD_ANDROID_PROJECTS=OFF \
-DANDROID_STL=c++_static \
-DBUILD_SHARED_LIBS=ON \
-DCMAKE_INSTALL_PREFIX:PATH=/home/achanot/02-Work/09-OpenCV/android-build/out \
-DANDROID_ABI=armeabi-v7a

make
# Saving result 
cp -rf lib/armeabi-v7a/libopencv_* ../android-build/armeabi-v7a/
# Cleaning directory
rm -rf ../opencv-build/*
# Compiling arm64
cmake ../src-github/ -DENABLE_NEON=ON \
-DBUILD_opencv_ittnotify=OFF \
-DWITH_PTHREADS_PF=OFF \
-DCV_TRACE=OFF \
-DCV_ENABLE_INTRINSICS=ON \
-DWITH_PNG=OFF \
-DWITH_TIFF=OFF \
-DWITH_JPEG=OFF \
-DBUILD_ITT=OFF \
-DBUILD_TESTING=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DBUILD_EXAMPLES=OFF \
-DBUILD_DOCS=OFF \
-DBUILD_opencv_apps=OFF \
-DBUILD_SHARED_LIBS=OFF \
-DOpenCV_STATIC=ON \
-DWITH_1394=OFF \
-DWITH_CUBLAS=OFF \
-DWITH_CUFFT=OFF \
-DWITH_FFMPEG=OFF \
-DWITH_GDAL=OFF \
-DWITH_GSTREAMER=OFF \
-DWITH_GTK=OFF \
-DWITH_HALIDE=OFF \
-DWITH_JASPER=OFF \
-DWITH_NVCUVID=OFF \
-DWITH_OPENEXR=OFF \
-DWITH_PROTOBUF=OFF \
-DWITH_QUIRC=OFF \
-DWITH_V4L=OFF \
-DWITH_WEBP=OFF \
-DBUILD_LIST=core,imgproc \
-DANDROID_NDK=/home/achanot/Android/Sdk/ndk/21.0.6113669 \
-DCMAKE_TOOLCHAIN_FILE=/home/achanot/Android/Sdk/ndk/21.0.6113669/build/cmake/android.toolchain.cmake \
-DANDROID_NATIVE_API_LEVEL=android-28 \
-DBUILD_JAVA=OFF \
-DBUILD_ANDROID_EXAMPLES=OFF \
-DBUILD_ANDROID_PROJECTS=OFF \
-DANDROID_STL=c++_static \
-DBUILD_SHARED_LIBS=ON \
-DCMAKE_INSTALL_PREFIX:PATH=/home/achanot/02-Work/09-OpenCV/android-build/out \
-DANDROID_ABI=arm64-v8a
make
# Saving result
cp -rf lib/arm64-v8a/libopencv_* ../android-build/arm64-v8a/
# Done!
```
