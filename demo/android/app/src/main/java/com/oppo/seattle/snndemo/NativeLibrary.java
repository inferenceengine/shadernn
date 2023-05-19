package com.oppo.seattle.snndemo;

import android.content.res.AssetManager;
import android.view.Surface;

import java.nio.ByteBuffer;

class NativeLibrary
{
    static {
        // Load native library
        System.loadLibrary("native-lib");
    }
    public static native void init(AssetManager am, String internalStoragePath, String externalStorageDir);
    public static native void resize(int w, int h);

    public static native void drawGL(AlgorithmConfig algorithmConfig);
    public static native void drawVulkan(AlgorithmConfig algorithmConfig);

    public static native void destroy();
    public static native void queueFrame(int w, int h, int rotationDegrees, long timestamp, ByteBuffer yPlane, ByteBuffer uPlane, ByteBuffer vPlane);
    public static native void queueMetaData(long timestamp, boolean lowExposure);
    public static native void startRecording(Surface surface);
    public static native void stopRecording();
    public static native int compareFrameExposure(int w, int h, ByteBuffer frame0, ByteBuffer frame1); // return -1, 0 or 1

    public static native void initGL();
    public static native void initVulkan(Surface surface);
}
