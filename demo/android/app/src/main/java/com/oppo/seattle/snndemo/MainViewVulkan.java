package com.oppo.seattle.snndemo;

import android.content.Context;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class MainViewVulkan extends SurfaceView implements Runnable {
    String TAG = "SNN";
    private boolean isRunning;
    private Thread renderingThread = null;
    private SurfaceHolder surfaceHolder;
    private MainActivity mainActivity;
    boolean surfaceCreatedCalled = false;

    public MainViewVulkan(Context context, AttributeSet attrs) {
        super(context, attrs);
        mainActivity = (MainActivity)context;
        surfaceHolder = getHolder();

        // Adding callback somehow disables rendering
        /* surfaceHolder.addCallback(callback2); */
        Log.d(TAG, "MainViewVulkan constructed");
    }

    /*
    // Adding callback somehow disables rendering
    // TODO: Figure out what's going on
    class SurfaceHolderCallback2 implements SurfaceHolder.Callback2 {
        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            onSurfaceCreated(holder);
        }

        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
        }

        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
            Surface surface = holder.getSurface();
            Log.d(TAG, "MainViewVulkan::surface changed: " + surface.toString());
        }

        @Override
        public void surfaceRedrawNeededAsync(SurfaceHolder holder, Runnable finishDrawing) {
        }

        @Override
        @Deprecated
        public void surfaceRedrawNeeded(SurfaceHolder holder) {
        }
    };

    SurfaceHolderCallback2 callback2 = new SurfaceHolderCallback2();
     */

    @Override
    public void run() {
        while (isRunning) {
            // Spinning until we can obtain a valid drawing surface...
            if (surfaceHolder.getSurface().isValid()) {
                if (!surfaceCreatedCalled) {
                    onSurfaceCreated(surfaceHolder);
                }
                mainActivity.runOnUiThread(() -> mainActivity.UpdateFps());
                AlgorithmConfig algorithmConfig = (mainActivity.mMenuCore != null) ? mainActivity.mMenuCore.getState() : new AlgorithmConfig();
                NativeLibrary.drawVulkan(algorithmConfig);

                if (algorithmConfig.isClassifierResnet18() || algorithmConfig.isClassifierMobilenetv2()) {
                    if (algorithmConfig.isClassifierResnet18() || algorithmConfig.isClassifierMobilenetv2()) {
                        mainActivity.runOnUiThread(()->mainActivity.UpdateClassifierResult(algorithmConfig));
                    }
                }
            }
        }
    }

    private void onSurfaceCreated(SurfaceHolder holder) {
        Surface surface = holder.getSurface();
        NativeLibrary.initVulkan(surface);

        mainActivity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mainActivity.initCamera();
            }
        });
        surfaceCreatedCalled = true;

        Log.d(TAG, "MainViewVulkan::surface created: " + surface.toString());
    }

    /**
     * Called by MainActivity.onPause() to stop the thread.
     */
    public void onPause() {
        isRunning = false;
        try {
            // Stop the thread == rejoin the main thread.
            renderingThread.join();
        } catch (InterruptedException e) {
        }
    }

    /**
     * Called by MainActivity.onResume() to start a thread.
     */
    public void onResume() {
        isRunning = true;
        renderingThread = new Thread(this);
        renderingThread.start();
    }
}
