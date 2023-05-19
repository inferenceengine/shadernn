package com.oppo.seattle.snndemo;

import android.content.Context;
import android.content.res.AssetManager;
import android.opengl.EGLExt;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;
import android.util.Log;
import android.widget.TextView;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.opengles.GL10;

public class MainViewGL extends GLSurfaceView {
    String TAG = "SNN";
    final MainRenderer renderer;
    public MainViewGL(Context context, AttributeSet attrs) {
        super(context, attrs);
        Log.d(TAG, "onCreate: main GL view.");
        getHolder().setFixedSize(1080, 1920);
        setEGLContextFactory(new ContextFactory());
        setEGLConfigChooser(new ConfigChooser());
        renderer = new MainRenderer(context);
        setRenderer(renderer);
    }

    private class ConfigChooser implements EGLConfigChooser {
        public EGLConfig chooseConfig(EGL10 egl, EGLDisplay display) {
            EGLConfig [] configs = new EGLConfig[1];
            int [] num_config = new int[1];
            int [] attrib_list  = new int[] {
                    EGL10.EGL_RED_SIZE, 8,
                    EGL10.EGL_GREEN_SIZE, 8,
                    EGL10.EGL_BLUE_SIZE, 8,
                    EGL10.EGL_ALPHA_SIZE, 8,
                    EGL10.EGL_DEPTH_SIZE, 24,
                    EGL10.EGL_STENCIL_SIZE, 8,
                    EGL10.EGL_SURFACE_TYPE, EGL10.EGL_WINDOW_BIT,
                    EGLExt.EGL_RECORDABLE_ANDROID, 1,
                    EGL10.EGL_NONE,
            };
            boolean res = egl.eglChooseConfig(display, attrib_list, configs, configs.length, num_config);
            if (res && num_config[0] > 0) {
                return configs[0];
            }

            return null;
        }
    }

    private class ContextFactory implements EGLContextFactory {
        int EGL_CONTEXT_CLIENT_VERSION = 0x3098;
        int EGL_CONTEXT_FLAGS_KHR = 0x30FC;
        int EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR = 1;
        public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig config) {
            int[] release_attrib_list = {
                    EGL_CONTEXT_CLIENT_VERSION, 3,
                    EGL10.EGL_NONE,
            };
            int[] debug_attrib_list = {
                    EGL_CONTEXT_CLIENT_VERSION, 3,
                    EGL_CONTEXT_FLAGS_KHR, EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR,
                    EGL10.EGL_NONE,
            };
            return egl.eglCreateContext(display, config, EGL10.EGL_NO_CONTEXT,
                    BuildConfig.DEBUG ? debug_attrib_list : release_attrib_list);
        }
        public void destroyContext(EGL10 egl, EGLDisplay display,
                                   EGLContext context) {
            egl.eglDestroyContext(display, context);
        }
    }

    public static float[] getGeometryProfile(float[] xArr, float[] yArr) {
        if (xArr.length == 0 || xArr.length != yArr.length)
            return null;
        float massX = 0.f;
        float massY = 0.f;
        float minX = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE;
        float minY = Float.MAX_VALUE;
        float maxY = Float.MIN_VALUE;
        for (int i = 0; i < xArr.length; ++ i) {
            massX += xArr[i];
            massY += yArr[i];
            minX = Math.min(minX, xArr[i]);
            minY = Math.min(minY, yArr[i]);
            maxX = Math.max(maxX, xArr[i]);
            maxY = Math.max(maxY, yArr[i]);
        }
        float spanX = maxX - minX;
        float spanY = maxY - minY;
        float span = Math.max(spanX, spanY);
        float massCenterX = massX / xArr.length;
        float massCenterY = massY / xArr.length;
        return new float[] {massCenterX, massCenterY, span};
    }
}

class MainRenderer implements GLSurfaceView.Renderer {
    private static String TAG = "SNN";
    private MainActivity mainActivity;
    private AssetManager am;
    boolean animated = true;

    MainRenderer(Context context)
    {
        mainActivity = (MainActivity)context;
        am = context.getAssets();
    }

    public void onSurfaceCreated(GL10 unused, EGLConfig config) {
        Log.d(TAG, "renderer surface created");
        // Moved to MainActivity.onCreate()
        //NativeLibrary.init(am, mainActivity.getFilesDir().getAbsolutePath(), mainActivity.getExternalFilesDir(null).getAbsolutePath());
        NativeLibrary.initGL();

        mainActivity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mainActivity.initCamera();
            }
        });
    }

    public void onDrawFrame(GL10 unused) {
        mainActivity.runOnUiThread(() -> mainActivity.UpdateFps());
        AlgorithmConfig algorithmConfig = (mainActivity.mMenuCore != null) ? mainActivity.mMenuCore.getState() : new AlgorithmConfig();
        NativeLibrary.drawGL(algorithmConfig);

        if (algorithmConfig.isClassifierResnet18() || algorithmConfig.isClassifierMobilenetv2()) {
            mainActivity.runOnUiThread(()->mainActivity.UpdateClassifierResult(algorithmConfig));
        }
    }

    public void onSurfaceChanged(GL10 unused, int w, int h) {
        Log.d(TAG, String.format("renderer surface size changed: width=%d height=%d", w, h));
        NativeLibrary.resize(w, h);
    }
}

