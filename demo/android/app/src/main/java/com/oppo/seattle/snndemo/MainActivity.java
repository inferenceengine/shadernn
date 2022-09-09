/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.oppo.seattle.snndemo;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.hardware.camera2.CameraManager;
import android.net.Uri;
import android.opengl.EGLExt;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;
import android.webkit.MimeTypeMap;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Objects;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.opengles.GL10;

public class MainActivity extends AppCompatActivity {
    private static final int RESULT_LOAD_VIDEO = 1;
    static String TAG = "SNN";
    CameraPreview cameraPreview = new CameraPreview();
    VideoReader videoReader;
    DSP dsp;
    boolean clearance = false;
    boolean fastCV = false;
    boolean aiDenoise = false;
    boolean aiDenoiseClearance = false;
    boolean aiDenoiseClearanceDsp = false;
    boolean spatialDenoiser = false;
    boolean spatialDenoiserClearance = false;
    boolean clearanceTemporalFiltering = false;
    boolean basic_cnn = false;
    int algVersion = 0;
    RecordingManager recordingManager;
    MenuCore mMenuCore;

    public MainActivity() {
        videoReader = new VideoReader(this);
    }

    // uncomment the following section to load AGA interceptors to enable AGA capture
    // on unrooted phone.

    /*static {
        try  {
            System.loadLibrary("AGA");
            Log.i(TAG, "AGA interceptor loaded.");
        }
        catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "GA interceptor is not loaded: " + e.getMessage());
        }
    }//*/

    @SuppressLint("SourceLockedOrientationActivity")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate: main activity created.");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT); // fix to portrait mode.
        setContentView(R.layout.activity_main);
        recordingManager = RecordingManager.getInstance(this);
        Button buttonRecord = findViewById(R.id.buttonRecord);
        buttonRecord.setOnClickListener( v -> toggleRecord(buttonRecord));
        //BEGIN DSP
        dsp = new DSP(this);
        //END DSP
    }

    /**
     * Toggles whether or not this is recording.
     * @param button The button who's text needs to be updated to what its action does now.
     */
    private void toggleRecord(Button button) {
        //Toggle recording state.
        //If it is recording right now.
        if (recordingManager.isRecording()) {
            //Stop recording video.
            recordingManager.stopRecording();

            //Update the button so user knows they can start recording.
            button.setText(R.string.start_record);

            //If it is not recording right now.
        } else {
            //Start recording video.
            recordingManager.startRecording();

            //Update the button so user knows they can stop recording.
            button.setText(R.string.stop_record);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu){
        if(super.onCreateOptionsMenu(menu)){
            Log.d(TAG, "onCreateOptionsMenu: menu created");
            getMenuInflater().inflate(R.menu.main_activity_menu, menu);
            mMenuCore = new MenuCore(this, menu);
            return true;
        }
        else {
            return false;
        }
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item){
         return mMenuCore.onOptionsItemSelected(item);
    }

    public boolean keepMenuOpen(MenuItem item) {
        item.setShowAsAction(MenuItem.SHOW_AS_ACTION_COLLAPSE_ACTION_VIEW);
        item.setActionView(new View(this));
        item.setOnActionExpandListener(new MenuItem.OnActionExpandListener(){
            @Override
            public boolean onMenuItemActionExpand(MenuItem item){
                return false;
            }

            @Override
            public boolean onMenuItemActionCollapse(MenuItem item){
                return false;
            }
        });
        return false;
    }

    @Override
    protected void onDestroy() {
        cameraPreview.stopCamera();
        NativeLibrary.destroy();
        super.onDestroy();
    }

    void stopAllSources() {
        if (cameraPreview != null) {
            cameraPreview.stopCamera();
        }
        if (videoReader != null) {
            videoReader.stopVideo();
        }
    }

    void initCamera() {
        Log.d(TAG, "initCamera");
        stopAllSources();
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Requesting permission");
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA}, 0);
        } else {
            Log.d(TAG, "Permission already granted");
            startCameraPreview();
        }
    }

    void initVideo() {
        Log.d(TAG, "initVideo");
        stopAllSources();

        if (!getPermission(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0)) {
            Log.d(TAG, "Permission requested");
        } else {
            Log.d(TAG, "Permission already granted");
            loadVideoFromGallery();
        }
    }

    boolean getPermission(String[] requiredPermissions, int requestCode) {
        ArrayList<String> permissionsToRequest = new ArrayList<>();
        for (String permissionType : requiredPermissions) {
            if (ContextCompat.checkSelfPermission(this,
                    permissionType)
                    != PackageManager.PERMISSION_GRANTED) {
                // Permission is not granted
                permissionsToRequest.add(permissionType);
                Log.d(TAG, "Permission required: " + permissionType);
            }
        }
        if (!permissionsToRequest.isEmpty()) {
            String[] p2rArray = new String[permissionsToRequest.size()];
            p2rArray = permissionsToRequest.toArray(p2rArray);
            Log.d(TAG, "Requesting permission(s)");
            ActivityCompat.requestPermissions(this, p2rArray, requestCode);
            return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                           int[] grantResults){
        Log.d(TAG, "onRequestPermissionsResult");
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int i = 0; i < Math.min(permissions.length, grantResults.length); i++) {
            if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
                Log.d(TAG, "Permission granted: " + permissions[i]);
                switch (permissions[i]) {
                    case Manifest.permission.WRITE_EXTERNAL_STORAGE:
                    case Manifest.permission.CAMERA:
                        startCameraPreview();
                        break;
                    default:
                        throw new IllegalStateException("Unexpected value: " + permissions[i]);
                }
            }
        }
    }

    private void startCameraPreview() {
        cameraPreview.startCamera((CameraManager) Objects.requireNonNull(getSystemService(CAMERA_SERVICE)),
                getWindowManager().getDefaultDisplay().getRotation());
    }

    public void loadVideoFromGallery() {
        // Create intent to Open Image applications like Gallery, Google Photos
        Intent galleryIntent = new Intent(Intent.ACTION_GET_CONTENT);
        galleryIntent.setType("video/*");
        // Start the Intent
        startActivityForResult(galleryIntent, RESULT_LOAD_VIDEO);
        Log.d(TAG, "Started activity for video picker");

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.d(TAG, "Activity Result: " + requestCode + " " + resultCode + " " + data);
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == RESULT_LOAD_VIDEO) {
            if (resultCode == RESULT_OK && null != data) {
                final Uri selectedVideo = data.getData();
                if (selectedVideo != null) {
                    Log.d(TAG, "Starting video: " + selectedVideo.getPath());
                    videoReader.startVideo(selectedVideo);
                }
            } else {
                initCamera();
            }
        }
    }

    public void videoStopped() {
        initCamera();
    }
}

// Copied as is from demo-app
class ModelFromFile extends ModelSelection {
    private String path;

    ModelFromFile(String path) {
        this.path = path;
        String ext = MimeTypeMap.getFileExtensionFromUrl(path);
        if (ext.length() > 0) {
            ext = "." + ext;
        }
        name = new File(Objects.requireNonNull(path)).getName();
        if (name.toLowerCase().endsWith("3x" + ext)) {
            scaleFactor = 3;
        } else {
            scaleFactor = 2;
        }
    }

    @Override
    public String getFullName() {
        return path;
    }
}

// Copied as is from demo-app
abstract class ModelSelection implements Comparable<ModelSelection> {
    @SuppressWarnings("WeakerAccess")
    protected int scaleFactor = 2;
    String name;

    int getScaleFactor() {
        return scaleFactor;
    }

    @Override
    @NonNull
    public String toString() {
        return getName();
    }

    public String getName() {
        return name;
    }

    public String getFullName() {
        return getName();
    }

    @Override
    public int compareTo(ModelSelection o) {
        return this.toString().compareTo(o.toString());
    }
}

class MainView extends GLSurfaceView {
    String TAG = "SNN";
    final MainRenderer renderer;
    public MainView(Context context, AttributeSet attrs) {
        super(context, attrs);
        Log.d(TAG, "onCreate: main view.");
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

class FrameTimeAverager
{
    private static final int N = 10;
    private float[] buffer = new float[N];
    private int cursor = 0;
    private boolean bufferNotFull = true;

    public float low, high, average, current;

    FrameTimeAverager()
    {
        for(int i = 0; i < N; ++i) buffer[i] = 0.0f;
    }

    void Update(float value)
    {
        current = value;
        if (value > 1e9) return; // ignore frame time over 1 second.
        buffer[cursor] = value;
        ++cursor;
        if (cursor >= N) bufferNotFull = false;
        cursor %= N;
        low = Float.MAX_VALUE;
        high = .0f;
        float sum = 0.0f;
        int count = bufferNotFull ? cursor : N;
        for(int i = 0; i < count; ++i) {
            value = buffer[i];
            sum += value;
            if (value < low) low = value;
            if (value > high) high = value;
        }
        average = sum / (float)count;
    }
}

class MainRenderer implements GLSurfaceView.Renderer {
    private static String TAG = "SNN";
    private long lastFpsTime = 0;
    private long frameCounter = 0;
    private MainActivity mainActivity;
    private AssetManager am;
    private TextView tvFps;
    private TextView classifierResult;
    private FrameTimeAverager frameTime = new FrameTimeAverager();
    boolean animated = true;

    private void UpdateFps()
    {
        if (0 == lastFpsTime)
            lastFpsTime = System.nanoTime();
        else {
            long current = System.nanoTime();
            long time = current - lastFpsTime;
            if (time > 1e9) {
                frameTime.Update((float)time / frameCounter);
                float fps = 1e9f / frameTime.current;
                lastFpsTime = current;
                frameCounter = 0;
                if (null != tvFps) {
                    final String text = String.format(
                            "FPS = %.2f ([%.2f, %.2f]ms)",
                            fps,
                            frameTime.current / 1e6f,
                            frameTime.low / 1e6f
                    );
                    mainActivity.runOnUiThread(() -> tvFps.setText(text));
                }
            }
        }
        ++frameCounter;
    }

    private void UpdateClassifierResult(AlgorithmConfig algorithmConfig) {
        final String text = String.format("Classifier output = " + algorithmConfig.getClassifierOutput());
        mainActivity.runOnUiThread(()->classifierResult.setText(text));
    }

    private void DefaultClassifierField() {
        final String text = String.format("");
        mainActivity.runOnUiThread(()->classifierResult.setText(text));
    }

    MainRenderer(Context context)
    {
        mainActivity = (MainActivity)context;
        am = context.getAssets();
    }

    public void onSurfaceCreated(GL10 unused, EGLConfig config) {
        Log.d(TAG, "renderer surface created");
        tvFps = mainActivity.findViewById(R.id.textViewFPS);
        classifierResult = mainActivity.findViewById(R.id.classifierOutput);
        NativeLibrary.init(am, mainActivity.getFilesDir().getAbsolutePath(), mainActivity.getExternalFilesDir(null).getAbsolutePath());

        mainActivity.runOnUiThread(new Runnable() {

            @Override
            public void run() {
                mainActivity.initCamera();
            }
        });
    }

    public void onDrawFrame(GL10 unused) {
        UpdateFps();
        AlgorithmConfig algorithmConfig = null;
        if (mainActivity.mMenuCore != null)
            algorithmConfig = mainActivity.mMenuCore.getState();
        else algorithmConfig = new AlgorithmConfig();

        NativeLibrary.draw(algorithmConfig);

        if (algorithmConfig.isClassifierResnet18() || algorithmConfig.isClassifierMobilenetv2()) {
            UpdateClassifierResult(algorithmConfig);
        }
    }

    public void onSurfaceChanged(GL10 unused, int w, int h) {
        Log.d(TAG, String.format("renderer surface size changed: width=%d height=%d", w, h));
        NativeLibrary.resize(w, h);
    }
}

class NativeLibrary
{
    static {
        // Load native library
        System.loadLibrary("native-lib");
    }
    public static native void init(AssetManager am, String internalStoragePath, String externalStorageDir);
    public static native void initDSP(String skelLocation);
    public static native void resize(int w, int h);

    public static native void draw(AlgorithmConfig algorithmConfig);

    public static native void destroy();
    public static native void queueFrame(int w, int h, int rotationDegrees, long timestamp, ByteBuffer yPlane, ByteBuffer uPlane, ByteBuffer vPlane);
    public static native void queueMetaData(long timestamp, boolean lowExposure);
    public static native void startRecording(Surface surface);
    public static native void stopRecording();
    public static native void updateSavePath(String savePath);
    public static native void renameRecording(String newName);
    public static native int compareFrameExposure(int w, int h, ByteBuffer frame0, ByteBuffer frame1); // return -1, 0 or 1

}
