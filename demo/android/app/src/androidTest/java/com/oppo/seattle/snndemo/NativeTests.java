/* Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
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
import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.logging.Logger;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class NativeTests {
    @Test
    public void ic2() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, ic2test(am));
    }

//    @Test
//    public void spacialdenoise() {
//        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
//        AssetManager am = appContext.getAssets();
//        assertEquals(0, spacialdenoisetest(am));
//    }

    @Test(timeout = 14400000)
    public void espcn2x() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xtest(am));
    }

//    @Test
//    public void conv2d() {
//        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
//        AssetManager am = appContext.getAssets();
//        assertEquals(0, conv2dtest(am));
//    }
//
    @Test
    public void resnet18test() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18test(am));
    }

    @Test
    public void yolov3tiny() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinytest(am));
    }

//    @Test
//    public void unet() {
//        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
//        AssetManager am = appContext.getAssets();
//        assertEquals(0, unettest(am));
//    }
//
    @Test
    public void mobilenetv2() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2test(am));
    }

    //    @Test
//    public void shaderTests() {
//        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
//        AssetManager am = appContext.getAssets();
//        assertEquals(0, shaderTest(am));
//    }

    static {
        // Load native library
        System.loadLibrary("native-lib");
    }
    public static native int resnet18test(AssetManager am);
    public static native int yolov3tinytest(AssetManager am);
    public static native int unettest(AssetManager am);
    public static native int mobilenetv2test(AssetManager am);
    public static native int ic2test(AssetManager am);
    public static native int spacialdenoisetest(AssetManager am);
    public static native int espcn2xtest(AssetManager am);
//    public static native int espcn2xtestmi(AssetManager am);
    // public static native int shaderTest(AssetManager am);
//    public static native int conv2dtest(AssetManager am);
}
