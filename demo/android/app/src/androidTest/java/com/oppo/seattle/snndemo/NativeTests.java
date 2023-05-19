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
import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.logging.Logger;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

@RunWith(AndroidJUnit4.class)
public class NativeTests {
    @Test
    public void conv2dCS32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, conv2dCS32(am));
    }
    @Test
    public void conv2dCS16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, conv2dCS16(am));
    }

    @Test
    public void conv2dVK32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, conv2dVK32(am));
    }

    @Test
    public void conv2dVK16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, conv2dVK16(am));
    }

    @Test(timeout = 14400000)
    public void resnet18FS32Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18FS32Single(am));
    }
    @Test(timeout = 14400000)
    public void resnet18FS32Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18FS32Double(am));
    }
    @Test(timeout = 14400000)
    public void resnet18FS16Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18FS16Single(am));
    }
    @Test(timeout = 14400000)
    public void resnet18FS16Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18FS16Double(am));
    }
    @Test(timeout = 14400000)
    public void resnet18CS32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18CS32(am));
    }
    @Test(timeout = 14400000)
    public void resnet18CS16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18CS16(am));
    }
    @Test(timeout = 14400000)
    public void resnet18VK32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18VK32(am));
    }
    @Test(timeout = 14400000)
    public void resnet18VK16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, resnet18VK16(am));
    }
    @Test(timeout = 14400000)
    public void yolov3tinyFS32Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinyFS32Single(am));
    }
    @Test(timeout = 14400000)
    public void yolov3tinyFS32Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinyFS32Double(am));
    }
    @Test(timeout = 14400000)
    public void yolov3tinyFS16Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinyFS16Single(am));
    }
    @Test(timeout = 14400000)
    public void yolov3tinyFS16Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinyFS16Double(am));
    }
    @Test(timeout = 14400000)
    public void yolov3tinyCS32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinyCS32(am));
    }
    @Test(timeout = 14400000)
    public void yolov3tinyCS16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinyCS16(am));
    }
    @Test(timeout = 14400000)
    public void yolov3tinyVK32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinyVK32(am));
    }
    @Test(timeout = 14400000)
    public void yolov3tinyVK16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, yolov3tinyVK16(am));
    }

    @Test(timeout = 14400000)
    public void mobilenetv2FS32Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2FS32Single(am));
    }
    @Test(timeout = 14400000)
    public void mobilenetv2FS32Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2FS32Double(am));
    }
    @Test(timeout = 14400000)
    public void mobilenetv2FS16Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2FS16Single(am));
    }
    @Test(timeout = 14400000)
    public void mobilenetv2FS16Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2FS16Double(am));
    }
    @Test(timeout = 14400000)
    public void mobilenetv2CS32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2CS32(am));
    }
    @Test(timeout = 14400000)
    public void mobilenetv2CS16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2CS16(am));
    }
    @Test(timeout = 14400000)
    public void mobilenetv2VK32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2VK32(am));
    }
    @Test(timeout = 14400000)
    public void mobilenetv2VK16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, mobilenetv2VK16(am));
    }

    @Test(timeout = 14400000)
    public void unetFS32Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, unetFS32Single(am));
    }
    @Test(timeout = 14400000)
    public void unetFS32Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, unetFS32Double(am));
    }
    @Test(timeout = 14400000)
    public void unetFS16Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, unetFS16Single(am));
    }
    @Test(timeout = 14400000)
    public void unetFS16Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, unetFS16Double(am));
    }
    @Test(timeout = 14400000)
    public void unetCS32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, unetCS32(am));
    }
    @Test(timeout = 14400000)
    public void unetCS16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, unetCS16(am));
    }
    @Test(timeout = 14400000)
    public void unetVK32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, unetVK32(am));
    }
    @Test(timeout = 14400000)
    public void unetVK16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, unetVK16(am));
    }

    @Test(timeout = 14400000)
    public void spatialdenoiseFS32Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, spatialdenoiseFS32Single(am));
    }
    @Test(timeout = 14400000)
    public void spatialdenoiseFS32Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, spatialdenoiseFS32Double(am));
    }
    @Test(timeout = 14400000)
    public void spatialdenoiseFS16Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, spatialdenoiseFS16Single(am));
    }
    @Test(timeout = 14400000)
    public void spatialdenoiseFS16Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, spatialdenoiseFS16Double(am));
    }
    @Test(timeout = 14400000)
    public void spatialdenoiseCS32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, spatialdenoiseCS32(am));
    }
    @Test(timeout = 14400000)
    public void spatialdenoiseCS16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, spatialdenoiseCS16(am));
    }
    @Test(timeout = 14400000)
    public void spatialdenoiseVK32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, spatialdenoiseVK32(am));
    }
    @Test(timeout = 14400000)
    public void spatialdenoiseVK16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, spatialdenoiseVK16(am));
    }

    @Test(timeout = 14400000)
    public void styletransferCS32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, styletransferCS32(am));
    }
    @Test(timeout = 14400000)
    public void styletransferCS16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, styletransferCS16(am));
    }
    @Test(timeout = 14400000)
    public void styletransferVK32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, styletransferVK32(am));
    }
    @Test(timeout = 14400000)
    public void styletransferVK16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, styletransferVK16(am));
    }

    @Test(timeout = 14400000)
    public void espcn2xFS32Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xFS32Single(am));
    }
    @Test(timeout = 14400000)
    public void espcn2xFS32Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xFS32Double(am));
    }
    @Test(timeout = 14400000)
    public void espcn2xFS16Single() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xFS16Single(am));
    }
    @Test(timeout = 14400000)
    public void espcn2xFS16Double() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xFS16Double(am));
    }
    @Test(timeout = 14400000)
    public void espcn2xCS32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xCS32(am));
    }
    @Test(timeout = 14400000)
    public void espcn2xCS16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xCS16(am));
    }
    @Test(timeout = 14400000)
    public void espcn2xVK32() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xVK32(am));
    }
    @Test(timeout = 14400000)
    public void espcn2xVK16() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, espcn2xVK16(am));
    }

    @Test
    public void imagetexture() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager am = appContext.getAssets();
        assertEquals(0, imagetexturetest(am));
    }

    @Test
    public void imageTextureVulkanResize() {
        imageTextureVulkanResizeTest();
    }

    static {
        // Load native library
        System.loadLibrary("native-lib");
    }

    //Resnet18 Unit Tests
    public static native int resnet18FS32Single(AssetManager am);
    public static native int resnet18FS32Double(AssetManager am);
    public static native int resnet18FS16Single(AssetManager am);
    public static native int resnet18FS16Double(AssetManager am);
    public static native int resnet18CS32(AssetManager am);
    public static native int resnet18CS16(AssetManager am);
    public static native int resnet18VK32(AssetManager am);
    public static native int resnet18VK16(AssetManager am);

    //YOLO V3 Tiny Unit Tests
    public static native int yolov3tinyFS32Single(AssetManager am);
    public static native int yolov3tinyFS32Double(AssetManager am);
    public static native int yolov3tinyFS16Single(AssetManager am);
    public static native int yolov3tinyFS16Double(AssetManager am);
    public static native int yolov3tinyCS32(AssetManager am);
    public static native int yolov3tinyCS16(AssetManager am);
    public static native int yolov3tinyVK32(AssetManager am);
    public static native int yolov3tinyVK16(AssetManager am);

    //Mobilenet V2 Unit Tests
    public static native int mobilenetv2FS32Single(AssetManager am);
    public static native int mobilenetv2FS32Double(AssetManager am);
    public static native int mobilenetv2FS16Single(AssetManager am);
    public static native int mobilenetv2FS16Double(AssetManager am);
    public static native int mobilenetv2CS32(AssetManager am);
    public static native int mobilenetv2CS16(AssetManager am);
    public static native int mobilenetv2VK32(AssetManager am);
    public static native int mobilenetv2VK16(AssetManager am);

    //UNet Unit Tests
    public static native int unetFS32Single(AssetManager am);
    public static native int unetFS32Double(AssetManager am);
    public static native int unetFS16Single(AssetManager am);
    public static native int unetFS16Double(AssetManager am);
    public static native int unetCS32(AssetManager am);
    public static native int unetCS16(AssetManager am);
    public static native int unetVK32(AssetManager am);
    public static native int unetVK16(AssetManager am);

    //Spat
    public static native int spatialdenoiseFS32Single(AssetManager am);
    public static native int spatialdenoiseFS32Double(AssetManager am);
    public static native int spatialdenoiseFS16Single(AssetManager am);
    public static native int spatialdenoiseFS16Double(AssetManager am);
    public static native int spatialdenoiseCS32(AssetManager am);
    public static native int spatialdenoiseCS16(AssetManager am);
    public static native int spatialdenoiseVK32(AssetManager am);
    public static native int spatialdenoiseVK16(AssetManager am);

    //Style Transfer Unit Tests
    public static native int styletransferCS32(AssetManager am);
    public static native int styletransferCS16(AssetManager am);
    public static native int styletransferVK32(AssetManager am);
    public static native int styletransferVK16(AssetManager am);

    //ESPCN Unit Tests
    public static native int espcn2xFS32Single(AssetManager am);
    public static native int espcn2xFS32Double(AssetManager am);
    public static native int espcn2xFS16Single(AssetManager am);
    public static native int espcn2xFS16Double(AssetManager am);
    public static native int espcn2xCS32(AssetManager am);
    public static native int espcn2xCS16(AssetManager am);
    public static native int espcn2xVK32(AssetManager am);
    public static native int espcn2xVK16(AssetManager am);

    //public static native int shaderTest(AssetManager am);
    public static native int conv2dCS32(AssetManager am);
    public static native int conv2dCS16(AssetManager am);
    public static native int conv2dVK32(AssetManager am);
    public static native int conv2dVK16(AssetManager am);

    public static native int imagetexturetest(AssetManager am);
    public static native int imageTextureVulkanResizeTest();

    public static native void setNumLoops(int numLoops);
}
