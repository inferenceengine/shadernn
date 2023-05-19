package com.oppo.seattle.snndemo;

import android.annotation.SuppressLint;
import android.os.Bundle;

public class MainActivityVulkan extends MainActivity {
    private MainViewVulkan viewVulkan;

    @SuppressLint("SourceLockedOrientationActivity")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        viewVulkan = findViewById(R.id.mainViewVulkan);
        setTitle("snndemo Vulkan");
    }

    @Override
    protected void setContentView() {
        setContentView(R.layout.activity_vulkan);
    }

    @Override
    protected void onPause() {
        super.onPause();
        viewVulkan.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        viewVulkan.onResume();
    }
}
