package com.oppo.seattle.snndemo;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.view.Menu;

public class MainActivityGL extends MainActivity {
    private MainViewVulkan viewVulkan;

    @SuppressLint("SourceLockedOrientationActivity")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setTitle("snndemo OpenGL");
    }

    @Override
    protected void setContentView() {
        setContentView(R.layout.activity_gl);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        if (super.onCreateOptionsMenu(menu)) {
            menu.setGroupVisible(R.id.detection_shader_choices, true);
            return true;
        }
        return false;
    }
}