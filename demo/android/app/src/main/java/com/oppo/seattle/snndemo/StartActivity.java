package com.oppo.seattle.snndemo;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class StartActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);

        Button button0 = (Button) findViewById(R.id.button0);
        // This is to prevent Android display button txt in ALL CAPS
        button0.setTransformationMethod(null);
        button0.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), MainActivityGL.class);
                view.getContext().startActivity(intent);
            }
        });

        Button button1 = (Button) findViewById(R.id.button1);
        button1.setTransformationMethod(null);
        // This is to prevent Android display button txt in ALL CAPS
        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), MainActivityVulkan.class);
                view.getContext().startActivity(intent);
            }
        });
    }
}