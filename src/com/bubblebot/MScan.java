package com.bubblebot;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.bubblebot.jni.Processor;

public class MScan extends Activity {
	
	private LayoutInflater inflater;
	private static int count = 1;
	private final Processor processor = new Processor();
	
	// Initialize the application
	@Override
	protected void onCreate(Bundle savedInstanceState) {
       super.onCreate(savedInstanceState);
       setContentView(R.layout.mscan); // Setup the UI
       
       LayoutInflater inflater = (LayoutInflater)
       		(LayoutInflater) this.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
       
       final String dir = "/sdcard/BubbleBot/capturedImages/";
       String filename = "VR_segment" + count++ + ".jpg";
       
       final ImageView image = (ImageView) findViewById(R.id.image);
       Bitmap bm = BitmapFactory.decodeFile(dir + filename);
       image.setImageBitmap(bm);
       
       image.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				if (count == 7) {
					count = 1;
				}
				String filename = dir + "VR_segment" + count++ + ".jpg";
				float f = 0.5f;
				int result = processor.processImage("/mnt/sdcard/external_sd/form.jpg",
						"/mnt/sdcard/external_sd/form.jpg", f);
				
				TextView text = (TextView) findViewById(R.id.text);
				text.setText("process image result: " + result);
				
				//Bitmap bm = BitmapFactory.decodeFile(filename);
				//image.setImageBitmap(bm);
			}
       });

	}

	@Override
	protected void onPause() {
		super.onPause();
	}

	@Override
	protected void onResume() {
		super.onResume();
	}
	
}
