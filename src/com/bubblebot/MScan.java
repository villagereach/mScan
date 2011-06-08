package com.bubblebot;

import java.io.File;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.bubblebot.jni.Processor;

public class MScan extends Activity {
	
	private static final int TAKE_PHOTO_CODE = 12;
	private static final String MSCAN_DIR = "/sdcard/mscan/";
	
	private final Processor processor = new Processor();
	
	// Initialize the application
	@Override
	protected void onCreate(Bundle savedInstanceState) {
       super.onCreate(savedInstanceState);
       setContentView(R.layout.mscan); // Setup the UI

       
       final ImageView image = (ImageView) findViewById(R.id.image);
       Bitmap bm = BitmapFactory.decodeFile("/mnt/sdcard/com.bubblebot/image.jpg");
       image.setImageBitmap(bm);
       
       image.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				takePhoto();
			}
       });

	}

	/**
	 * Creates a new intent with the camera so the user can take a picture of
	 * the form.
	 */
	private void takePhoto(){
		  final Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
		  intent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(getTempFile(this)) ); 
		  startActivityForResult(intent, TAKE_PHOTO_CODE);
	}

	/**
	 * Returns the file of the most recently taken picture.
	 * @param context
	 * @return
	 */
	private File getTempFile(Context context){
	  //it will return /sdcard/image.tmp
	  return new File("/mnt/sdcard/com.bubblebot/image.jpg");
	}
	
	/**
	 * Catches the result of taking the photo and saves the photo to a file.
	 * Shows the picture to the user. Before processing the picture.
	 */
	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data) {
	  if (resultCode == RESULT_OK && requestCode == TAKE_PHOTO_CODE) {
		  final TextView text = (TextView) findViewById(R.id.text);
	      text.setText("Received image.");
	      
	      // Draw the bitmap on the screen.
	      Bitmap bm = BitmapFactory.decodeFile("/mnt/sdcard/com.bubblebot/image.jpg");
		  final ImageView image = (ImageView) findViewById(R.id.image);
		  image.setImageBitmap(bm);
		  image.invalidate();
		  process("/mnt/sdcard/com.bubblebot/image.jpg");
	  }
	}
	
	/**
	 * To be called after a photo is taken. Process the photo as a form.
	 */
	private void process(String absolutePath) {
		final TextView text = (TextView) findViewById(R.id.text);
		text.setText("Processing form... ");
		
		float f = 0.4f;
		int result = processor.processImage(absolutePath,
				"/mnt/sdcard/external_sd/form.jpg", f);
		
		text.setText("processing completed with result: " + result);
		ImageView image = (ImageView) findViewById(R.id.image);
		// Choose an interesting file to show after the form has been processed.
		//Bitmap bm = BitmapFactory.decodeFile(filename);
		//image.setImageBitmap(bm);
	}
	
	@Override
	protected void onResume() {
		super.onResume();
	}
	
	@Override
	protected void onPause() {
		super.onPause();
	}
}
