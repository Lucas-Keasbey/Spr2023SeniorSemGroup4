package com.example.myapplication;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    String[] labels = {"T-shirt/top",
	                    "Trouser",
                    	"Pullover",
                    	"Dress",
                    	"Coat",
                    	"Sandal",
                    	"Shirt",
                    	"Sneaker",
                    	"Bag",
                    	"Ankle boot",
                        "Not implemented"};
    ImageView pic;
    Button select;
    Button classify;
    TextView text;
    RadioButton basicButton;
    RadioButton linearButton;
    RadioButton cnnButton;
    RadioGroup modelGroup;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //getting the elements and assigning them to variables
        pic = findViewById(R.id.IVPic);
        select = findViewById(R.id.btnLoadImg);
        classify = findViewById(R.id.btnClassify);
        text = findViewById(R.id.tvGuess);
        basicButton = findViewById(R.id.BasicRadio);
        linearButton = findViewById(R.id.LinearRadio);
        cnnButton = findViewById(R.id.CNNRadio);
        modelGroup = findViewById(R.id.ModelGroup);

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                //sets the click event for load image button
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");

                startActivityForResult(Intent.createChooser(intent,"Pick an image"), 1);


            }

        });

        classify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                //sets the click event for classify image button
                text.setText(labels[predictLabel()]);
            }

        });
    }

    //this method handles grabbing the image and converting it into a bitmap
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == 1) {
            try {
                InputStream inputStream = getContentResolver().openInputStream(data.getData());
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                pic.setImageBitmap(bitmap); //sets the image view to be the bitmap created from the user's image
                classify.setEnabled(true);


            } catch (FileNotFoundException e) {
                System.out.println("Here!\n"); //debugging
                e.printStackTrace();
            }
        }

    }

    protected int predictLabel(){
        try{
            // run image through model and process prediction to get label index

        }
        catch (Exception e){
            e.printStackTrace();
        }
        return 10;
    }


}