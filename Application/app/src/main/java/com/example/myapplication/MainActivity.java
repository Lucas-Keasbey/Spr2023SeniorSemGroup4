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

import org.pytorch.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;

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
                        "Model Broke"};
    ImageView pic;
    Button select, classify;
    TextView text;
    RadioButton basicButton, linearButton, cnnButton;
    RadioGroup modelGroup;

    Module modelBasic, modelLinear, modelCNN;
    Bitmap bitmap;
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

        try {
            modelBasic = LiteModuleLoader.load(assetFilePath("BasicModel.pt"));
            modelLinear = LiteModuleLoader.load(assetFilePath("LinearModel.pt"));
            modelCNN = LiteModuleLoader.load(assetFilePath("CNNModel.pt"));
        } catch (IOException e) {
            e.printStackTrace();
        }

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
                Module active;
                int id = modelGroup.getCheckedRadioButtonId();

                if (id == basicButton.getId())
                    active = modelBasic;
                else if (id == linearButton.getId())
                    active = modelLinear;
                else
                    active = modelCNN;

                //sets the click event for classify image button
                text.setText(labels[predictLabel(active)]);
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
                bitmap = BitmapFactory.decodeStream(inputStream);
                pic.setImageBitmap(bitmap); //sets the image view to be the bitmap created from the user's image
                classify.setEnabled(true);


            } catch (FileNotFoundException e) {
                System.out.println("Here!\n"); //debugging
                e.printStackTrace();
            }
        }

    }

    protected int predictLabel(Module active){
        try{
            // run image through model and process prediction to get label index
            Tensor inputTensor = generateTensor();
            float[] outputArr = active.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
            return highestIndex(outputArr);
        }
        catch (Exception e){
            e.printStackTrace();
        }
        //return fail message if it doesn't work
        return 10;
    }

    public int highestIndex(float[] arr){
        int highest = -1;
        float val = 0;
        for (int i=0;i<arr.length;i++){
            if (arr[i] > val){
                highest = i;
                val = arr[i];
            }
        }
        return highest;
    }

    public Tensor generateTensor(){
        Tensor out = null;
        try{
            Bitmap scaled = Bitmap.createScaledBitmap(bitmap, 28, 28, false);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(scaled.getByteCount());
            scaled.copyPixelsToBuffer(byteBuffer);
            byteBuffer.rewind();
            long[] shape = {3, 28, 28}; // adjust shape and channels as needed
            float[] data = new float[byteBuffer.limit() / 4]; // Assuming 4 bytes per float
            byteBuffer.asFloatBuffer().get(data);
            out = Tensor.fromBlob(data, shape);
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return out;
    }

    //following method is pulled from standard Android Studio tools to get file path
    public String assetFilePath(String assetName) throws IOException {
        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }


}