package pl.cosidwo.facedetection;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import java.io.File;
import java.io.IOException;

import pl.cosidwo.facedetection.helper.AgeEstimationHelper;

public class MainActivity extends AppCompatActivity {

    private AgeEstimationHelper ageEstimationHelper;

    private FaceDetector faceDetector;

    private ImageView imageView;

    private MaterialCardView cardView;
    private TextView viewInfoKids;
    private TextView viewInfoAdults;
    private TextView viewNumber;
    private TextView viewCardTitle;
    private TextView viewKidsNumber;
    private TextView viewAdultsNumber;

    private int numberOfKids;
    private int numberOfAdults;

    private final int REQUEST_IMAGE_CAPTURE = 101;
    private final int SELECT_PICTURE = 200;
    private String currentPhotoPath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //initalizing object of helper class
        ageEstimationHelper = new AgeEstimationHelper();

        //initialization of FaceDetector
        FaceDetectorOptions.Builder builder = new FaceDetectorOptions.Builder();
        FaceDetectorOptions faceDetectorOptions = builder.setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST).build();
        faceDetector = FaceDetection.getClient(faceDetectorOptions);

        //initialization of layout components
        MaterialButton cameraButton = findViewById(R.id.button_camera);
        MaterialButton galleryButton = findViewById(R.id.button_gallery);
        imageView = findViewById(R.id.image_view);
        cardView = findViewById(R.id.view_card);
        viewInfoKids = findViewById(R.id.view_info_kids);
        viewInfoAdults = findViewById(R.id.view_info_adults);
        viewNumber = findViewById(R.id.number_view);
        viewCardTitle = findViewById(R.id.title_card);
        viewKidsNumber = findViewById(R.id.number_kids);
        viewAdultsNumber = findViewById(R.id.number_adults);

        //initialization of age model
        try {
            initializeAgeModel();
        } catch (IOException e) {
            Toast.makeText(this,"Cannot initialize age model",Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }

        //if user clicks on "take picture" button
        cameraButton.setOnClickListener(view -> {
            //cannot take pictures from emulator
            if(!isEmulator())
                takePicture();
            else{
                Toast.makeText(this,"This function is not available in emulators", Toast.LENGTH_LONG).show();
            }

            numberOfAdults = 0;
            numberOfKids = 0;
        });

        //if user clicks on "choose from gallery" button
        galleryButton.setOnClickListener(view -> {
            choosePicture();
            numberOfAdults = 0;
            numberOfKids = 0;
        });
    }

    //called when user takes photo or chooses it from gallery
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        //if user took picture
        if(resultCode == RESULT_OK && requestCode == REQUEST_IMAGE_CAPTURE){

            //creating bitmap from image taken with camera
            Bitmap bitmap = BitmapFactory.decodeFile(currentPhotoPath);

            //detecting faces
            detectFaces(bitmap);

        } //if user wanted to choose picture from gallery
        else if(resultCode == RESULT_OK && requestCode == SELECT_PICTURE){

            //getting uri of chosen image
            Uri selectedImageUri = data.getData();

            if(selectedImageUri != null){
                Bitmap bitmap = null;
                try {
                    //creating bitmap with chosen image
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImageUri);

                    //detecting faces
                    detectFaces(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    //method used to initialize model
    private void initializeAgeModel() throws IOException {
        Interpreter.Options options = new Interpreter.Options();

        //loading tflite file (from https://github.com/shubham0204/Age-Gender_Estimation_TF-Android)
        Interpreter ageModelInterpreter = new Interpreter(FileUtil.loadMappedFile(this, "model_age_q.tflite"), options.addDelegate(new NnApiDelegate()));

        //initializing interpreter from helper class
        ageEstimationHelper.interpreter = ageModelInterpreter;
    }

    //method called when user clicks take picture button
    private void takePicture(){

        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        //checking if user's device has a camera
        if(getApplicationContext().getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {

            //checking if resolveActivity() returns component which can display takePictureIntent
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {

                //creating temp. image file and getting uri of that file
                File photo = createImageFile();
                Uri photoUri = FileProvider.getUriForFile(this, "pl.cosidwo.facedetection", photo);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);

                //starting activity
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        } else{
            Toast.makeText(this,"Your device doesn't have a camera",Toast.LENGTH_SHORT).show();
        }
    }

    //method called when user clicked "choose from gallery" button
    private void choosePicture(){
        //creating intent
        Intent choosePictureIntent = new Intent();
        choosePictureIntent.setType("image/*");
        choosePictureIntent.setAction(Intent.ACTION_GET_CONTENT);

        //starting activity
        startActivityForResult(Intent.createChooser(choosePictureIntent, "SELECT PICTURE"),SELECT_PICTURE);
    }

    //method used to create temp. file
    private File createImageFile(){
        File temporaryPhotoFile = getExternalFilesDir(Environment.DIRECTORY_PICTURES);

        try {
            //creating temp. image.jpg file
            File tempFile = File.createTempFile("image",".jpg",temporaryPhotoFile);

            //getting path to that file
            currentPhotoPath = tempFile.getAbsolutePath();
            return tempFile;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    //method used to detect, count and identify age of faces
    private void detectFaces(Bitmap image){
        InputImage inputImage = InputImage.fromBitmap(image,0);

        //pass picture to MLKIT face detector
        faceDetector.process(inputImage).addOnSuccessListener(faces ->{

            //if there is at least one face detected
           if(faces.size()!=0){

               //creating temp. bitmap and canvas object
               Bitmap tempBitmap = Bitmap.createBitmap(image.getWidth(),image.getHeight(), Bitmap.Config.RGB_565);
               Canvas tempCanvas = new Canvas(tempBitmap);

               tempCanvas.drawBitmap(image,0,0,null);

               //making CardView visible and setting its title
               cardView.setVisibility(View.VISIBLE);
               viewCardTitle.setText("Results");


               //setting textview with number of faces
               viewNumber.setText("Number of faces: "+faces.size());

               //declaring array used to store age of detected faces
               float [] age = new float[faces.size()];
               int i = 0;

               //loop for each of detected faces
               for(Face face : faces){

                   //creating rectangle with bounds of detected face
                   Rect rect = face.getBoundingBox();
                   try{
                       //estimating age of person and drawing rectangle around detected face on bitmap
                       age[i] = (float) Math.floor(ageEstimationHelper.predictAge(cropBitmap(image,rect)));
                       tempCanvas.drawRoundRect(new RectF(face.getBoundingBox()),2,2,createPaint(age[i],image.getWidth()));
                   }catch(IllegalArgumentException e){
                       //if theres a problem with inappropriate frame
                       e.printStackTrace();
                       Toast.makeText(this,"Cannot estimate age - rearrange frame of picture", Toast.LENGTH_LONG).show();
                   }
                   i++;
               }
               //displaying bitmap with rectangles inside ImageView
               imageView.setImageDrawable(new BitmapDrawable(getResources(),tempBitmap));

               //setting text within textviews
               viewKidsNumber.setText("Number of kids: "+ numberOfKids);
               viewAdultsNumber.setText("Number of adults: "+ numberOfAdults);
               viewInfoKids.setText("Kids' rectangle");
               viewInfoAdults.setText("Adults' rectangle");

           } else{
               //if there are no faces detected
               Toast.makeText(this,"No faces found in analyzed picture",Toast.LENGTH_SHORT).show();
           }
        });
    }

    //method used to crop bitmap into a rectangle with single face
    private Bitmap cropBitmap(Bitmap image, Rect rect) throws IllegalArgumentException{
        return Bitmap.createBitmap(
                image,
                rect.left + 5,
                    rect.top,
                rect.width() - 10,
                rect.height() - 10);
    }

    //method used to create and return Paint object which helps to draw rectangle around detected face
    private Paint createPaint(float age, int width){
        Paint paintForRectangle = new Paint();
        if(age<18){
            //drawing yellow rectangle if person < 18yo
            paintForRectangle.setColor(Color.YELLOW);
            numberOfKids++;

        }else{
            //drawing green rectangle if person is >= 18yo
            paintForRectangle.setColor(Color.GREEN);
            numberOfAdults++;
        }

        //setting outlined empty rectangle
        paintForRectangle.setStyle(Paint.Style.STROKE);

        //painting thinner edges of rectangle in case of lower resolution images
        if(width<500)
            paintForRectangle.setStrokeWidth(5);
        else if(width>=500 && width<1000)
            paintForRectangle.setStrokeWidth(10);
        else if(width>=1000 && width<2000)
            paintForRectangle.setStrokeWidth(15);
        else if(width>=2000 && width<3000)
            paintForRectangle.setStrokeWidth(20);
        else
            paintForRectangle.setStrokeWidth(25);

        return paintForRectangle;
    }

    //returns true if user runs app on emulator
    private boolean isEmulator() {
        return (Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic"))
                || Build.FINGERPRINT.startsWith("generic")
                || Build.FINGERPRINT.startsWith("unknown")
                || Build.HARDWARE.contains("goldfish")
                || Build.HARDWARE.contains("ranchu")
                || Build.MODEL.contains("google_sdk")
                || Build.MODEL.contains("Emulator")
                || Build.MODEL.contains("Android SDK built for x86")
                || Build.MANUFACTURER.contains("Genymotion")
                || Build.PRODUCT.contains("sdk_google")
                || Build.PRODUCT.contains("google_sdk")
                || Build.PRODUCT.contains("sdk")
                || Build.PRODUCT.contains("sdk_x86")
                || Build.PRODUCT.contains("sdk_gphone64_arm64")
                || Build.PRODUCT.contains("vbox86p")
                || Build.PRODUCT.contains("emulator")
                || Build.PRODUCT.contains("simulator");
    }
}