package pl.cosidwo.facedetection.helper;

import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.nio.ByteBuffer;

public class AgeEstimationHelper {

    public Interpreter interpreter;

    private ImageProcessor.Builder builder;
    private ImageProcessor inputImageProcessor;

    private final int inputImageSize = 200;
    private final int p = 116;

    public AgeEstimationHelper(){
        builder = new ImageProcessor.Builder();
        inputImageProcessor = builder.add(new ResizeOp(inputImageSize,inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
                                        .add(new NormalizeOp(0f,255f))
                                        .build();
    }

    //method returns estimated age of face from bitmap
    public float predictAge(Bitmap image){

        TensorImage tensorInputImage = TensorImage.fromBitmap(image);

        float [][] ageArray = new float[1][1];

        ByteBuffer processedImageBuffer = inputImageProcessor.process(tensorInputImage).getBuffer();
        interpreter.run(processedImageBuffer,ageArray);

        //model returns value of age from range of [ 0 , 1 ] and it needs to be multiplied by 116
        return ageArray[0][0] * p;
    }
}
