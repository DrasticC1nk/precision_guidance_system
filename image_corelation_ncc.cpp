//DEPENDENCIES
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

//FUNCTION TO FIND THE SIMILARITY MEASURE. HERE I AM USING ZERO-MEAN NCC (NORMALIZED CROSS CORRELATION[ALSO CALLED PEARSON'S CORRELATION COEFFICIENT]) 
double compute_ncc(const Mat& patch, const Mat& templ) 
{
    //USED THIS TO DEBUG, NOT REQUIRED ANYMORE TBH
    //CV_Assert(patch.size() == templ.size());
    //CV_Assert(patch.type() == CV_8U && templ.type() == CV_8U);

	//EXTRACTING THE MEAN INTENSITY VALUES OF BOTH THE PATCH AND THE TARGET
    double meanPatch = mean(patch)[0];
    double meanTempl = mean(templ)[0];

    double num = 0.0, denomPatch = 0.0, denomTempl = 0.0;

	//ITERATING THROUGH EACH PIXEL OF THE PATCH AND THE TARGET
    for (int y = 0; y < patch.rows; ++y) 
    {
        const uchar* pRow = patch.ptr<uchar>(y);
        const uchar* tRow = templ.ptr<uchar>(y);

        for (int x = 0; x < patch.cols; ++x) 
        {
			//CHECKING WETHER THE PIXEL'S RAW BRIGHTNESS IS GREATER THAN THE MEAN BRIGHTNESS. "ZERO-MEAN"
            double pVal = pRow[x] - meanPatch;
            double tVal = tRow[x] - meanTempl;

            num += pVal * tVal;

            denomPatch += pVal * pVal;
            denomTempl += tVal * tVal;
        }
    }

	//THIS MAKES IT IMPERVEOUS TO THE CONTRAST CHANGES SOMEHOW. STILL FIGURING OUT THE MATH
    double denom = sqrt(denomPatch * denomTempl);

	//FAIL SAFE TO PREVENT DIVISION BY ZERO
    if (denom < 1e-10)
    {
        return 0.0;
    }

    //RETURN THE NCC
    return num / denom;
}


//DRIVER 
int main() 
{
	//READING THE IMAGE AND CONVERTING IT TO GRAYSCALE(JUST THE INTENSITY, NO RGB VALUES)
    Mat img = imread("yo.jpeg", IMREAD_GRAYSCALE);
       
    //RETURNING IF IT FAILS
    if (img.empty()) 
    {
        cerr << "No image...\n";

        return -1;
    }

	//GIVING A 80% VIEWPORT OF THE ORIGNAL IMAGE SO THE SHIFT SEEMS VALID
    float viewportFraction = 0.8f; 

    int viewW = static_cast<int>(img.cols * viewportFraction);
    int viewH = static_cast<int>(img.rows * viewportFraction);

    int padX = (img.cols - viewW) / 2;
    int padY = (img.rows - viewH) / 2;

    Mat padded;
	copyMakeBorder(img, padded, padY, padY, padX, padX, BORDER_REPLICATE); //MAKING A NEW PADDED IMAGE TO PREVENT AN OUTOFBOUNDS EDGE CASE

	//PREPARING THE IMAGE TO BE DISPLAYED 
    Rect viewRect(padX, padY, viewW, viewH);
    Mat viewport = padded(viewRect).clone();

    //SOME INTERACTIVE STUFF 
    Rect trgt = selectROI("Select Target", viewport);

    if (trgt.width == 0 || trgt.height == 0) 
    {
        cerr << "Nothing selected\n";

        return -1;
    }

    //THIS STORES THE SELECTED REGION
    Mat templ = viewport(trgt).clone();
    destroyWindow("Select Target");

	//SHIFTING THE VIEWPORT TOWARDS BOTTOM-RIGHT TO SIMULATE MOTION 
    int shiftX = 30;
    int shiftY = 40;

    Rect shiftedRect(viewRect.x + shiftX, viewRect.y + shiftY, viewW, viewH);
    Mat shifted = padded(shiftedRect).clone();

    double bestScore = -2.0;

    Point bestLoc;

	//RUNNING THE SLIDING WINDOW 
    for (int v = 0; v <= shifted.rows - templ.rows; ++v) 
    {
        for (int u = 0; u <= shifted.cols - templ.cols; ++u) 
        {
            Rect window(u, v, templ.cols, templ.rows);
            Mat patch = shifted(window);

            double score = compute_ncc(patch, templ);

            if (score > bestScore) 
            {
                bestScore = score;
                bestLoc = Point(u, v);
            }
        }
    }

    cout << "Best match at >>> " << bestLoc << "\n";
    cout << "Best NCC score >>> " << bestScore << "\n";

    //MAKING A RECTANGLES AROUND THE SELECTION
    rectangle(shifted, Rect(bestLoc, templ.size()), Scalar(255), 2);
    
    //MANAGING THE WINDOWS
    imshow("Template", templ);
    imshow("Shifted Viewport with Match", shifted);
    waitKey(0);

    return 0;
}
