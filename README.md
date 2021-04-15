# FeaturesBenchmark

Hongyu Ge, biomed ssSEM LAB, SIBET, Suzhou, China

### Tips

1. All algorithms are OpenCV C++ implementation.

2. To run all tests, you have to install OpenCV with the following options:

   ```
   -DOPENCV_EXTRA_MODULES_PATH={prefix}/opencv_contrib-master/modules
   -DWITH_CUDA=ON
   -DOPENCV_ENABLE_NONFREE=ON
   ```

3. The following results are tested on:

   System: Ubuntu18.04

   CPU: Intel Core i9-9920X (12c24t)

   GPU: Nvidia RTX 2070 Super (8GB Mem.)

### Exp.1 The speed of detecting features

##### Parameters: 

orb & orb_cuda: nfeatures=100000

other: default

|           |  2048*2048   |  4096*4096  | 12288*12288 |
| :-------: | :----------: | :---------: | :---------: |
|   SIFT    |   0.825380   |   3.24006   |   30.9637   |
|   SURF    |   0.418311   |   1.34376   |   9.88681   |
|    ORB    |   0.249954   |   0.87945   |   6.61783   |
| SURF_CUDA |   0.155271   |   0.27863   |   1.34502   |
| ORB_CUDA  | **0.039196** | **0.07346** | **0.35331** |
|   AKAZE   |   0.499058   |   2.04194   |   10.0434   |

### Exp.2 Detected features' number

##### Parameters:

orb & orb_cuda: nfeatures=100000

other: default

|           | 2048*2048  | 4096*4096  | 12288*12288 |
| :-------: | :--------: | :--------: | :---------: |
|   SIFT    | **106285** | **555219** |   2335809   |
|   SURF    |   84007    |   291089   |   2134577   |
|    ORB    |  *100000*  |  *100000*  |  *100000*   |
| SURF_CUDA |   41943    |  *65535*   |   *65535*   |
| ORB_CUDA  |   97264    |  *100000*  |  *100000*   |
|   AKAZE   |   83912    |   357588   | **2757210** |

*10000* : Set by the input arguments.

*65535* : The maximum value set by OpenCV and reason is unknown.

### Exp.3 Matched features' number and ratio

##### Parameters:

orb & orb_cuda: nfeatures=100000

SIFT/SURF matching method: FlannBasedMatcher

ORB/ORB_CUDA/AKAZE matching method: descriptors converted to CV_32F

SURF_CUDA matching method: CUDA_FlannBasedMatcher

other: default

|           |   2048*2048    |    4096*4096    |   12288*12288   |
| :-------: | :------------: | :-------------: | :-------------: |
|   SIFT    | **6014/5.66%** | **29285/5.27%** |   46439/1.99%   |
|   SURF    |   2726/3.25%   |   11750/4.04%   |   46188/2.16%   |
|    ORB    |   866/0.87%    |   1422/1.42%    |   1354/1.35%    |
| SURF_CUDA |   1037/2.47%   |   2507/3.83%    |    447/0.68%    |
| ORB_CUDA  |   203/0.21%    |    399/0.40%    |    497/0.50%    |
|   AKAZE   |   2608/3.11%   |   13823/3.87%   | **77164/2.80%** |

### Exp.4 Inliers' number and ratio

##### Parameters:

orb & orb_cuda: nfeatures=100000

SIFT/SURF matching method: FlannBasedMatcher

ORB/ORB_CUDA/AKAZE matching method: descriptors converted to CV_32F and then use FlannBasedMatcher

SURF_CUDA matching method: CUDA_FlannBasedMatcher

Filter: RANSAC

RANSAC threshold: 0.95

other: default

|           |    2048*2048    |    4096*4096     |   12288*12288    |
| :-------: | :-------------: | :--------------: | :--------------: |
|   SIFT    | **5868/97.57%** | **25562**/87.29% |   17619/37.94%   |
|   SURF    |   2417/88.66%   |   9552/81.29%    |   12503/27.07%   |
|    ORB    |   387/44.69%    |    622/43.74%    |    141/10.41%    |
| SURF_CUDA |   984/94.80%    | 2343/**93.46%**  |  338/**75.62%**  |
| ORB_CUDA  |    53/27.46%    |    208/52.13%    |    86/17.30%     |
|   AKAZE   |   2522/96.52%   |   11943/86.40%   | **25699**/33.30% |