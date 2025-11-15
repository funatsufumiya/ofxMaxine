/****************************************************************
 * Copyright (c) 2025 Fumiya Funatsu
 * Copyright (c) 2024 Ryan Powell
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 ****************************************************************/

#include "ofxMaxine.h"
#include "ofxCv.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvAR.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"

#if CV_MAJOR_VERSION >= 4
  #define CV_CAP_PROP_FPS           cv::CAP_PROP_FPS
  #define CV_CAP_PROP_FRAME_COUNT   cv::CAP_PROP_FRAME_COUNT
  #define CV_CAP_PROP_FRAME_HEIGHT  cv::CAP_PROP_FRAME_HEIGHT
  #define CV_CAP_PROP_FRAME_WIDTH   cv::CAP_PROP_FRAME_WIDTH
  #define CV_CAP_PROP_POS_FRAMES    cv::CAP_PROP_POS_FRAMES
  #define CV_INTER_AREA             cv::INTER_AREA
  #define CV_INTER_LINEAR           cv::INTER_LINEAR
#endif // CV_MAJOR_VERSION

ofxMaxine::ofxMaxine() { 
  // initialize members
  _landmarks.clear();
  _expressions.clear();
  _landmarkConfidence.clear();
  _expressionOutputBboxData.clear();
  _referencePose.clear();

  _landmarkCount = 0;
  _exprCount = 0;
  _numKeyPoints = 0;
  _globalExpressionParam = 1.0f;
  
  _pose.rotation = NvAR_Quaternion{0.0f, 0.0f, 0.0f, 1.0f};
  _pose.translation = NvAR_Vector3f{0.0f, 0.0f, 0.0f};

  _expressionOutputBboxData.resize(25, {0.0, 0.0, 0.0, 0.0});
  _expressionOutputBboxes.boxes = _expressionOutputBboxData.data();
  _expressionOutputBboxes.max_boxes = static_cast<uint8_t>(_expressionOutputBboxData.size());
  _expressionOutputBboxes.num_boxes = 0;
  
  nvErr = NvAR_CudaStreamCreate(&_expressionStream);
  if (nvErr!=NVCV_SUCCESS) {
    ofLogError("ofxMaxine") << ("failed to create expression CUDA stream");
  }

  nvErr = NvAR_CudaStreamCreate(&_bodyStream);
  if (nvErr!=NVCV_SUCCESS) {
    ofLogError("ofxMaxine") << ("failed to create body CUDA stream");
  }

  nvErr = NvAR_CudaStreamCreate(&_gazeStream);
  if (nvErr!=NVCV_SUCCESS) {
    ofLogError("ofxMaxine") << ("failed to create gaze CUDA stream");
  }

}

ofxMaxine::~ofxMaxine() {
  // continue_processing = false;
  // if (processing_thread.joinable()) {
  //   processing_thread.join();
  // }
  
  
  if (_expressionStream) {
    nvErr = NvAR_CudaStreamDestroy(_expressionStream);
    if (nvErr!=NVCV_SUCCESS) {
      ofLogError("ofxMaxine") << ("failed to destroy expression CUDA stream");
    }
  }

  if (_bodyStream) {
    nvErr = NvAR_CudaStreamDestroy(_bodyStream);
    if (nvErr!=NVCV_SUCCESS) {
      ofLogError("ofxMaxine") << ("failed to destroy body CUDA stream");
    }
  }

  if (_gazeStream) {
    nvErr = NvAR_CudaStreamDestroy(_gazeStream);
    if (nvErr!=NVCV_SUCCESS) {
      ofLogError("ofxMaxine") << ("failed to destroy gaze CUDA stream");
    }
  }
  
  // if (_vidIn.isOpened()) {
  //   _vidIn.release();
  // }

  // Deallocate NvCVImage objects
  NvCVImage_Dealloc(&_srcGpu);
  NvCVImage_Dealloc(&_srcImg);

  // Destroy feature handle
  if (_expressionFeature) {
    NvAR_Destroy(_expressionFeature);
    _expressionFeature = nullptr;
  }

  if (_bodyFeature) {
    NvAR_Destroy(_bodyFeature);
    _bodyFeature = nullptr;
  }

  if (_gazeFeature) {
    NvAR_Destroy(_gazeFeature);
    _gazeFeature = nullptr;
  }

  _bodyOutputBboxData.clear();
  _expressionOutputBboxData.clear();
  _landmarks.clear();
  _landmarkConfidence.clear();
  _keypoints.clear();
  _keypoints3D.clear();
  _keypoints_confidence.clear();
  _expressions.clear();
  _expressionScale.clear();
  _expressionZeroPoint.clear();
  _expressionExponent.clear();
  _eigenvalues.clear();
  _jointAngles.clear();
  _referencePose.clear();
  _bodyOutputBboxConfData.clear();
  
  _ocvSrcImg.release();
  // _processingFrame.release();
}

void ofxMaxine::setup(ofVideoGrabber& grabber) {
  setup(ofxCv::toCv(grabber.getPixels()));
}

void ofxMaxine::setup(cv::Mat image) {

    // // get model directory
    // const char* _model_path = getenv("NVAR_MODEL_DIR");
    // if (_model_path) {
    //   modelPath = _model_path;
    //   ofLogNotice("ofxMaxine") << ("NVAR model dir env var located at: ");
    //   ofLogNotice("ofxMaxine") << (modelPath.c_str());

    // } else {
    //   ofLogNotice("ofxMaxine") << ("failed to located NVAR model dir env var");
    // }

    // open video capture
    // if (_vidIn.open(_camera_device_id)) {
    //   ofLogNotice("ofxMaxine") << ("successfully opened video capture");

    //   // intialize face expression feature
    //   unsigned width, height, frame_rate;
    //   width = (unsigned)_vidIn.get(CV_CAP_PROP_FRAME_WIDTH);
    //   height = (unsigned)_vidIn.get(CV_CAP_PROP_FRAME_HEIGHT);
    //   frame_rate = (unsigned)_vidIn.get(CV_CAP_PROP_FPS);

    float width = image.size().width;
    float height = image.size().height;

    //   static const int fps_precision = FPS_PRECISION; // round frame rate for opencv compatibility
    //   frame_rate = static_cast<int>((frame_rate + 0.5) * fps_precision) / static_cast<double>(fps_precision);

    //   ofLogNotice("ofxMaxine") << ("video width: ", ofToString(width));
    //   ofLogNotice("ofxMaxine") << ("video height: ", ofToString(height));
    //   ofLogNotice("ofxMaxine") << ("video FPS: ", ofToString(frame_rate));

      _expressionFiltering = 0x037; // bitfield, default, all on except 0x100 enhaced closures
      _poseMode = 1; // 0 - 3DOF implicit for only rotation, 1 - 6DOF explicit for head position
      _enableCheekPuff = 0; // experimental, 0 - off, 1 - on
      
      _bodyTrackMode = 0; // 0 - High Quality, 1 - High Performance
      _bodyFiltering = 1; // 0 - disabled, 1 - enabled
      _bodyUseCudaGraph = true;

      _gazeSensitivity = 3; // Unsigned integer in the range of 2-5 to increase the sensitivity of the algorithm to the redirected eye size. 2 uses a smaller eye region and 5 uses a larger eye size. (<-- from docs)
      _gazeFiltering = -1; // unsigned int, 1 - enabled, 0 - diabled, (-1 is all filtering? from comments in sample)
      _gazeRedirect = false; 
      _gazeUseCudaGraph = false; 

      // allocate src images
      nvErr = NvCVImage_Alloc(&_srcGpu, width, height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to allocate srcGpu NvImage memory");
      }

      nvErr = NvCVImage_Alloc(&_srcImg, width, height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU_PINNED, 0);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to allocate srcImg NvImage memory");
      }

      CVWrapperForNvCVImage(&_srcImg, &_ocvSrcImg);
      _landmarkCount = NUM_LANDMARKS;


      // create features
      nvErr = NvAR_Create(NvAR_Feature_FaceExpressions, &_expressionFeature);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to create facial expression feature handle");
      }

      nvErr = NvAR_Create(NvAR_Feature_BodyPoseEstimation, &_bodyFeature);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to create body track feature handle");
      }

      nvErr = NvAR_Create(NvAR_Feature_GazeRedirection, &_gazeFeature);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to create gaze track feature handle");
      }


      // set feature cuda streams
      nvErr = NvAR_SetCudaStream(_expressionFeature, NvAR_Parameter_Config(CUDAStream), _expressionStream);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set CUDA stream for facial expression feature handle");
      }

      nvErr = NvAR_SetCudaStream(_bodyFeature, NvAR_Parameter_Config(CUDAStream), _bodyStream);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set CUDA stream for body track feature handle");
      }
      
      nvErr = NvAR_SetCudaStream(_gazeFeature, NvAR_Parameter_Config(CUDAStream), _gazeStream);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set CUDA stream for gaze track feature handle");
      }


      // set temporal filtering
      nvErr = NvAR_SetU32(_expressionFeature, NvAR_Parameter_Config(Temporal), _expressionFiltering);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set temporal filtering for facial expression feature handle");
      }

      nvErr = NvAR_SetU32(_bodyFeature, NvAR_Parameter_Config(Temporal), _bodyFiltering);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set temporal filtering for body track feature handle");
      }

      nvErr = NvAR_SetU32(_gazeFeature, NvAR_Parameter_Config(Temporal), _gazeFiltering);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set temporal filtering for gaze track feature handle");
      }

      // set facial expression config
      nvErr = NvAR_SetU32(_expressionFeature, NvAR_Parameter_Config(PoseMode), _poseMode);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set pose mode for facial expression feature handle");
      }

      nvErr = NvAR_SetU32(_expressionFeature, NvAR_Parameter_Config(EnableCheekPuff), _enableCheekPuff);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set enable cheek puff for facial expression feature handle");
      }

      // set body track config
      nvErr = NvAR_SetU32(_bodyFeature, NvAR_Parameter_Config(Mode), _bodyTrackMode);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set tracking mode for body tracking feature handle");
      }

      nvErr = NvAR_SetF32(_bodyFeature, NvAR_Parameter_Config(UseCudaGraph), _bodyUseCudaGraph);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set use CUDA grpah for body tracking feature handle");
      }

      // set gaze track config
      nvErr = NvAR_SetU32(_gazeFeature, NvAR_Parameter_Config(GazeRedirect), _gazeRedirect);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set enable redirect for gaze tracking feature handle");
      }

      nvErr = NvAR_SetU32(_gazeFeature, NvAR_Parameter_Config(UseCudaGraph), _gazeUseCudaGraph);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set sensitivity for gaze tracking feature handle");
      }
     
      nvErr = NvAR_SetU32(_gazeFeature, NvAR_Parameter_Config(EyeSizeSensitivity), _gazeSensitivity);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set sensitivity for gaze tracking feature handle");
      }

      // load features
      nvErr = NvAR_Load(_expressionFeature);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to load facial expression feature handle");
      }

      nvErr = NvAR_Load(_bodyFeature);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to load body tracking feature handle");
      }

      nvErr = NvAR_Load(_gazeFeature);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to load gaze tracking feature handle");
      }

      // set feature IO
      _expressionOutputBboxData.assign(25, {0.f, 0.f, 0.f, 0.f});
      _expressionOutputBboxes.boxes = _expressionOutputBboxData.data();
      _expressionOutputBboxes.max_boxes = (uint8_t)_expressionOutputBboxData.size();
      _expressionOutputBboxes.num_boxes = 0;
      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Output(BoundingBoxes), &_expressionOutputBboxes, sizeof(NvAR_BBoxes));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set bounding boxes output for facial expression feature handle");
      }

      _landmarks.resize(_landmarkCount);
      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Output(Landmarks), _landmarks.data(), sizeof(NvAR_Point2f));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set landmarks output for facial expression feature handle");
      }

      _landmarkConfidence.resize(_landmarkCount);
      nvErr = NvAR_SetF32Array(_expressionFeature, NvAR_Parameter_Output(LandmarksConfidence), _landmarkConfidence.data(), _landmarkCount);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set landmark confidence output for facial expression feature handle");
      }

      // get feature counts
      nvErr = NvAR_GetU32(_expressionFeature, NvAR_Parameter_Config(ExpressionCount), &_exprCount);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to get expression count for facial expression feature handle");
      }
      _expressions.resize(_exprCount);
      _expressionZeroPoint.resize(_exprCount, 0.0f);
      _expressionScale.resize(_exprCount, 1.0f);
      _expressionExponent.resize(_exprCount, 1.0f);

      nvErr = NvAR_GetU32(_bodyFeature, NvAR_Parameter_Config(NumKeyPoints), &_numKeyPoints);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to get keypoint count for boody tracking feature handle");
      } else {
        ofLogNotice("ofxMaxine") << ("keypoints retrieved: ", ofToString(_numKeyPoints));
      }

      _keypoints.assign(_numKeyPoints, { 0.f, 0.f });
      _keypoints3D.assign(_numKeyPoints, { 0.f, 0.f, 0.f });
      _jointAngles.assign(_numKeyPoints, { 0.f, 0.f, 0.f, 1.f });
      _keypoints_confidence.assign(_numKeyPoints, 0.f);
      _referencePose.assign(_numKeyPoints, { 0.f, 0.f, 0.f });

      const void* pReferencePose;
      nvErr = NvAR_GetObject(_bodyFeature, NvAR_Parameter_Config(ReferencePose), &pReferencePose,
                            sizeof(NvAR_Point3f));
      memcpy(_referencePose.data(), pReferencePose, sizeof(NvAR_Point3f) * _numKeyPoints);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to get reference pose for boody tracking feature handle");
      }


      nvErr = NvAR_SetF32Array(_expressionFeature, NvAR_Parameter_Output(ExpressionCoefficients), _expressions.data(), _exprCount);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set expression coefficient output for facial expression feature handle");
      }

      // set feature inputs
      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Input(Image), &_srcGpu, sizeof(NvCVImage));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set image input for facial expression feature handle");
      }
      
      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Input(Image), &_srcGpu, sizeof(NvCVImage));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set image input for body tracking feature handle");
      }

      _cameraIntrinsicParams[0] = static_cast<float>(_srcGpu.height);
      _cameraIntrinsicParams[1] = static_cast<float>(_srcGpu.width) / 2.0f;
      _cameraIntrinsicParams[2] = static_cast<float>(_srcGpu.height) / 2.0f;

      nvErr = NvAR_SetF32Array(_expressionFeature, NvAR_Parameter_Input(CameraIntrinsicParams), _cameraIntrinsicParams, NUM_CAMERA_INTRINSIC_PARAMS);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set camera intrinsic params for facial expression feature handle");
      }

      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Output(Pose), &_pose.rotation, sizeof(NvAR_Quaternion));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set pose rotation output for facial expression feature handle");
      }

      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Output(PoseTranslation), &_pose.translation, sizeof(NvAR_Vector3f));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set pose translation output for facial expression feature handle");
      }

      // finish body track output, TODO reorder this once it is all working
      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Output(KeyPoints), _keypoints.data(), sizeof(NvAR_Point2f));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set keypoints output for body track feature handle");
      }

      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Output(KeyPoints3D), _keypoints3D.data(), sizeof(NvAR_Point3f));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set keypoints3D output for body track feature handle");
      }

      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Output(JointAngles), _jointAngles.data(), sizeof(NvAR_Quaternion));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set joint angles output for body track feature handle");
      }

      nvErr = NvAR_SetF32Array(_bodyFeature, NvAR_Parameter_Output(KeyPointsConfidence), _keypoints_confidence.data(), sizeof(float));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set keypoints confidence output for body track feature handle");
        ofLogError("ofxMaxine") << ("ERROR CODE: ", ofToString(nvErr));
      }

      _bodyOutputBboxData.assign(25, { 0.f, 0.f, 0.f, 0.f });
      _bodyOutputBboxConfData.assign(25, 0.f);
      _bodyOutputBboxes.boxes = _bodyOutputBboxData.data();
      _bodyOutputBboxes.max_boxes = (uint8_t)_bodyOutputBboxData.size();
      _bodyOutputBboxes.num_boxes = 0;

      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Output(BoundingBoxes), &_bodyOutputBboxes, sizeof(NvAR_BBoxes));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set bounding box output for body track feature handle");
      }

      nvErr = NvAR_SetF32Array(_bodyFeature, NvAR_Parameter_Output(BoundingBoxesConfidence), _bodyOutputBboxConfData.data(), _bodyOutputBboxes.max_boxes);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set bounding box confidence output for body track feature handle");
      }

      // gaze IO
      nvErr = NvAR_SetS32(_gazeFeature, NvAR_Parameter_Input(Width), width);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set image width input for gaze track feature handle");
      }

      nvErr = NvAR_SetS32(_gazeFeature, NvAR_Parameter_Input(Height), height);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set image height input for gaze track feature handle");
      }

      nvErr = NvAR_SetObject(_gazeFeature, NvAR_Parameter_Input(Image), &_srcGpu, sizeof(NvCVImage));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set image input for gaze track feature handle");
      }

      nvErr = NvAR_SetF32Array(_gazeFeature, NvAR_Parameter_Output(OutputGazeVector), _gaze_angles_vector, 2);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set gaze vector output for gaze track feature handle");
        ofLogError("ofxMaxine") << ("ERROR CODE: ", ofToString(nvErr));
      }

      nvErr = NvAR_SetObject(_gazeFeature, NvAR_Parameter_Output(GazeDirection), _gaze_direction, sizeof(NvAR_Point3f));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set gaze direction output for gaze track feature handle");
        ofLogError("ofxMaxine") << ("ERROR CODE: ", ofToString(nvErr));
      }

      nvErr = NvAR_GetU32(_gazeFeature, NvAR_Parameter_Config(Landmarks_Size), &_gazeNumLandmarks);
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to get number of gaze landmarks for gaze track feature handle");
        ofLogError("ofxMaxine") << ("ERROR CODE: ", ofToString(nvErr));
      }

      _gazeFacialLandmarks.assign(_gazeNumLandmarks, {0.f, 0.f});
      nvErr = NvAR_SetObject(_gazeFeature, NvAR_Parameter_Output(Landmarks), _gazeFacialLandmarks.data(), sizeof(NvAR_Point2f));
      if (nvErr!=NVCV_SUCCESS) {
        ofLogError("ofxMaxine") << ("failed to set gaze landmarks output for gaze track feature handle");
        ofLogError("ofxMaxine") << ("ERROR CODE: ", ofToString(nvErr));
      }

      
      image.copyTo(_ocvSrcImg); // WORKAROUND

      // capture image
      // if (_vidIn.read(_ocvSrcImg)) {
        // process image
        nvErr = NvCVImage_Transfer(&_srcImg, &_srcGpu, 1.f, _expressionStream, nullptr);
        if (nvErr!=NVCV_SUCCESS) {
          ofLogError("ofxMaxine") << ("failed to transfer image to gpu");
        }

        nvErr = NvAR_Run(_expressionFeature);
        if (nvErr!=NVCV_SUCCESS) {
          ofLogError("ofxMaxine") << ("failed to run facial expression feature");
        } else {
          normalizeExpressionsWeights();

          // // start capture on separate thread
          // start_processing_thread();

          ofLogNotice("ofxMaxine") << ("successfully initialized facial expression feature");
        }

        nvErr = NvAR_Run(_bodyFeature);
        if (nvErr!=NVCV_SUCCESS) {
          ofLogError("ofxMaxine") << ("failed to run body tracking feature");
        } else {
          ofLogNotice("ofxMaxine") << ("successfully initialized body tracking feature");
        }

        nvErr = NvAR_Run(_gazeFeature);
        if (nvErr!=NVCV_SUCCESS) {
          ofLogError("ofxMaxine") << ("failed to run gaze tracking feature");
        } else {
          ofLogNotice("ofxMaxine") << ("successfully initialized gaze tracking feature");
        }


      // } else {
      //   ofLogError("ofxMaxine") << ("failed to capture video frame");
      // }
    // } else {
    //   ofLogError("ofxMaxine") << ("failed to open video capture");
    // }
}

void ofxMaxine::printCapture() {
  // validate expression capture
  printPoseRotation();

  if (_poseMode == 1) {
    printPoseTranslation();
  }
  
  printExpressionCoefficients();
  printLandmarkLocations();
  printLandmarkConfidence();
  printBoundingBoxes();
}

void ofxMaxine::update(ofVideoGrabber& grabber) {
  update(ofxCv::toCv(grabber.getPixels()));
}

void ofxMaxine::update(cv::Mat image) {
    {
        image.copyTo(_ocvSrcImg); // WORKAROUND

        // // lock to ensure thread-safe access to _ocvSrcImg
        // std::lock_guard<std::mutex> lock(processing_mutex);

        // assumes _ocvSrcImg has already been updated by processing_loop
        // transfer image to GPU
        nvErr = NvCVImage_Transfer(&_srcImg, &_srcGpu, 1.f, _expressionStream, nullptr);


        if (nvErr!=NVCV_SUCCESS) {
            ofLogNotice("ofxMaxine") << ("failed to transfer image to gpu");
            return;
        }


        // run gaze tracking feature
        nvErr = NvAR_Run(_gazeFeature);
        if (nvErr!=NVCV_SUCCESS) {
          ofLogNotice("ofxMaxine") << ("failed to run gaze tracking feature");
        }

        // run body track feature
        nvErr = NvAR_Run(_bodyFeature);
        if (nvErr!=NVCV_SUCCESS) {
          ofLogNotice("ofxMaxine") << ("failed to run body tracking feature");
        } 

        // run facial expression feature
        nvErr = NvAR_Run(_expressionFeature);
        if (nvErr!=NVCV_SUCCESS) {
            ofLogNotice("ofxMaxine") << ("failed to run facial expression feature");
            return;
        }

        normalizeExpressionsWeights();
        
        // if(_show_capture){
        //   cv::imshow("Src Image", _ocvSrcImg);
        // }
    }

    // additional process logic
}


void ofxMaxine::printPoseRotation() {
  ofLogNotice("ofxMaxine") << ("Facial Pose Rotation:");
  
  const auto& rotation = _pose.rotation;
  std::string rotationStr = "Pose Rotation ofQuaternion: (" 
  + ofToString(rotation.x) + ", " 
  + ofToString(rotation.y) + ", " 
  + ofToString(rotation.z) + ", " 
  + ofToString(rotation.w) + ")";
  ofLogNotice("ofxMaxine") << (rotationStr);
}

void ofxMaxine::printExpressionCoefficients() {
  ofLogNotice("ofxMaxine") << ("Facial Expression Coefficients:");

  std::string coeffsStr = "";
  for (size_t i = 0; i < _expressions.size(); ++i) {
      coeffsStr += ofToString(_expressions[i]);
      if (i < _expressions.size() - 1) {
          coeffsStr += ", ";
      }
  }
  ofLogNotice("ofxMaxine") << (coeffsStr);
}

void ofxMaxine::printLandmarkLocations() {
  ofLogNotice("ofxMaxine") << ("Facial Landmark Locations:");
  
  std::string landmarksStr = "";
  for (size_t i = 0; i < _landmarks.size(); ++i) {
      landmarksStr += "(" + ofToString(_landmarks[i].x) + ", " 
                          + ofToString(_landmarks[i].y) + ")";
      if (i < _landmarks.size() - 1) {
          landmarksStr += ", ";
      }
  }
  ofLogNotice("ofxMaxine") << (landmarksStr);
}

void ofxMaxine::printBoundingBoxes() {
  ofLogNotice("ofxMaxine") << ("Bounding Boxes: (x, y, width, height)");

  std::string bboxesStr = "";
  for (size_t i = 0; i < _expressionOutputBboxes.num_boxes; ++i) {
    const auto& box = _expressionOutputBboxes.boxes[i];
    bboxesStr += "("+ ofToString(box.x) + ", " 
                    + ofToString(box.y) + ", " 
                    + ofToString(box.width) + ", " 
                    + ofToString(box.height) + ")";
    if (i < _expressionOutputBboxes.num_boxes - 1) {
      bboxesStr += ", ";
    }
  }
  ofLogNotice("ofxMaxine") << (bboxesStr);
}

void ofxMaxine::printLandmarkConfidence() {
  ofLogNotice("ofxMaxine") << ("Facial Landmark Confidence:");

  std::string confidenceStr = "";
  for (size_t i = 0; i < _landmarkConfidence.size(); ++i) {
    confidenceStr += ofToString(_landmarkConfidence[i]);
    if (i < _landmarkConfidence.size() - 1) {
      confidenceStr += ", ";
    }
  }
  ofLogNotice("ofxMaxine") << (confidenceStr);
}

void ofxMaxine::printPoseTranslation() {
  ofLogNotice("ofxMaxine") << ("Facial Pose Translation:");

  const auto& translation = _pose.translation;
  std::string translationStr = "(" 
    + ofToString(translation.vec[0]) + ", "   // X component
    + ofToString(translation.vec[1]) + ", "   // Y component
    + ofToString(translation.vec[2]) + ")";   // Z component
  ofLogNotice("ofxMaxine") << (translationStr);
}

std::vector<ofVec2f> ofxMaxine::get_landmarks() const {
  std::vector<ofVec2f> landmarks;
  for (const auto& landmark : _landmarks) {
    landmarks.push_back(ofVec2f(landmark.x, landmark.y));
  }
  return landmarks;
}

int ofxMaxine::get_landmark_count() const {
  return _landmarkCount;
}

int ofxMaxine::get_expression_count() const {
  return _exprCount;
}

std::vector<float> ofxMaxine::get_expressions() const {
    return _expressions;
}

std::vector<float> ofxMaxine::get_landmark_confidence() const {
  return _landmarkConfidence;
}

ofQuaternion ofxMaxine::get_pose_rotation() const {
  const auto& rotation = _pose.rotation;
  return ofQuaternion(rotation.x, rotation.y, rotation.z, rotation.w);
}


ofVec3f ofxMaxine::get_pose_translation() const {
  if (_poseMode == 1) {
    const auto& translation = _pose.translation;
    return ofVec3f(translation.vec[0], translation.vec[1], translation.vec[2]);
  } else {
    return ofVec3f(0, 0, 0);
  }
}

ofMatrix4x4 ofxMaxine::get_pose_transform() const {
    ofQuaternion rotation_quat = get_pose_rotation();
    rotation_quat.normalize();
    ofVec3f translation = get_pose_translation();

    glm::quat quat = rotation_quat;
    glm::vec3 trans = translation;

    // auto result = glm::rotate(quat, trans);

    return glm::translate(glm::toMat4(quat), trans);
}

ofRectangle ofxMaxine::bounding_box_to_rect(const NvAR_Rect& box) const {
    ofRectangle bbox;
    bbox.x = box.x;
    bbox.x = box.y;
    bbox.width = box.width;
    bbox.height = box.height;
    return bbox;
}

std::vector<ofRectangle> ofxMaxine::get_bounding_boxes() const {
  std::vector<ofRectangle> boxes;
  for (size_t i = 0; i < _expressionOutputBboxes.num_boxes; ++i) {
    boxes.push_back(bounding_box_to_rect(_expressionOutputBboxes.boxes[i]));
  }
  return boxes;
}

// void ofxMaxine::start_processing_thread() {
//     continue_processing = true;
//     processing_thread = std::thread(&ofxMaxine::processing_loop, this);
// }

// void ofxMaxine::processing_loop() {
//   while (continue_processing) {
//     if (_vidIn.read(_processingFrame)) {
//       {
//         std::lock_guard<std::mutex> lock(processing_mutex);
//         // copy async frame data to _ocvSrcImg
//         _processingFrame.copyTo(_ocvSrcImg); 
//       }
//     } else {
//       ofLogNotice("ofxMaxine") << ("failed to read frame");
//     }
//   }
// }

void ofxMaxine::normalizeExpressionsWeights() {
  assert(_expressions.size() == _exprCount);
  assert(_expressionScale.size() == _exprCount);
  assert(_expressionZeroPoint.size() == _exprCount);

  for (size_t i = 0; i < _exprCount; i++) {
    float tempExpr = _expressions[i];
    // Normalize expression based on zero point and scale
    _expressions[i] = 1.0f - std::pow(1.0f - (std::max(_expressions[i] - _expressionZeroPoint[i], 0.0f) * _expressionScale[i]),
                                      _expressionExponent[i]);
    // Blend with the previous value using a global parameter
    _expressions[i] = _globalExpressionParam * _expressions[i] + (1.0f - _globalExpressionParam) * tempExpr;
  }
}

// Function to convert NvAR_Point2f to Godot Vector2
ofVec2f point2f_to_vector2(const NvAR_Point2f& point) {
    return ofVec2f(point.x, point.y);
}

// Function to convert NvAR_Point3f to Godot ofVec3f
ofVec3f point3f_to_ofVec3f(const NvAR_Point3f& point) {
    return ofVec3f(point.x, point.y, point.z);
}

// Function to convert NvAR_ofQuaternion to Godot ofQuaternion
ofQuaternion Quaternion_to_of(const NvAR_Quaternion& quat) {
    return ofQuaternion(quat.x, quat.y, quat.z, quat.w);
}

std::vector<ofVec2f> ofxMaxine::get_keypoints() const {
    std::vector<ofVec2f> keypoints;
    for (const auto& kp : _keypoints) {
        keypoints.push_back(point2f_to_vector2(kp));
    }
    return keypoints;
}

std::vector<ofVec3f> ofxMaxine::get_keypoints3D() const {
    std::vector<ofVec3f> keypoints3D;
    for (const auto& kp : _keypoints3D) {
        keypoints3D.push_back(point3f_to_ofVec3f(kp));
    }
    return keypoints3D;
}

std::vector<ofQuaternion> ofxMaxine::get_joint_angles() const {
    std::vector<ofQuaternion> jointAngles;
    for (const auto& ja : _jointAngles) {
        jointAngles.push_back(Quaternion_to_of(ja));
    }
    return jointAngles;
}

std::vector<float> ofxMaxine::get_keypoints_confidence() const {
    return _keypoints_confidence;
}

std::vector<ofRectangle> ofxMaxine::get_body_bounding_boxes() const {
    std::vector<ofRectangle> boxes;
    for (size_t i = 0; i < _bodyOutputBboxes.num_boxes; ++i) {
        boxes.push_back(bounding_box_to_rect(_bodyOutputBboxes.boxes[i]));
    }
    return boxes;
}

std::vector<float> ofxMaxine::get_body_bounding_box_confidence() const {
    return _bodyOutputBboxConfData;
}

std::vector<float> ofxMaxine::get_gaze_angles_vector() const {
  std::vector<float> gaze_angles;
  for (float angle : _gaze_angles_vector) {
    gaze_angles.push_back(angle);
  }
  return gaze_angles;
}

ofVec3f ofxMaxine::get_gaze_direction() const {
  return ofVec3f(_gaze_direction->x, _gaze_direction->y, _gaze_direction->z);
}

// void ofxMaxine::set_show_capture(const bool p_should_show){
//   _show_capture = p_should_show;
// }

// bool ofxMaxine::get_show_capture() const {
//   return _show_capture;
// }
