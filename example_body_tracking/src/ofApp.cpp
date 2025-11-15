#include "ofApp.h"
#include "ofxCv.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofLogToConsole();
    ofSetFrameRate(60);
    ofSetVerticalSync(true);

    video.setup(640, 480);
    maxine.setup(video);
}

//--------------------------------------------------------------
void ofApp::update(){
    video.update();
    maxine.update(video);
}

//--------------------------------------------------------------
void ofApp::draw(){
    video.draw(0, 0);

    auto keypoints = maxine.get_keypoints3D();

    for(size_t i=0; i<keypoints.size(); ++i){
        ofVec3f vec = keypoints.at(i);
        for(size_t j=0; j<3; ++j){
            float v = vec.x;
            if(j==1){
                v = vec.y;
            }else if(j==2){
                v = vec.z;
            }

            ofPushStyle();
            ofSetColor(ofMap(v, -10, 10, 0, 255, true), 0, 0);
            ofDrawRectangle(i * 10, 500 + 10 * j, 10, 10);
            ofPopStyle();
        }
    }


    ofDrawBitmapStringHighlight(ofToString(ofGetFrameRate(), 2, 0, '\0'), 20, 20);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}