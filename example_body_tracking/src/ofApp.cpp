#include "ofApp.h"
#include "ofxCv.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofLogToConsole();
    ofSetFrameRate(60);
    ofSetVerticalSync(true);

    cam.setupPerspective();
    cam.enableMouseInput();
    cam.enableMouseMiddleButton();

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

    cam.begin();

    auto keypoints3d = maxine.get_keypoints3D();

    for(size_t i=0; i<keypoints3d.size(); ++i){
        ofVec3f v = keypoints3d.at(i);
        ofPushStyle();
        ofSetColor(0, 255, 0);
        ofPushMatrix();
        ofTranslate(v);
        ofDrawBox(10, 10, 10);
        ofPopMatrix();
        ofPopStyle();
    }

    cam.end();

    auto keypoints2d = maxine.get_keypoints();
    for(size_t i=0; i<keypoints2d.size(); ++i){
        ofVec2f v = keypoints2d.at(i);
        ofPushStyle();
        ofSetColor(255, 0, 0);
        ofDrawRectangle(v.x - 5, v.y - 5, 10, 10);
        ofPopStyle();
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