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

    auto expression_coefficients = maxine.get_expressions();

    for(size_t i=0; i<expression_coefficients.size(); ++i){
        float v = expression_coefficients.at(i);
        ofPushStyle();
        ofSetColor(ofMap(v, 0, 1, 0, 255, true), 0, 0);
        ofDrawRectangle(i * 10, 500, 10, 10);
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