#American Sign Language Recognition using MediaPipe and Random Forest Classifier — Intel oneAPI Optimised Scikit-Learn Library
![Untitled design (4)](https://user-images.githubusercontent.com/100186186/226212948-2793b61d-1fba-4f6e-b0be-ad526b263154.png)
## 研究背景與動機 Research Background and Motivation
隨著多元溝通方式的需求上升，手語作為聽障人士與社會溝通的重要橋梁，其辨識技術也逐漸受到重視。然而，多數現有系統僅支援靜態手勢識別，無法處理實際溝通中常見的「連續動作」與「單字拼寫」。我們希望透過本專題，整合機器學習模型與視覺擷取技術，打造一套能即時辨識手語英文字母的互動式系統，協助一般人與手語使用者溝通，並提升手語學習的便利性。<br>
As the demand for diverse communication methods continues to rise, sign language has become an essential bridge for communication between the hearing-impaired community and society. However, most existing systems only support static gesture recognition and are unable to handle commonly used features in real-life communication, such as continuous motions and fingerspelling. Through this project, we aim to integrate machine learning models with visual capture technology to develop an interactive system capable of real-time recognition of sign language alphabets. This system is designed to facilitate communication between the general public and sign language users, while also enhancing the accessibility and convenience of learning sign language.<br>

資料來源與標註
## Dependencies
opencv-python<br>
mediapipe<br>
scikit-learn intelex<br>
Numpy<br>
Pickle<br>
## To run signlanguage_recognition
~~~~
$pip install scikit-learn intelex
$pip install opencv-python
$pip install mediapipe
$git clone https://github.com/vatika17/signlanguage_recognition.git
$cd signlanguage_recognition
$python model_test.py
~~~~
# Demo
https://youtu.be/168r68b_yfM
