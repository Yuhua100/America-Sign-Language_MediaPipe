#American Sign Language Recognition using MediaPipe and Random Forest Classifier (Detectron2、Bayesian classifier)

## 研究背景與動機 Research Background and Motivation
隨著多元溝通方式的需求上升，手語作為聽障人士與社會溝通的重要橋梁，其辨識技術也逐漸受到重視。然而，多數現有系統僅支援靜態手勢識別，無法處理實際溝通中常見的「連續動作」與「單字拼寫」。我們希望透過本專題，整合機器學習模型與視覺擷取技術，打造一套能即時辨識手語英文字母的互動式系統，協助一般人與手語使用者溝通，並提升手語學習的便利性。<br>
As the demand for diverse communication methods continues to rise, sign language has become an essential bridge for communication between the hearing-impaired community and society. However, most existing systems only support static gesture recognition and are unable to handle commonly used features in real-life communication, such as continuous motions and fingerspelling. Through this project, we aim to integrate machine learning models with visual capture technology to develop an interactive system capable of real-time recognition of sign language alphabets. This system is designed to facilitate communication between the general public and sign language users, while also enhancing the accessibility and convenience of learning sign language.<br>

##資料來源與標註 Data Sources and Annotation
第一 First<br>
資料來源：Roboflow - American Sign Language Letters (Object Detection)<br>
https://public.roboflow.com/object-detection/american-sign-language-letters/1/download<br>
資料格式：COCO JSON（含 bounding box 與標籤<br>
標註類別數：24 類（A–Y，不含 J 與 Z，因這兩個手勢涉及動態動作）<br>
標註工具：Roboflow 平台<br>
Data Source: Roboflow – American Sign Language Letters (Object Detection)<br>
https://public.roboflow.com/object-detection/american-sign-language-letters/1/download<br>
Data Format: COCO JSON (including bounding boxes and labels)<br>
Number of Classes: 24 classes (A–Y, excluding J and Z as they involve motion)<br>
Annotation Tool: Roboflow platform<br>

第二 Second<br>
資料來源：Kaggle – Sign Language MNIST<br>
https://www.kaggle.com/datasets/datamunge/sign-language-mnist<br>
資料格式：CSV 檔案（每一筆資料為 28×28 像素的灰階圖像，數值為像素值<br>
標註類別數：24 類（A–Y，不含 J 與 Z，因這兩個手勢涉及動態動作）<br>
標註方式：每張圖片皆對應一個英文字母的標籤，已在 CSV 中標註完成<br>
Data Source: Kaggle – Sign Language MNIST<br>
https://www.kaggle.com/datasets/datamunge/sign-language-mnist<br>
Data Format: CSV files (each entry is a 28×28 grayscale image represented by pixel values)<br>
Number of Classes: 24 classes (A–Y, excluding J and Z as they involve motion)<br>
Annotation Method: Each image is labeled with a corresponding English letter, embedded in the CSV file<br>

第三 Third<br>
This project references code and ideas from the open-source repository:  
🔗 [vatika17/signlanguage_recognition](https://github.com/vatika17/signlanguage_recognition)
All credits go to the original author(s).


## Dependencies on Windows
opencv-python<br>
mediapipe<br>
scikit-learn intelex<br>
Numpy<br>
Pickle<br>
## Dependencies on Mac M1/M2
opencv-python<br>  
mediapipe<br>  
scikit-learn<br>  
Numpy<br>  
Pickle<br>


# Demo

