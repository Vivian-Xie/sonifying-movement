# Sonfying Movement
This project is composed of three parts, [emotion from movement](https://github.com/Vivian-Xie/sonfying-movement/tree/main/Emotion-detection-main), [rhythm from movement](https://github.com/Vivian-Xie/sonfying-movement/tree/main/video-bgm-generation-main/video-bgm-generation-main), and [music generation with emotion and rhythm](https://github.com/Vivian-Xie/sonfying-movement/tree/main/music%20generation).


# Music Generation with Emotion and Rhythm
To run this file, we need to first create a file in ec2vae named model_param and add the following file downloaded from google drive in model_param.
https://drive.google.com/file/d/16egAtz7rPRhU_2dv2EFX4us-EbRMvYSA/view?usp=sharing


# Server
This project needs to be run on several computers and a [server](https://github.com/Vivian-Xie/sonfying-movement/blob/main/server.py) is needed to transmit information among computers. The server sends emotion and rhythm every 3.75 seconds to the end of music generation.
