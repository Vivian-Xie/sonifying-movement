# Sonfying Movement
This project is composed of three parts, [emotion from movement](https://github.com/Vivian-Xie/sonfying-movement/tree/main/Emotion-detection-main), [rhythm from movement](https://github.com/Vivian-Xie/sonfying-movement/tree/main/video-bgm-generation-main/video-bgm-generation-main), and [music generation with emotion and rhythm](https://github.com/Vivian-Xie/sonfying-movement/tree/main/music%20generation).
![demo](https://github.com/Vivian-Xie/sonfying-movement/blob/main/flow_chart.png)
The main idea of this project is to extract elements from the video capturing the user’s reactions when he/she listens to a piece of music, and adjust the music chord and rhythm with the movement elements to make the user feel in control of the body and the music.
Technically, our project has two main parts, movement analysis and music generation. After a lot of research, reading, and practice, and with the help of professors, we successfully realized real-time analysis of movement emotion, as well as non-real-time analysis of movement rhythm and strength. In order to realize an integrated project, we finally decided to adopt all the non-real-time results as phased results. In terms of music, we adopted the EC2VAE model to adjust the chord, rhythm, and intensity of the original song. We use emotion as the input for the chord, and the output of visbeat model as the input of the rhythm. After inputting the two parameters, we would get a series of lists of numbers as the output music. We use Pygame to play the music flow. Until now, what we can achieve is that by inputting a video of movement with original music, we could get a new piece of music with adjusted chord, rhythm, and volume in accordance with the user’s movement.


# Music Generation with Emotion and Rhythm
To run this file, we need to first create a file in ec2vae named model_param and add the following file downloaded from google drive in model_param.
https://drive.google.com/file/d/16egAtz7rPRhU_2dv2EFX4us-EbRMvYSA/view?usp=sharing


# Server
This project needs to be run on several windows and a [server](https://github.com/Vivian-Xie/sonfying-movement/blob/main/server.py) is needed to transmit information among computers. The server sends emotion and rhythm every 3.75 seconds to the end of music generation.

