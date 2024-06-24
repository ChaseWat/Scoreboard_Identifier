# An object locator trained on MLB game screenshots to identify and locate the scoreboard.
This object locator was trained to identify the location of a scoreboard from a screenshot of a Major League Baseball game and output its location for text analysis purposes. 

## Requirements
- tensorflow
- numpy 
- glob
- pandas
- xml
- sklearn
- matplotlib
- PIL
- pathlib
  
## Train and Test
dowload the annotated dataset and the pretrained model from https://drive.google.com/drive/folders/1oX6slGWSIVCVSrakMl0sPAEFii5RbFcT?usp=sharing

## Model Architecture
<div align="center">
    <img src="https://github.com/ChaseWat/Scoreboard_Identifier/blob/main/app_01_model.png" width="200">
</div>

## Model Output
A green box is drawn around the scoreboard based on the vertices identified by the obeject locator. 
<div align="center">
    <img src="https://github.com/ChaseWat/Scoreboard_Identifier/blob/main/OutputExample.PNG" width="1080">
</div>
