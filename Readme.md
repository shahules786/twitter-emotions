[![](https://img.shields.io/github/issues/shahules786/twitter-emotions)]()
[![](https://img.shields.io/github/license/shahules786/twitter-emotions)]()
[![](https://img.shields.io/github/stars/shahules786/twitter-emotions)]()


# Emotional phrase extractor


<p align="center">
  <img src="https://user-images.githubusercontent.com/25312635/118586138-d4adf300-b7b7-11eb-9dcd-99b9cf9d4236.png" />
</p>


Extract phrase in the given text that is used to express the sentiment. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? This project aims to solve this problem.

## Try it out.


```bash

git clone https://github.com/shahules786/twitter-emotions.git

cd twitter-emotions

sudo docker build --tag twitter-emotions:api .

sudo docker run -p 9999:9999  -it twitter-emotions:api python twitteremotions/app.py


```

Server will start running on port 9999 of localhost


## Example



<p align="center">
  <img src="https://user-images.githubusercontent.com/25312635/119227824-7a04f600-bb2d-11eb-8c01-d33a12b32186.gif" />
</p>



