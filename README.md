# TrQA
GPU is needed to run this code.

This is a PyTorch implementation of the TrQA system described based on TrQA and Transformers.
```angular2
git clone https://github.com/erenup/Trqa.git
cd Trqa
python setup.py develop
```
Download Data and Models from [pretrained models and database](https://drive.google.com/open?id=1tilewF8o4OufQ3WjaoHZ312btigbWAry) and unzip to Trqa/data

Run `python scripts/pipeline/interactive-transformers.py` to drop into an interactive session. For each question, the top span and the Wikipedia paragraph it came from are returned.

```
>>> process('Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.')
Top Predictions:
+------+---------------------+----------------------+--------------+-----------+
| Rank |        Answer       |         Doc          | Answer Score | Doc Score |
+------+---------------------+----------------------+--------------+-----------+
|  1   |      Badr Hari      |      Badr Hari       |   0.96028    |   541.09  |
|  2   |                     |      Badr Hari       |   0.99999    |   541.09  |
|  3   |    Zabit Samedov    |      Badr Hari       |   0.98162    |   541.09  |
|  4   |      Mayweather     | Floyd Mayweather Jr. |   0.98801    |   209.69  |
|  5   | Sugar Ray Leonard", | Floyd Mayweather Jr. |   0.36359    |   209.69  |
+------+---------------------+----------------------+--------------+-----------+

Contexts:
[ Doc = Badr Hari ]
Badr Hari (; born 8 December 1984) is a Moroccan-Dutch super heavyweight kickboxer from Amsterdam, 
fighting out of Mike's Gym in Oostzaan. He is a former K-1 Heavyweight champion (2007—2008), 
It's Showtime Heavyweight world champion (2009-2010) and "K-1 World Grand Prix 2009" finalist. 
Hari has been a prominent figure in the world of kickboxing and considered one of the best kickboxers in the world, 
however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.

[ Doc = Badr Hari ]
Aside from the sport, he has been arrested multiple times since 2006 on various assault charges.

[ Doc = Badr Hari ]
Having been released from detention, Hari was given the chance to fight in the "K-1 World Grand Prix 2012 Final" in Zagreb,
Croatia on 15 March 2013 when Ben Edwards withdrew. He rematched Zabit Samedov in the quarter-finals. Hari scored a knockdown early in round one and forced a standing eight count in three, 
after which he simply jogged away from Samedov for the rest of the fight while the Azerbaijani taunted him. He won by unanimous decision but injured his foot in the bout and was forced to bow out of the tournament.

```