# Data-analysis-on-EEG-investigations-of-human-touch
This repository contains data analysis structures for EEG recordings of human touch. Python 2.7 has been used.

ABSTRACT

The investigations on this project aim to analyse Electroencephalographic (EEG) data regarding human touch. A study of the neural correlates of attention to touch was carried out about some experiments based on a variation of the Posner’s cue-target paradigm, with the aim of clarifying the nature of relation between endogenous and exogenous attention. The design and development of a data structure which allows the organised execution of multiple test was carried out. This granted the implementation of several analysis tools, such as frequency and time analysis, comparing measuring locations and experimental conditions. Laterality studies to check the relative contribution of the two hemispheres in different stages of the experiments and conditions were alse performed. Finally, a multi-class classification algorithm based on state of the art machine learning tools was implemented to predict experimental condition on recorded data. 

DATA CHARACTERISTICS

The data consists of a series of trials collected from the experiment based on a variant of the Posner paradigm. Each trial contains information on neuronal activation before a tactile stimulus, collected through four pair of electrodes located in and around different regions of the somatosensory area. It also contains the response time that each subject has needed to respond to said stimuli.
The trials can be distinguished among them according to certain parameters, which should be considered when exporting the data to the Python’s programming environment. Those parameters have been used as labels, allowing the distinction between different trials, which has been very useful not only when it comes to import the data, but also later while analysing them. The parameters are the following: subjects subdued to the experiments, electrode measuring each trial, task type (endogenous predictive, endogenous counter-predictive or exogenous), side of appearence of the target, attended or unattended location and laterality information.

STUDY 1: Analysis of endogenous and exogenous alpha oscillations in touch. 

STUDY 2: Relationship between Event Related Potentials (ERPs) and subjects' behavioural outcome.

STUDY 3: Classification of the alpha oscilattions before different attentional scenarios.
