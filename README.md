# DANN_EEG
HIT undergraduate programs include: BCI IV 2a and 2b dataset, EEGLAB preprocessing code, Matlab CWT code, CNN, Fine-tune, DANN, DANN+attention


matlab code includes preprocess and cwt to remove artifact and generate dataset.If you want to generate,you can follow that first run eeglabhist,then open gained '.set' with eeglab and select channel 'pz fz c3 c4 cz' and remove artifact and save it, run cwt_2a, final you gain train set and test set.


python includes CNN, Fine-tune and DANN for tranfer learning. The filename suffix with _4 means four classification task that asks BCI IV 2a as input.
