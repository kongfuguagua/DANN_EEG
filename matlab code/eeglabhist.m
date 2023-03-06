% EEGLAB history file generated on the 21-Feb-2023
% ------------------------------------------------
eeg_file_name='1T';
EEG.etc.eeglabvers = '2022.1'; % this tracks which version of EEGLAB is being used, you may ignore it
EEG = pop_biosig(['.\A0',eeg_file_name,'.gdf']);%file address e.g. '.\A01T.gdf'
EEG.setname='7eGDF file';
EEG = eeg_checkset( EEG );
EEG=pop_chanedit(EEG, 'lookup','.\my_loc.ced');%channel location file address
EEG = eeg_checkset( EEG );
EEG = pop_select( EEG, 'nochannel',{'EOG-left','EOG-central','EOG-right'});%remove useless channel
EEG = eeg_checkset( EEG );
EEG = pop_eegfiltnew(EEG, 'locutoff',4.5,'hicutoff',30,'plotfreqz',1);%filter
EEG = eeg_checkset( EEG );
if eeg_file_name(2)=='E' %extract event (epoch)
    EEG = pop_epoch( EEG, {'cue unknown/undefined (used for BCI competition) '}, [0  3], 'newname', '7eGDF file epochs', 'epochinfo', 'yes');
elseif eeg_file_name(2)=='T'
    EEG = pop_epoch( EEG, { 'class1, Left hand	- cue onset (BCI experiment)'  'class2, Right hand	- cue onset (BCI experiment)'  'class3, Foot, towards Right - cue onset (BCI experiment)'  'class4, Tongue		- cue onset (BCI experiment)'}, [0  3], 'newname', '7eGDF file epochs', 'epochinfo', 'yes');
end
EEG = eeg_checkset( EEG );
EEG = pop_rmbase( EEG, [],[]);
EEG = eeg_checkset( EEG );
EEG = eeg_checkset( EEG );
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'pca',15,'interrupt','on');%ICA 15 means getting 15 components
EEG = eeg_checkset( EEG );
EEG = pop_iclabel(EEG, 'default');
EEG = eeg_checkset( EEG );
EEG = pop_saveset( EEG, 'filename',['A0',eeg_file_name,'_ALLCLASS.set'],'.\\',eeg_file_name(1),'','\\']);
EEG = eeg_checkset( EEG );
