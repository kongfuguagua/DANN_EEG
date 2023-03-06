clc;clear;
fid_in='1';
fid=fopen(['.\',fid_in,'\','train.txt'],'w');%to creat train dataset label file
fid_test=fopen(['C:\Users\ASUS\Desktop\cwt1\',fid_in,'\','test.txt'],'w');%to creat test dataset label file

%A01T, load preprocessed train dataset
setFileName = ['A0',fid_in,'T_ALLCLASS.set'];
filePath =[ '.\',fid_in,'','\'];eeg = pop_loadset(setFileName, filePath);sample = eeg.data;
load(['.\A0',fid_in,'T.mat']);%load label file

Fs = 250; % sampling frequency
T = 1/Fs;
spec={};
L=length(sample);
t=1:L;
N = 2^nextpow2(L);%N>=L
num=1;

%Generate Train Dataset
for event=1:288%event is class1-4(left hand right hand tongue foot)
    for channel=1:5%channel (fz pz c3 c4 cz)
        y=double(sample(channel,1:L,event));
        [wt,f,coi] = cwt(y,'morse',Fs);
        spec{channel,1}=abs(wt);
    end
    pho1=[spec{1,1};spec{2,1};spec{3,1};spec{4,1};spec{5,1}];%crop and joint
    set(figure,'visible', 'off');%no visualization
    imagesc(pho1);
    set( gca, 'XTick', [], 'YTick', [] );
    F1=getframe;
    changepho=imresize(F1.cdata,'OutputSize',[96,96]);
    imwrite(changepho,['.\1\',fid_in,'\morse\',num2str(num),'.png']);
    fprintf(fid,'%s %d\n',[num2str(num) '.png'],classlabel(event));%write label
    num=num+1
    close all
end

%A01E, load preprocessed test dataset
setFileName = ['A0',fid_in,'E_ALLCLASS.set'];
filePath = ['.\',fid_in,'','\'];eeg = pop_loadset(setFileName, filePath);sample = eeg.data;
load(['.\A0',fid_in,'E.mat']);

for event=1:288
    for channel=1:5
        y=double(sample(channel,1:L,event));
        [wt,f,coi] = cwt(y,'morse',Fs);
        spec{channel,1}=abs(wt);
    end
    pho1=[spec{1,1};spec{2,1};spec{3,1};spec{4,1};spec{5,1}];
    set(figure,'visible', 'off');
    imagesc(pho1);
    set( gca, 'XTick', [], 'YTick', [] );
    F1=getframe;
    changepho=imresize(F1.cdata,'OutputSize',[96,96]);
    imwrite(changepho,['C:\Users\ASUS\Desktop\cwt1\',fid_in,'\morse\',num2str(num),'.png']);
    if num>504
        fprintf(fid_test,'%s %d\n',[num2str(num) '.png'],classlabel(event));
    else
        fprintf(fid,'%s %d\n',[num2str(num) '.png'],classlabel(event));
    end
    num=num+1
    close all
end