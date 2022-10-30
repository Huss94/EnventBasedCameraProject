function [i,j,type,Tps,filename]=mise_en_memoire(file,path,type_camera,nb_Events)
% mise en memoire d'une nouvelle sequence
% 
% optional IN_arguments: 
% * type_camera:
%                       'dvs' (default)
%                       'atis'
% * nb_Events:
%               maximum of events to extract (works only with atis camera)
%
% TODO : change the name of function (-> english)
%        recognize automatically the type of camera (with the type of file ?)
%        max of Events -> dvs
%        size of retina matrix (!?) dvs=128x128 
%                                  atis= max(j) x max(i) (or 240x320) 
% Modified by Adam Heriban, heriban[at]crans[dot]org


pwd;


addpath ./aer_toolbox/
addpath ./aer_toolbox/cochlea/
addpath ./aer_toolbox/retina/
addpath ./aer_toolbox/monitor_sequencer/
addpath ./atis_toolbox/


if nargin<3,
    type_camera='dvs';
elseif nargin==3 && strcmp(type_camera,'atis'),
    nb_Events=5000000;
end
   
if  strcmp(type_camera,'dvs'),
    if nargin>1
        [Add,Tps,filename]=loadaerdat_xc(file,path);
    end
    if nargin<1
        [Add,Tps,filename]=loadaerdat_xc();
    end
    %Add : adresse brute et Tps=timestamp (nanosecondes).
    
    disp('read DVS data : OK...')
    
    %conversion en format lisible:
    [x,y,type]=extractRetina128EventsFromAddr(Add);
    %x=colonne, y=ligne, pol= polarite.
    
    disp('convert DVS data : OK...')
    
    i=128-y;
    j=x+1;
    
elseif  strcmp(type_camera,'atis'),
    
    [AE,filename]=loadbindat_xc([],nb_Events);
        
    %gl
    [~,~,evt_gl]=GL_slice(AE,1,length(AE.t),1e-2,1);
    %dvs
    evt_dvs=Evt_extract(AE,1);
    
    nb_max_col=max(320,max(double(evt_dvs.x)));
    
    j=nb_max_col-double(evt_dvs.x)';
    i=double(evt_dvs.y)';
    Tps=double(evt_dvs.t');

    format long;
    disp(['-> ATIS : dt= ',num2str(Tps(1))]);
    disp(['-> ATIS : dt= ',num2str(Tps(2))]);
    
    
    if (Tps(1)<1)
        Tps = (double(Tps))*10^-3*10^6; % ?
    else
        Tps = (double(Tps))*10^-4*10^6; % ?        
    end
    type=evt_dvs.source'; % comment r�cup�rer gl ? GL_slice deconne !!
    
end
