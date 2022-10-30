function [AE,filename]=loadbindat_xc(file,max_Events)
%function [AE,filename]=loadbindat_xc(file,max_Events);
% loads events from a .bin file.
% version xavier clady : reply the filename...
% 

if nargin==0,
    [filename,path,filterindex]=uigetfile('*.bin','Select recorded ATIS data file');
    if filename==0, return; end
    maxEvents=30e6;
elseif nargin==1,
    path='';
    filename=file;
    maxEvents=30e6;
elseif nargin==2,
    if isempty(file), 
            [filename,path,filterindex]=uigetfile('*.bin','Select recorded ATIS data file');
    end
    maxEvents=max_Events;
else
    disp('Error using loadbindat_xc');
    help loadbindat_xc
    exit(1);
end

ae=ae_bin2mat(fullfile(path,filename),maxEvents);
AE=eth2itn(ae,bitmask(11));
