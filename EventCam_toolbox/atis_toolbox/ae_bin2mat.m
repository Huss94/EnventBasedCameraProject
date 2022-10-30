function ae = ae_bin2mat( fn ,nb_Events)

%/***********************************************************************
% *     PROJECT: Silicon Retina
% *   COPYRIGHT: Austrian Research Centers GmbH - ARC, Department S&S
% *    $RCSfile: ae_bin2mat.m,v $
% *   $Revision: 1.3 $
% *      AUTHOR: R.Wohlgenannt
% *  SUBSTITUTE: N.Donath
% *    LANGUAGE: Matlab
% *
% * DESCRIPTION: ae = ae_bin2mat( fn )
% *
% * Reads binary file with raw AE data and converts to matlab array in
% * 3x32bit format (CAVIAR format).
% *
% ************************************************************************/

% ************************************************************************/
% * $Log: ae_bin2mat.m,v $
% 
% * Revision 1.3.1 2013/02/25 11:26:00 clady
% * *** new argin : max of events to read ****
%
% * Revision 1.3  2010/01/27 15:32:27  wohlgenanntr
% * *** empty log message ***
% *
% * Revision 1.2  2009/03/05 10:15:10  litzenbergerm
% * changed company name in file header
%*
% ************************************************************************/

if nargin==1,
    maxEvents=50000000;
elseif nargin==2,
    maxEvents=nb_Events;
else
    disp('Bad use of ae_bin2mat() : ');
    help ae_bin2mat
    exit(1);
end;
    


% liest ein Binärfile mir rohen Address-Events ein und konvertiert in eine Matrix (ETH-Format)
    fid =fopen(fn, 'rb');
    [buf, cnt] = fread(fid, 50000000, 'uint32');
    ae=reshape(buf, 2, cnt/2);
    clear buf;
    
    ae=double(ae);
    %ae(2,:)=ae(2,:)-ae(2,1);
    tmp=find(0>ae(2,:));
    if ~isempty( tmp )
        ae(2,tmp)=ae(2,tmp)+2^32;
    end;
    clear tmp;
    %ae(2,:)=ae(2,:)*1e3;
    ae=uint32(ae);
    fclose(fid);

return;