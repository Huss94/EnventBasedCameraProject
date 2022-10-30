function aeitn=eth2itn(aeeth, mask)

%/***********************************************************************
% *     PROJECT: Silicon Retina
% *   COPYRIGHT: Austrian Research Centers GmbH - ARC, Department S&S
% *    $RCSfile: eth2itn.m,v $
% *   $Revision: 1.6 $
% *      AUTHOR: M.Litzenberger
% *  SUBSTITUTE: N.Donath
% *    LANGUAGE: Matlab
% *
% * DESCRIPTION: AEitn = eth2itn(AEeth [,bitmask])
%
%   Convert AE-stream from ETH's (2 x N array of uint32 ) to ARCS's format.
%   By default bitmasks for the 64x64 pixel sensor geometry are used.
%   Other bitmasks can by given in the bitmask variable.
%
% PARAMETER
% IN:
% AEitn             address event stream in ETH (INI) format (array of uint32)
%
% bitmask.x         x-address bitmask
% bitmask.y         y-address bitmask
% bitmask.pol       polarity bit bitmask (ON/OFF for tmpdiff, threshH/L for grayscale event)
% bitmask.typ       event type indentifier (ATIS tmpdiff/grayscale event)
% bitmask.unused    unsed bits bitmask (optional)
%
% OUT:
% AEitn.x(..)       array of pixel x-coordinates UINT16
% AEitn.y(..)       array of pixel y-coordinates UINT16   
% AEitn.source(..)  array of pixel source (=polarity for transient sensor) UINT16
% AEitn.t(..)       array of delta-times since last event in units of dt ;UINT32
% AEitn.tt(..)      array of absolute times in units of seconds ;DOUBLE
% AEitn.dt          time base in s  (default is 1 us) 
% *
% ************************************************************************/

% ************************************************************************/
% * $Log: eth2itn.m,v $
% * Revision 1.6  2009/03/05 10:15:10  litzenbergerm
% * changed company name in file header
% *
% * Revision 1.5  2009/02/26 14:35:23  wohlgenanntr
% * added time resolution detection and adjustment for ATIS/GAEP (10ns) (depending on bitmask)
% *
% * Revision 1.4  2006/04/07 08:47:35  litzenbergerm
% * Reads also ms time resolution files (but no date conversion yet).
% *
% * Revision 1.3  2006/04/06 13:35:54  litzenbergerm
% * accept small neg. time steps
% *
% * Revision 1.2  2006/04/05 14:55:05  litzenbergerm
% * now handles time stamp overflows
% *
% * Revision 1.1  2005/12/21 11:58:41  litzenbergerm
% * Migration from VSS to CVS. Use VSS to retrive all earlier versions of this file.
% * 
% 
% 5     22.11.05 14:45 Litzenbergerm
% removed for loop, makes Matlab processing faster
% 
% 4     16.02.05 17:13 Litzenbergerm
% added standard header, corrected T0 bug
% ************************************************************************/

MAX32 = 2^32;
TRESMASK = bin2dec('1000 0000 0000 0000 0000 0000 0000 0000'); 

if nargin < 2;
    mask = bitmask;
end

N = length(aeeth(1,:));

%alloc mem for AE-stream, makes matlab faster
aeitn.t=uint32(zeros(1,N));
aeitn.x=uint16(zeros(1,N));
aeitn.y=uint16(zeros(1,N));
aeitn.source=uint16(zeros(1,N));
aeitn.type=uint16(zeros(1,N));


% handle timestamp overflow(s)
DT = diff(double(aeeth(2,:)));
% accept small negative time increments
ii=find(DT < -MAX32/2);
if length(ii) > 0;
    DT(ii) = DT(ii) + MAX32;
    fprintf('WARNING: timestamp overflow detected at index %i in AE-data!\n',ii);
end

% calc. delta time stamps
aeitn.t=uint32([double(aeeth(2,1)) DT] );
aeitn.t(1)=0;

% test for time resolution in binary AER data:

% is it an ATIS ?
if((mask.x == 16773120) && (mask.y == 4095)) 
% yes, then time resolution equals GAEP time resolution (10ns)
    aeitn.dt = 1e-8;    
else
% no    
    if bitand(aeeth(1,1), TRESMASK) == 1;
        aeitn.dt=1e-3;
    else
        aeitn.dt=1e-6;
    end
end

aeitn.t=double(aeeth(2,:)) * aeitn.dt; %in sec

xshift = -bitlsb(mask.x);
yshift = -bitlsb(mask.y);
polshift = -bitlsb(mask.pol);
typshift = -bitlsb(mask.type);

aeitn.x   =  uint16( double(bitshift(bitand (aeeth(1,:), mask.x), xshift  )) + 1);
aeitn.y   =  uint16( double(bitshift(bitand (aeeth(1,:), mask.y), yshift  )) + 1);
aeitn.source=uint16( bitshift(bitand (aeeth(1,:), mask.pol), polshift ));
aeitn.type=uint16( bitshift(bitand (aeeth(1,:), mask.type), typshift ));

return;
% eof
