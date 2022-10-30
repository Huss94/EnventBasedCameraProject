function mask=bitmask(Ver)

%/***********************************************************************
% *     PROJECT: Silicon Retina
% *   COPYRIGHT: Austrian Research Centers GmbH - ARC, Department S&S
% *    $RCSfile: bitmask.m,v $
% *   $Revision: 1.4 $
% *      AUTHOR: M.Litzenberger
% *  SUBSTITUTE: 
% *    LANGUAGE: Matlab
% *
% * DESCRIPTION: mask = bitmask([Version])
%                Returns a bitmask for address-event conversion.
%                The bitmasks are stored in UINT32 values.
%
% *    PARAMETER
% *          IN:
%               1  .. old bitmask for 64x64 tmpdiff imager 
%               2  .. new bitmask for the 64x64 tmpdiff imager --> Default 
%               3  .. testchip line sensor
%               128.. 128x128 tmpdiff sensor
%               0 ..  128x128 tmpdiff sensor for vidcenter recorded data
%               10 .. ATIS (or up to 12 bit for x- and y-address)
%                     MSB: P: polarity (P=0: on event, P=1: off event)
%               11 .. ATIS (or up to 
%                     12 bit for x- and y-address
%                     MSB: P: polarity (P=0: on event, P=1: off event)
%                     MSB-1: T: TD or APS event (T=0: TD, T=1: APS)
%
%
% *         OUT:
%               mask.x         x-address bitmask
%               mask.y         y-address bitmask
%               mask.pol       polarity bit bitmask (ON/OFF for tmpdiff, threshH/L for grayscale event)
%               mask.typ       event type indentifier (ATIS tmpdiff/grayscale event)
%               mask.unused    unused bits bitmask (optional)
% *
% ************************************************************************/

% ************************************************************************/
% * $Log: bitmask.m,v $
% * Revision 1.4  2009/03/05 10:15:10  litzenbergerm
% * changed company name in file header
% *
% * Revision 1.3  2009/02/26 14:33:44  wohlgenanntr
% * added general (and ATIS) bitmask (parameter 10): 12 bit for x and y; additional TD/APS flag for events
% *
% * Revision 1.2  2006/02/08 12:16:52  litzenbergerm
% * new bitmask for vidcenter recorded data from 128^2 sensor
% *
% * Revision 1.1  2005/12/21 11:58:41  litzenbergerm
% * Migration from VSS to CVS. Use VSS to retrive all earlier versions of this file.
% *
% 
% 7     22.11.05 14:42 Litzenbergerm
% new bitmask for 128x128 pixel tmpdiff imager
% 
% 6     7.10.05 9:33 Litzenbergerm
% new bitmask for testchip line sensor (Ver=3)
% 
% 5     16.02.05 16:09 Litzenbergerm
% added standard header
% *
% ************************************************************************/

if nargin == 0; Ver = 11; end

switch Ver
case 0
    % 128^2 tmpdiff sensor for vidcenter recorded data
                      %3         2         1        
                      %0987654321098765432109876543210
    mask.x=     uint32(bin2dec(      '0000000001111111')); 
    mask.y=     uint32(bin2dec(      '0111111100000000')); 
    mask.pol=   uint32(bin2dec(      '0000000010000000')); 
    mask.unused=uint32(bin2dec(      '1000000000000000')); 
case 128
    % 128^2 tmpdiff sensor:
                      %3         2         1        
                      %0987654321098765432109876543210
    mask.x=     uint32(bin2dec(      '0000000011111110')); 
    mask.y=     uint32(bin2dec(      '0111111100000000')); 
    mask.pol=   uint32(bin2dec(      '0000000000000001')); 
    mask.unused=uint32(bin2dec(      '1000000000000000')); 
    
case 2
    % NEW VERSION:
                      %3         2         1        
                      %0987654321098765432109876543210
    mask.x=     uint32(bin2dec(      '0000000000111111')); %%  Hex: 3F
    mask.y=     uint32(bin2dec(      '0011111100000000')); %%  Hex: 3F00
    mask.pol=   uint32(bin2dec(      '0000000010000000')); %%  Hex: 80
    mask.unused=uint32(bin2dec(      '0000000001000000')); %%  Hex: 40

 case 3
    % ctc-itn2005-1 line sensor
                      %3         2         1        
                      %0987654321098765432109876543210
    mask.x=     uint32(bin2dec(      '0000000000111111')); %%  Hex: 3F
    mask.unused=uint32(bin2dec(      '1111111100000000')); %%  Hex: 3F00
    mask.pol=   uint32(bin2dec(      '0000000010000000')); %%  Hex: 80
    mask.y=     uint32(bin2dec(      '0000000001000000')); %%  Hex: 40

 case 10
    % ATIS  - only TmpDiff events
    mask.x=     uint32(hex2dec('00fff000')); %% 00000000111111111111000000000000
    mask.unused=uint32(hex2dec('7f000000')); %% 01111111000000000000000000000000
    mask.pol=   uint32(hex2dec('80000000')); %% 10000000000000000000000000000000
    mask.y=     uint32(hex2dec('00000fff')); %% 00000000000000000000111111111111
 
 case 11   
    % ATIS  - both event types
    mask.y=     uint32(hex2dec('00000fff')); %% 00000000000000000000111111111111
    mask.x=     uint32(hex2dec('00fff000')); %% 00000000111111111111000000000000
    mask.unused=uint32(hex2dec('3f000000')); %% 00111111000000000000000000000000
    mask.pol=   uint32(hex2dec('80000000')); %% 10000000000000000000000000000000
    mask.type=   uint32(hex2dec('40000000')); %% 01000000000000000000000000000000
    
    
    otherwise
    % OLD VERSION:
                      %3         2         1        
                      %0987654321098765432109876543210
    mask.x=     uint32(bin2dec(      '0000000001111110')); %%  
    mask.y=     uint32(bin2dec(      '0011111100000000')); %%  
    mask.pol=   uint32(bin2dec(      '0000000000000001')); %%  
    mask.unused=uint32(bin2dec(      '0000000010000000')); %%  
    
end

return